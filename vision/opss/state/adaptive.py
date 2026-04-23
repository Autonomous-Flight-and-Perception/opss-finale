"""
Adaptive multi-filter tracker.

Runs Kalman, CTRV-UKF, and Particle Filter on the SAME detection stream
and picks one of their outputs per-tick based on a lightweight motion
classifier. The choice is exposed in the per-state ``track_id`` extra
metadata via the ``ObjectState.bbox`` ``filter_used`` field (so the
dashboard / cobot can see which filter was authoritative for that tick).

Selection policy (per-track, per-tick):
  - **Detection lost recently** (k of last N ticks unmatched) → Particle
    filter. PF carries multi-modal uncertainty cleanly across YOLO drops.
  - **Linear motion** (low recent yaw rate, low Kalman innovation) →
    Kalman. Constant-velocity model is unbiased and minimum-variance.
  - **Nonlinear motion** (high yaw rate or sustained high innovation) →
    CTRV UKF. Constant-turn model handles turning targets.

The selector tracks per-track stats internally; the underlying filters
are not aware of the switching — they all run every tick so when the
selector hands off, the chosen filter is already up to date.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .kalman import MultiObjectKalmanFilter, ObjectState, create_tracker
from .ctrv_ukf_tracker import MultiObjectCTRVUKF, create_ctrv_ukf_tracker
from .pf_tracker import MultiObjectParticleFilter, create_pf_tracker


@dataclass
class AdaptiveConfig:
    # Number of recent ticks to consider when deciding linearity vs nonlinearity.
    history_window: int = 10
    # Yaw-rate threshold (rad/s) above which motion is "nonlinear" — favors UKF.
    yaw_rate_threshold: float = 0.6
    # Innovation threshold (px) above which Kalman is struggling — favors UKF.
    innovation_threshold_px: float = 25.0
    # If raw YOLO has been missing for >= this many of the last `history_window`
    # ticks, fall back to PF (it tolerates dropouts best).
    miss_to_pf_threshold: int = 3


class _PerTrackSelector:
    """One selector per primary track. Keeps short rolling history."""

    def __init__(self, cfg: AdaptiveConfig):
        self.cfg = cfg
        self.miss_history: Deque[bool] = deque(maxlen=cfg.history_window)
        self.yaw_rate_history: Deque[float] = deque(maxlen=cfg.history_window)
        self.innovation_history: Deque[float] = deque(maxlen=cfg.history_window)
        self.last_choice: str = "kalman"

    def record(self, missed: bool, yaw_rate: float, innovation_px: float) -> None:
        self.miss_history.append(bool(missed))
        self.yaw_rate_history.append(abs(float(yaw_rate)))
        self.innovation_history.append(abs(float(innovation_px)))

    def choose(self) -> str:
        cfg = self.cfg
        # Recent dropout pressure → PF
        if sum(self.miss_history) >= cfg.miss_to_pf_threshold:
            self.last_choice = "pf"
            return "pf"
        # Sustained turning → UKF
        if self.yaw_rate_history:
            avg_yaw = sum(self.yaw_rate_history) / len(self.yaw_rate_history)
        else:
            avg_yaw = 0.0
        if self.innovation_history:
            avg_inn = sum(self.innovation_history) / len(self.innovation_history)
        else:
            avg_inn = 0.0
        if avg_yaw > cfg.yaw_rate_threshold or avg_inn > cfg.innovation_threshold_px:
            self.last_choice = "ukf"
            return "ukf"
        self.last_choice = "kalman"
        return "kalman"


class MultiObjectAdaptive:
    """
    Composite tracker that exposes the multi-object interface but
    internally maintains three independent trackers (Kalman, CTRV-UKF,
    PF). On every ``update``, all three run; per-track output is then
    drawn from whichever filter the selector picks for that track.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_tracks: int = 50,
        cfg: Optional[AdaptiveConfig] = None,
    ):
        self.cfg = cfg or AdaptiveConfig()
        self.kf = create_tracker(max_distance=max_distance, max_tracks=max_tracks)
        self.ukf = create_ctrv_ukf_tracker(max_distance=max_distance, max_tracks=max_tracks)
        self.pf = create_pf_tracker(max_distance=max_distance, max_tracks=max_tracks)
        self._selectors: Dict[int, _PerTrackSelector] = {}
        self._last_detected_track_ids: set = set()

    @property
    def trackers(self) -> Dict:
        # Aggregate so pipeline.stats["tracks_active"] reports something
        # sensible. Use the Kalman side as the canonical track table since
        # all three filters spawn from the same detection stream and the
        # IDs roughly align.
        return self.kf.trackers

    def update(self, detections: List[Dict], timestamp: float) -> List[ObjectState]:
        # Drive all three filters with the same detection batch.
        kf_states = self.kf.update(detections, timestamp)
        ukf_states = self.ukf.update(detections, timestamp)
        pf_states = self.pf.update(detections, timestamp)

        kf_by_id = {s.track_id: s for s in kf_states}
        # The three trackers don't share track IDs (they assign IDs
        # independently). For the picker we match across filters by
        # nearest-position to the Kalman track. Kalman is the "anchor"
        # because its IDs are stable from the Hungarian assignment.
        out: List[ObjectState] = []

        # Identify currently-detected ids on the Kalman side so we can
        # mark "missed this tick" for the selector.
        # Kalman's confirmed list excludes brand-new tracks; we use it as
        # the authoritative "tracks producing output this tick" set.
        kf_ids_now = set(kf_by_id.keys())

        for kid, kstate in kf_by_id.items():
            sel = self._selectors.setdefault(kid, _PerTrackSelector(self.cfg))

            # "missed" = Kalman track existed last tick but didn't get a
            # match this tick; we approximate by checking the underlying
            # tracker's miss counter.
            kf_internal = self.kf.trackers.get(kid)
            missed = bool(kf_internal and kf_internal.misses > 0)

            # Innovation magnitude: nearest UKF track gives us its last
            # innovation; if no UKF match, treat as 0.
            ukf_match = self._nearest_state_to(kstate, ukf_states)
            ukf_internal = self._internal_track_for(self.ukf, ukf_match)
            innovation_px = (
                ukf_internal.last_innovation_px
                if ukf_internal is not None
                else 0.0
            )
            yaw_rate = (
                ukf_internal.filter.yaw_rate
                if ukf_internal is not None
                else 0.0
            )
            sel.record(missed=missed, yaw_rate=yaw_rate, innovation_px=innovation_px)

            choice = sel.choose()
            if choice == "kalman":
                chosen = kstate
            elif choice == "ukf":
                chosen = ukf_match or kstate
            else:  # pf
                chosen = self._nearest_state_to(kstate, pf_states) or kstate

            # Re-stamp with the Kalman track_id so downstream IDs are
            # stable across filter switches; tag bbox dict with which
            # filter produced this estimate so the visualizer can show it.
            tagged_bbox = dict(chosen.bbox)
            tagged_bbox["filter_used"] = choice
            chosen = ObjectState(
                track_id=kid,
                timestamp=chosen.timestamp,
                x=chosen.x,
                y=chosen.y,
                z=chosen.z,
                vx=chosen.vx,
                vy=chosen.vy,
                vz=chosen.vz,
                pos_uncertainty=chosen.pos_uncertainty,
                vel_uncertainty=chosen.vel_uncertainty,
                confidence=chosen.confidence,
                bbox=tagged_bbox,
                units=chosen.units,
                frame=chosen.frame,
            )
            out.append(chosen)

        # GC selectors for tracks that no longer exist on the Kalman side.
        for dead in set(self._selectors) - kf_ids_now:
            del self._selectors[dead]

        return out

    @staticmethod
    def _nearest_state_to(anchor: ObjectState, candidates: List[ObjectState]) -> Optional[ObjectState]:
        if not candidates:
            return None
        best, best_d = None, float("inf")
        for c in candidates:
            d = math.hypot(c.x - anchor.x, c.y - anchor.y)
            if d < best_d:
                best, best_d = c, d
        return best

    @staticmethod
    def _internal_track_for(multi_tracker, state: Optional[ObjectState]):
        if state is None:
            return None
        return multi_tracker.trackers.get(state.track_id)


def create_adaptive_tracker(
    max_distance: float = 100.0,
    max_tracks: int = 50,
) -> MultiObjectAdaptive:
    return MultiObjectAdaptive(max_distance=max_distance, max_tracks=max_tracks)
