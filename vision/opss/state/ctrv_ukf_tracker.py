"""
Multi-object tracker built on top of the CTRV UKF (`opss.state.ctrv_ukf`).

Same external interface as ``MultiObjectKalmanFilter`` so the pipeline can
swap one for the other:

    update(detections: List[Dict], timestamp: float) -> List[ObjectState]

Per-track lifecycle:
  - new track on every unmatched detection (with min-hits confirmation)
  - Hungarian (linear-sum) detection→track association on predicted position
  - track is removed after ``max_misses`` consecutive un-matched ticks

Output ObjectState uses ``frame="pixel"`` since the CTRV state is pixel-space;
``z`` is passed through from the detection's ``depth`` (meters from RealSense)
purely for display.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import ObjectState
from .ctrv_ukf import CTRVUKF


class CTRVUKFTrack:
    def __init__(
        self,
        track_id: int,
        detection: Dict,
        timestamp: float,
        sigma_a: float,
        sigma_omega: float,
        sigma_z: float,
    ):
        self.track_id = track_id
        self.last_update = timestamp
        self.age = 0
        self.hits = 1
        self.misses = 0

        self.filter = CTRVUKF(sigma_a=sigma_a, sigma_omega=sigma_omega, sigma_z=sigma_z)
        center = detection.get("center", {})
        self.filter.init_from_measurement(
            np.array([center.get("x", 0.0), center.get("y", 0.0)], dtype=float)
        )

        self.confidence = float(detection.get("confidence", 0.0))
        self.bbox = dict(detection.get("bbox", {}))
        self.depth_m = float(detection.get("depth", 0.0))
        self.last_innovation_px = 0.0

    def predict(self, dt: float) -> None:
        if dt <= 0:
            dt = 1e-3
        self.filter.predict(dt)
        self.age += 1

    def update(self, detection: Dict, timestamp: float) -> None:
        center = detection.get("center", {})
        z = np.array([center.get("x", 0.0), center.get("y", 0.0)], dtype=float)
        self.last_innovation_px = self.filter.update(z)
        self.last_update = timestamp
        self.hits += 1
        self.misses = 0
        self.confidence = float(detection.get("confidence", self.confidence))
        self.bbox = dict(detection.get("bbox", self.bbox))
        self.depth_m = float(detection.get("depth", self.depth_m))

    def mark_missed(self) -> None:
        self.misses += 1

    def to_state(self) -> ObjectState:
        px, py = self.filter.position
        vx, vy = self.filter.velocity_xy
        # Diagonal-only uncertainty proxy: sqrt of trace of position / velocity blocks.
        P = self.filter.P
        pos_unc = float(np.sqrt(max(P[0, 0] + P[1, 1], 0.0)))
        vel_unc = float(np.sqrt(max(P[2, 2], 0.0)))  # speed variance only
        return ObjectState(
            track_id=self.track_id,
            timestamp=self.last_update,
            x=float(px),
            y=float(py),
            z=float(self.depth_m),
            vx=float(vx),
            vy=float(vy),
            vz=0.0,
            pos_uncertainty=pos_unc,
            vel_uncertainty=vel_unc,
            confidence=self.confidence,
            bbox=self.bbox,
            units="pixels",
            frame="pixel_xy_metric_z",
        )

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    def is_dead(self, max_misses: int) -> bool:
        return self.misses > max_misses


class MultiObjectCTRVUKF:
    """
    CTRV-UKF tracker matching the ``MultiObjectKalmanFilter`` interface so the
    pipeline can use it as a drop-in tracker.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_tracks: int = 50,
        max_misses: int = 5,
        sigma_a: float = 250.0,
        sigma_omega: float = 1.5,
        sigma_z: float = 8.0,
    ):
        self.trackers: Dict[int, CTRVUKFTrack] = {}
        self.next_id = 0
        self.max_distance = float(max_distance)
        self.max_tracks = int(max_tracks)
        self.max_misses = int(max_misses)
        self.sigma_a = float(sigma_a)
        self.sigma_omega = float(sigma_omega)
        self.sigma_z = float(sigma_z)
        self._last_timestamp: Optional[float] = None

    def update(self, detections: List[Dict], timestamp: float) -> List[ObjectState]:
        # First call: nothing to associate — bootstrap tracks from detections.
        if not self.trackers:
            for det in detections:
                self._spawn(det, timestamp)
            self._last_timestamp = timestamp
            return self._confirmed_states()

        # dt from previous tick (per-tracker last_update would be more precise
        # but the pipeline ticks at uniform 30 Hz so a single dt is fine).
        if self._last_timestamp is None:
            dt = 0.033
        else:
            dt = max(timestamp - self._last_timestamp, 1e-3)
        self._last_timestamp = timestamp

        # Predict every tracker forward
        for tr in self.trackers.values():
            tr.predict(dt)

        # Associate detections to predicted positions (Hungarian)
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(self.trackers.keys())

        if detections and self.trackers:
            tids = list(self.trackers.keys())
            cost = np.full((len(detections), len(tids)), 1e9, dtype=float)
            for i, det in enumerate(detections):
                c = det.get("center", {})
                dx, dy = float(c.get("x", 0)), float(c.get("y", 0))
                for j, tid in enumerate(tids):
                    px, py = self.trackers[tid].filter.position
                    cost[i, j] = float(np.hypot(dx - px, dy - py))
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_d, matched_t = set(), set()
            for di, tj in zip(row_ind, col_ind):
                if cost[di, tj] > self.max_distance:
                    continue
                tid = tids[tj]
                self.trackers[tid].update(detections[di], timestamp)
                matched_d.add(di)
                matched_t.add(tid)
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]
            unmatched_trks = [tid for tid in tids if tid not in matched_t]

        for tid in unmatched_trks:
            self.trackers[tid].mark_missed()

        for di in unmatched_dets:
            self._spawn(detections[di], timestamp)

        dead = [tid for tid, tr in self.trackers.items() if tr.is_dead(self.max_misses)]
        for tid in dead:
            del self.trackers[tid]

        return self._confirmed_states()

    def _spawn(self, detection: Dict, timestamp: float) -> None:
        if len(self.trackers) >= self.max_tracks:
            return
        self.trackers[self.next_id] = CTRVUKFTrack(
            track_id=self.next_id,
            detection=detection,
            timestamp=timestamp,
            sigma_a=self.sigma_a,
            sigma_omega=self.sigma_omega,
            sigma_z=self.sigma_z,
        )
        self.next_id += 1

    def _confirmed_states(self) -> List[ObjectState]:
        return [t.to_state() for t in self.trackers.values() if t.is_confirmed]

    def clear(self) -> None:
        self.trackers.clear()


def create_ctrv_ukf_tracker(
    max_distance: float = 100.0,
    max_tracks: int = 50,
    sigma_a: float = 250.0,
    sigma_omega: float = 1.5,
    sigma_z: float = 8.0,
) -> MultiObjectCTRVUKF:
    return MultiObjectCTRVUKF(
        max_distance=max_distance,
        max_tracks=max_tracks,
        sigma_a=sigma_a,
        sigma_omega=sigma_omega,
        sigma_z=sigma_z,
    )
