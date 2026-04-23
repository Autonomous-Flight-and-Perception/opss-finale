"""
Multi-object tracker using one ParticleFilter3D per track. Same external
interface as MultiObjectKalmanFilter / MultiObjectCTRVUKF.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import ObjectState
from .pf import PFConfig, ParticleFilter3D


class PFTrack:
    def __init__(
        self,
        track_id: int,
        detection: Dict,
        timestamp: float,
        config: PFConfig,
    ):
        self.track_id = track_id
        self.last_update = timestamp
        self.age = 0
        self.hits = 1
        self.misses = 0

        self.filter = ParticleFilter3D(config)
        c = detection.get("center", {})
        depth = float(detection.get("depth", 0.0))
        self.filter.initialize((float(c.get("x", 0.0)), float(c.get("y", 0.0)), depth))

        self.confidence = float(detection.get("confidence", 0.0))
        self.bbox = dict(detection.get("bbox", {}))
        self.last_ess = float(config.particle_count)

    def predict(self, dt: float) -> None:
        self.filter.predict(dt)
        self.age += 1

    def update(self, detection: Dict, timestamp: float) -> None:
        c = detection.get("center", {})
        z = (float(c.get("x", 0.0)),
             float(c.get("y", 0.0)),
             float(detection.get("depth", 0.0)))
        _, ess = self.filter.update(z)
        self.last_ess = ess
        self.last_update = timestamp
        self.hits += 1
        self.misses = 0
        self.confidence = float(detection.get("confidence", self.confidence))
        self.bbox = dict(detection.get("bbox", self.bbox))

    def mark_missed(self) -> None:
        self.misses += 1

    def to_state(self) -> ObjectState:
        px, py, pz = self.filter.position
        vx, vy, vz = self.filter.velocity
        return ObjectState(
            track_id=self.track_id,
            timestamp=self.last_update,
            x=float(px),
            y=float(py),
            z=float(pz),
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            pos_uncertainty=float(self.filter.position_std_xy),
            vel_uncertainty=float(self.filter.velocity_std_xy),
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


class MultiObjectParticleFilter:
    def __init__(
        self,
        max_distance: float = 100.0,
        max_tracks: int = 50,
        max_misses: int = 8,         # PFs handle dropouts better; bigger budget
        config: Optional[PFConfig] = None,
    ):
        self.trackers: Dict[int, PFTrack] = {}
        self.next_id = 0
        self.max_distance = float(max_distance)
        self.max_tracks = int(max_tracks)
        self.max_misses = int(max_misses)
        self.config = config or PFConfig()
        self._last_timestamp: Optional[float] = None

    def update(self, detections: List[Dict], timestamp: float) -> List[ObjectState]:
        if not self.trackers:
            for det in detections:
                self._spawn(det, timestamp)
            self._last_timestamp = timestamp
            return self._confirmed_states()

        if self._last_timestamp is None:
            dt = 0.033
        else:
            dt = max(timestamp - self._last_timestamp, 1e-3)
        self._last_timestamp = timestamp

        for tr in self.trackers.values():
            tr.predict(dt)

        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(self.trackers.keys())

        if detections and self.trackers:
            tids = list(self.trackers.keys())
            cost = np.full((len(detections), len(tids)), 1e9, dtype=float)
            for i, det in enumerate(detections):
                c = det.get("center", {})
                dx, dy = float(c.get("x", 0)), float(c.get("y", 0))
                for j, tid in enumerate(tids):
                    px, py, _ = self.trackers[tid].filter.position
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
        self.trackers[self.next_id] = PFTrack(
            track_id=self.next_id,
            detection=detection,
            timestamp=timestamp,
            config=self.config,
        )
        self.next_id += 1

    def _confirmed_states(self) -> List[ObjectState]:
        return [t.to_state() for t in self.trackers.values() if t.is_confirmed]


def create_pf_tracker(
    max_distance: float = 100.0,
    max_tracks: int = 50,
    particle_count: int = 300,
) -> MultiObjectParticleFilter:
    return MultiObjectParticleFilter(
        max_distance=max_distance,
        max_tracks=max_tracks,
        config=PFConfig(particle_count=particle_count),
    )
