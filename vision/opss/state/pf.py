"""
Single-object 3D Particle Filter.

Ported from PF/particle_filter.py
(https://github.com/Autonomous-Flight-and-Perception/PF). The original
exposes a ``ParticleFilter3D`` with a ``step()`` API and a CameraBoxTracker
helper that converts bbox + intrinsics into a 3D measurement. This module
keeps the core stochastic-update / weighting / systematic-resampling math but
runs in **pixel space + RealSense depth** so it composes cleanly with the
existing detection front-end (no extra intrinsics plumbing).

State (6-D):
    [px, py, pz, vx, vy, vz]
        px, py  : pixel-space center
        pz      : depth (meters; from RealSense)
        vx, vy  : pixel velocity (px / s)
        vz      : depth-rate (m / s)

Measurement (3-D):
    [px, py, pz]
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PFConfig:
    particle_count: int = 300
    process_sigma_pos: float = 8.0     # px / step jitter on position
    process_sigma_vel: float = 80.0    # px/s jitter on velocity
    process_sigma_depth: float = 0.05  # m on depth
    measurement_sigma_xy: float = 12.0  # px
    measurement_sigma_z: float = 0.10   # m
    resample_threshold: float = 0.5
    seed: int = 42


def _systematic_resample(particles: List[List[float]], weights: List[float], rng: random.Random) -> List[List[float]]:
    n = len(particles)
    start = rng.random() / n
    positions = [start + i / n for i in range(n)]
    cum = []
    s = 0.0
    for w in weights:
        s += w
        cum.append(s)
    indexes = []
    i = 0
    for p in positions:
        while p > cum[i]:
            i += 1
            if i >= n:
                i = n - 1
                break
        indexes.append(i)
    return [particles[k][:] for k in indexes]


class ParticleFilter3D:
    """
    Pixel-x-y + meter-z particle filter, same algorithm as the upstream PF
    repo (gaussian likelihood, systematic resample on low ESS).

    All sigmas are mixed-units: position xy in pixels, depth in meters,
    velocity xy in px/s, vz in m/s. The mixed units are wrapped inside this
    class — callers see consistent measurement and estimate vectors.
    """

    def __init__(self, config: PFConfig | None = None):
        self.cfg = config or PFConfig()
        self.rng = random.Random(self.cfg.seed)
        self.particles: List[List[float]] = []
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, z: Tuple[float, float, float]) -> None:
        sx, sy, sz = z
        sigp = self.cfg.process_sigma_pos
        sigd = self.cfg.process_sigma_depth
        sigv = self.cfg.process_sigma_vel
        self.particles = [
            [
                sx + self.rng.gauss(0, sigp),
                sy + self.rng.gauss(0, sigp),
                sz + self.rng.gauss(0, sigd),
                self.rng.gauss(0, sigv),
                self.rng.gauss(0, sigv),
                self.rng.gauss(0, sigd * 5.0),
            ]
            for _ in range(self.cfg.particle_count)
        ]
        self._initialized = True

    def predict(self, dt: float) -> None:
        if dt <= 0:
            dt = 1e-3
        sigp = self.cfg.process_sigma_pos
        sigd = self.cfg.process_sigma_depth
        sigv = self.cfg.process_sigma_vel
        # Random walk on velocity, integrate to position
        for p in self.particles:
            p[3] += self.rng.gauss(0, sigv * dt)
            p[4] += self.rng.gauss(0, sigv * dt)
            p[5] += self.rng.gauss(0, sigd * dt)
            p[0] += p[3] * dt + self.rng.gauss(0, sigp * 0.1)
            p[1] += p[4] * dt + self.rng.gauss(0, sigp * 0.1)
            p[2] += p[5] * dt + self.rng.gauss(0, sigd * 0.1)

    def update(self, z: Tuple[float, float, float]) -> Tuple[Tuple[float, ...], float]:
        """
        Apply gaussian-likelihood weighting around ``z`` and (if needed)
        systematic-resample. Returns (mean_state_6, effective_count).
        """
        zx, zy, zz = z
        sigxy = self.cfg.measurement_sigma_xy
        sigz = self.cfg.measurement_sigma_z
        # Independent gaussian on (xy in px) and (z in m). Combine in log space.
        weights = []
        for p in self.particles:
            sq_xy = (p[0] - zx) ** 2 + (p[1] - zy) ** 2
            sq_z = (p[2] - zz) ** 2
            ll = -0.5 * (sq_xy / (sigxy * sigxy) + sq_z / (sigz * sigz))
            weights.append(math.exp(ll))
        total = sum(weights)
        if total <= 0:
            n = len(self.particles)
            normalized = [1.0 / n] * n
        else:
            normalized = [w / total for w in weights]

        # Weighted mean
        est = [0.0] * 6
        for p, w in zip(self.particles, normalized):
            for ax in range(6):
                est[ax] += p[ax] * w

        ess = 1.0 / sum(w * w for w in normalized) if normalized else 0.0
        if ess < self.cfg.particle_count * self.cfg.resample_threshold:
            self.particles = _systematic_resample(self.particles, normalized, self.rng)
        return tuple(est), ess

    @property
    def position(self) -> Tuple[float, float, float]:
        if not self._initialized or not self.particles:
            return (0.0, 0.0, 0.0)
        n = len(self.particles)
        sx = sum(p[0] for p in self.particles) / n
        sy = sum(p[1] for p in self.particles) / n
        sz = sum(p[2] for p in self.particles) / n
        return sx, sy, sz

    @property
    def velocity(self) -> Tuple[float, float, float]:
        if not self._initialized or not self.particles:
            return (0.0, 0.0, 0.0)
        n = len(self.particles)
        vx = sum(p[3] for p in self.particles) / n
        vy = sum(p[4] for p in self.particles) / n
        vz = sum(p[5] for p in self.particles) / n
        return vx, vy, vz

    @property
    def position_std_xy(self) -> float:
        if not self.particles:
            return 0.0
        sx, sy, _ = self.position
        var = sum((p[0] - sx) ** 2 + (p[1] - sy) ** 2 for p in self.particles) / len(self.particles)
        return math.sqrt(var * 0.5)

    @property
    def velocity_std_xy(self) -> float:
        if not self.particles:
            return 0.0
        vx, vy, _ = self.velocity
        var = sum((p[3] - vx) ** 2 + (p[4] - vy) ** 2 for p in self.particles) / len(self.particles)
        return math.sqrt(var * 0.5)
