"""
Observation noise model for synthetic detection generation.

Adds realistic noise to projected pixel + depth observations:
  - Gaussian pixel noise (isotropic)
  - Gaussian depth noise (proportional to depth² for realism, or constant)
  - Depth dropout (models stereo failures on texture-poor surfaces)
  - Optional integer rounding of pixel coordinates (matches YOLO output)
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ObservationNoise:
    """Configuration for synthetic observation noise."""
    pixel_noise_std: float = 1.5       # pixels, isotropic Gaussian
    depth_noise_std: float = 0.05      # meters (constant component)
    depth_noise_pct: float = 0.01      # fraction of depth (range-dependent)
    depth_dropout_prob: float = 0.02   # probability of depth=0 (stereo failure)
    round_pixels: bool = True          # round to int (match YOLO behavior)

    def apply(self, u, v, depth, rng):
        """
        Apply noise to a clean (u, v, depth) observation.

        Args:
            u, v: pixel coordinates (float)
            depth: depth in meters (along optical axis, >0)
            rng: numpy random Generator

        Returns:
            (u_noisy, v_noisy, depth_noisy) or None if depth dropped out.
            depth_noisy=0.0 signals dropout.
        """
        # Depth dropout
        if rng.random() < self.depth_dropout_prob:
            depth_out = 0.0
        else:
            # Depth noise: constant + range-proportional
            depth_sigma = self.depth_noise_std + self.depth_noise_pct * depth
            depth_out = depth + rng.normal(0, depth_sigma)
            depth_out = max(depth_out, 0.01)  # clamp to positive

        # Pixel noise
        u_out = u + rng.normal(0, self.pixel_noise_std)
        v_out = v + rng.normal(0, self.pixel_noise_std)

        if self.round_pixels:
            u_out = round(u_out)
            v_out = round(v_out)

        return u_out, v_out, depth_out
