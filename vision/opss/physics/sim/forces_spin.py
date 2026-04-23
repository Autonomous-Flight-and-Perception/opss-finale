from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
import numpy as np

from .forces import (
    Vec, ForceFn, ForceConfig,
    wind_function, combine_forces, _ensure_vec,
    GravityParams, WindParams, DragParams  # for config helper
)

@dataclass(frozen=True, slots=True)
class MagnusParams:
    """
    Magnus acceleration: a_M = k * (omega(t) × v_rel)
    Note: k already includes 1/m so result is acceleration.
    """
    k: float = 0.0
    omega0: Vec = field(default_factory=lambda: np.zeros(3, dtype=float))
    decay_lambda: float = 0.0
    use_wind_relative: bool = True

def accel_magnus(params: Optional[MagnusParams], wind_fn: Callable[[float], Vec]) -> ForceFn:
    if params is None or params.k == 0.0:
        return lambda t, x, v: np.zeros(3, dtype=float)

    k = float(params.k)
    lam = float(params.decay_lambda)
    omega0 = _ensure_vec(params.omega0).copy()
    use_rel = bool(params.use_wind_relative)

    def omega_t(t: float) -> Vec:
        return omega0 * (np.exp(-lam * t) if lam != 0.0 else 1.0)

    def _a(t: float, x: Vec, v: Vec) -> Vec:
        v_rel = _ensure_vec(v) - (wind_fn(t) if use_rel else np.zeros(3, dtype=float))
        om = omega_t(float(t))
        return k * np.cross(om, v_rel)
    return _a

def combine_forces_with_spin(base: ForceConfig, spin: Optional[MagnusParams]) -> ForceFn:
    base_fn = combine_forces(base)
    w_fn = wind_function(base.wind)
    aM = accel_magnus(spin, w_fn)
    def a(t: float, x: Vec, v: Vec) -> Vec:
        return base_fn(t, x, v) + aM(t, x, v)
    return a

def from_config_dict_spin(cfg: Dict) -> ForceFn:
    """
    Extend Stage-3 config with optional:
      "spin": { "k": ..., "omega0": [..3..], "decay_lambda": ..., "use_wind_relative": true }
    """
    # Build base ForceConfig
    grav = cfg.get("gravity", None)
    wind = cfg.get("wind", None)
    drag = cfg.get("drag", None)

    gp = None if grav is None else GravityParams(
        g=float(grav.get("g", 9.80665)),
        scale=float(grav.get("scale", 1.0)),
    )
    wp = None
    if wind is not None:
        w0 = _ensure_vec(wind.get("w0", [0.0, 0.0, 0.0]))
        w1 = wind.get("w1", None)
        wp = WindParams(
            w0=w0,
            w1=None if w1 is None else _ensure_vec(w1),
            t_gust=None if wind.get("t_gust", None) is None else float(wind["t_gust"]),
        )
    dp = None
    if drag is not None:
        dp = DragParams(
            rho=float(drag.get("rho", 0.0)),
            Cd=float(drag.get("Cd", 0.0)),
            A=float(drag.get("A", 0.0)),
            m=float(drag.get("m", 1.0)),
        )
    base_cfg = ForceConfig(gravity=gp, wind=wp, drag=dp)

    sp = cfg.get("spin", None)
    if sp is None:
        return combine_forces_with_spin(base_cfg, None)

    mp = MagnusParams(
        k=float(sp.get("k", 0.0)),
        omega0=_ensure_vec(sp.get("omega0", [0.0, 0.0, 0.0])),
        decay_lambda=float(sp.get("decay_lambda", 0.0)),
        use_wind_relative=bool(sp.get("use_wind_relative", True)),
    )
    return combine_forces_with_spin(base_cfg, mp)
