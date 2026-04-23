from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict
import numpy as np

Vec = np.ndarray  # shape (3,)
ForceFn = Callable[[float, Vec, Vec], Vec]  # a(t, x, v) -> (3,)

# ---- Parameter containers (pure, hashable-ish) ----

@dataclass(frozen=True, slots=True)
class GravityParams:
    g: float = 9.80665          # m/s^2
    scale: float = 1.0          # optional global scale

@dataclass(frozen=True, slots=True)
class WindParams:
    w0: Vec                     # base wind vector (m/s), ENU
    w1: Optional[Vec] = None    # gust wind vector after t_gust (m/s)
    t_gust: Optional[float] = None

@dataclass(frozen=True, slots=True)
class DragParams:
    rho: float                  # air density (kg/m^3)
    Cd: float                   # drag coefficient
    A: float                    # reference area (m^2)
    m: float = 1.0              # mass (kg), default 1 per Stage 1 energy convention

@dataclass(frozen=True, slots=True)
class ForceConfig:
    gravity: Optional[GravityParams] = GravityParams()
    wind: Optional[WindParams] = None
    drag: Optional[DragParams] = None

# ---- Helpers ----

def _ensure_vec(a) -> Vec:
    a = np.asarray(a, dtype=float)
    if a.shape != (3,):
        raise ValueError("Vector must be shape (3,)")
    return a

def wind_function(params: Optional[WindParams]) -> Callable[[float], Vec]:
    """
    Returns w(t). If params is None, wind is zero.
    Supports a single step gust at t_gust changing from w0 -> w1.
    """
    if params is None:
        zero = np.zeros(3, dtype=float)
        return lambda t: zero

    w0 = _ensure_vec(params.w0).copy()
    if params.w1 is None or params.t_gust is None:
        return lambda t: w0

    w1 = _ensure_vec(params.w1).copy()
    tg = float(params.t_gust)
    return lambda t: (w0 if t < tg else w1)

# ---- Pure force components (accelerations) ----

def accel_gravity(params: Optional[GravityParams]) -> ForceFn:
    """
    Gravity acceleration in ENU: acts along -Z.
    """
    if params is None:
        return lambda t, x, v: np.zeros(3, dtype=float)
    g_eff = float(params.g) * float(params.scale)
    gvec = np.array([0.0, 0.0, -g_eff], dtype=float)
    return lambda t, x, v: gvec

def accel_quadratic_drag(params: Optional[DragParams], wind_fn: Callable[[float], Vec]) -> ForceFn:
    """
    Quadratic drag: a = - (0.5 * rho * Cd * A / m) * |v_rel| * v_rel
    where v_rel = v - w(t).
    """
    if params is None or params.Cd == 0.0 or params.A == 0.0 or params.rho == 0.0:
        return lambda t, x, v: np.zeros(3, dtype=float)

    k = 0.5 * float(params.rho) * float(params.Cd) * float(params.A) / float(params.m)

    def _a(t: float, x: Vec, v: Vec) -> Vec:
        v_rel = _ensure_vec(v) - wind_fn(t)
        speed = float(np.linalg.norm(v_rel))
        if speed == 0.0:
            return np.zeros(3, dtype=float)
        return -k * speed * v_rel
    return _a

def combine_forces(config: ForceConfig) -> ForceFn:
    """
    Build a single acceleration function that sums gravity and drag with wind.
    Pure: no mutation, all closures over immutable params.
    """
    w_fn = wind_function(config.wind)
    a_g = accel_gravity(config.gravity)
    a_d = accel_quadratic_drag(config.drag, w_fn)

    def a(t: float, x: Vec, v: Vec) -> Vec:
        return a_g(t, x, v) + a_d(t, x, v)
    return a

# ---- Minimal JSON-like loader (optional) ----

def from_config_dict(cfg: Dict) -> ForceFn:
    """
    Construct a combined force from a plain dict schema.

    Schema:
    {
      "gravity": {"g": 9.80665, "scale": 1.0} | null,
      "wind": {"w0": [..3..], "w1": [..3..], "t_gust": 1.2} | null,
      "drag": {"rho": 1.225, "Cd": 0.47, "A": 0.01, "m": 0.145} | null
    }
    """
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
    return combine_forces(ForceConfig(gravity=gp, wind=wp, drag=dp))
