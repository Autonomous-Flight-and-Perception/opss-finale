from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import json
import numpy as np

from .integrators import step as integ_step, integrate as integ_integrate
from .forces import (
    ForceFn, ForceConfig, GravityParams, WindParams, DragParams,
    _ensure_vec, wind_function, accel_gravity, accel_quadratic_drag, combine_forces
)
from .forces_spin import MagnusParams, accel_magnus, combine_forces_with_spin

Vec = np.ndarray
StateTuple = Tuple[Vec, Vec]  # (x, v)
Callback = Callable[[Dict], None]

@dataclass
class EngineConfig:
    method: str = "rk4"
    detect_ground: bool = True
    plane_n: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    plane_h: float = 0.0
    z_tol: float = 1e-9
    bisection_iters: int = 3
    seed: Optional[int] = None

@dataclass
class Simulation:
    """
    Orchestrates stepping with modular forces and event callbacks.
    API:
      reset(cfg_dict, x0, v0, t0=0.0)
      step(dt) -> state_dict
      run(dt, steps) -> state_dict
      on(event_name, fn)  # events: "step", "ground_hit", "reset"
    State dict is JSON-serializable: {t, x, v, a, forces:{gravity,drag,spin}, hit_ground?}
    """
    config: EngineConfig = field(default_factory=EngineConfig)

    # runtime
    _t: float = 0.0
    _x: Vec = field(default_factory=lambda: np.zeros(3, dtype=float))
    _v: Vec = field(default_factory=lambda: np.zeros(3, dtype=float))
    _force_total: ForceFn = lambda t, x, v: np.zeros(3, dtype=float)
    _force_parts: Dict[str, ForceFn] = field(default_factory=dict)
    _callbacks: Dict[str, List[Callback]] = field(default_factory=lambda: {"step": [], "ground_hit": [], "reset": []})
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def on(self, event: str, fn: Callback) -> None:
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(fn)

    def _emit(self, event: str, payload: Dict) -> None:
        for fn in self._callbacks.get(event, []):
            fn(payload)

    # ---- configuration and forces setup ----
    def reset(self, cfg: Dict, x0: Vec, v0: Vec, t0: float = 0.0) -> Dict:
        # seed handling
        if self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)

        # Build force functions from config, supporting optional spin block.
        has_spin = ("spin" in cfg) and (cfg["spin"] is not None)
        grav = cfg.get("gravity", {"g": 9.80665, "scale": 1.0})
        wind = cfg.get("wind", None)
        drag = cfg.get("drag", None)
        gp = None if grav is None else GravityParams(g=float(grav.get("g", 9.80665)), scale=float(grav.get("scale", 1.0)))
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
        w_fn = wind_function(base_cfg.wind)
        a_g = accel_gravity(base_cfg.gravity)
        a_d = accel_quadratic_drag(base_cfg.drag, w_fn)
        if has_spin:
            sp = cfg["spin"]
            mp = MagnusParams(
                k=float(sp.get("k", 0.0)),
                omega0=_ensure_vec(sp.get("omega0", [0.0, 0.0, 0.0])),
                decay_lambda=float(sp.get("decay_lambda", 0.0)),
                use_wind_relative=bool(sp.get("use_wind_relative", True)),
            )
            a_s = accel_magnus(mp, w_fn)
            total = combine_forces_with_spin(base_cfg, mp)
        else:
            a_s = (lambda t, x, v: np.zeros(3, dtype=float))
            total = combine_forces(base_cfg)

        self._force_parts = {"gravity": a_g, "drag": a_d, "spin": a_s}
        self._force_total = total

        # Initialize state
        self._t = float(t0)
        self._x = _ensure_vec(x0).copy()
        self._v = _ensure_vec(v0).copy()

        st = self._serialize(hit=False)
        self._emit("reset", st)
        return st

    # ---- core stepping ----
    def _accel_total(self, t: float, x: Vec, v: Vec) -> Vec:
        return self._force_total(t, x, v)

    def step(self, dt: float) -> Dict:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        t0 = self._t
        if self.config.detect_ground:
            res = integ_integrate(
                self.config.method, self._x, self._v, t0, t0 + dt, dt,
                self._accel_total,
                detect_ground=True,
                plane_n=np.asarray(self.config.plane_n, dtype=float),
                plane_h=float(self.config.plane_h),
                z_tol=float(self.config.z_tol),
                bisection_iters=int(self.config.bisection_iters),
            )
            self._t, self._x, self._v = res.t, res.x, res.v
            st = self._serialize(hit=res.hit_ground)
            self._emit("step", st)
            if res.hit_ground:
                self._emit("ground_hit", st)
            return st
        else:
            x1, v1 = integ_step(self.config.method, (self._x, self._v), t0, dt, self._accel_total)
            self._t = t0 + dt
            self._x, self._v = x1, v1
            st = self._serialize(hit=False)
            self._emit("step", st)
            return st

    def run(self, dt: float, steps: int) -> Dict:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        last = {}
        for _ in range(steps):
            last = self.step(dt)
            if self.config.detect_ground and last.get("hit_ground", False):
                break
        return last

    # ---- JSON state ----
    def _forces_breakdown(self, t: float, x: Vec, v: Vec) -> Dict[str, List[float]]:
        out = {}
        for name, fn in self._force_parts.items():
            a = fn(t, x, v)
            out[name] = [float(a[0]), float(a[1]), float(a[2])]
        return out

    def _serialize(self, hit: bool) -> Dict:
        a = self._accel_total(self._t, self._x, self._v)
        st = {
            "t": float(self._t),
            "x": [float(self._x[0]), float(self._x[1]), float(self._x[2])],
            "v": [float(self._v[0]), float(self._v[1]), float(self._v[2])],
            "a": [float(a[0]), float(a[1]), float(a[2])],
            "forces": self._forces_breakdown(self._t, self._x, self._v),
            "hit_ground": bool(hit),
        }
        return st

    def to_json(self) -> str:
        return json.dumps(self._serialize(hit=False))

    def state(self) -> Dict:
        """Return JSON-serializable current state without stepping."""
        return self._serialize(hit=False)
