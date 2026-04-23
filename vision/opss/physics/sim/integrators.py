from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple, Optional, Dict

__all__ = [
    "Vec", "State", "ForceFn", "StepResult",
    "euler", "semi_implicit_euler", "rk2", "rk4",
    "STEPPERS", "step", "integrate",
]

Vec = np.ndarray  # shape (3,)
State = Tuple[Vec, Vec]  # (x, v)
ForceFn = Callable[[float, Vec, Vec], Vec]  # a(t, x, v) -> (3,)

@dataclass(frozen=True, slots=True)
class StepResult:
    """
    Result of integrating from t0 to t_end or until plane hit.
    If hit_ground=True, (t, x, v) is the state at the *exact* hit time.
    """
    t: float
    x: Vec
    v: Vec
    hit_ground: bool
    t_hit: Optional[float] = None
    x_hit: Optional[Vec] = None
    v_hit: Optional[Vec] = None
    meta: Optional[Dict] = None

def _ensure_vec(a: np.ndarray) -> Vec:
    a = np.asarray(a, dtype=float).astype(float, copy=False)
    if a.shape != (3,):
        raise ValueError("Vector must be shape (3,)")
    return a

def euler(x: Vec, v: Vec, t: float, dt: float, a: ForceFn) -> State:
    """Explicit Euler."""
    a0 = a(t, x, v)
    x1 = x + dt * v
    v1 = v + dt * a0
    return x1, v1

def semi_implicit_euler(x: Vec, v: Vec, t: float, dt: float, a: ForceFn) -> State:
    """Symplectic / semi-implicit Euler."""
    a0 = a(t, x, v)
    v1 = v + dt * a0
    x1 = x + dt * v1
    return x1, v1

def rk2(x: Vec, v: Vec, t: float, dt: float, a: ForceFn) -> State:
    """Midpoint (RK2)."""
    a0 = a(t, x, v)
    x_mid = x + 0.5 * dt * v
    v_mid = v + 0.5 * dt * a0
    a_mid = a(t + 0.5 * dt, x_mid, v_mid)
    x1 = x + dt * v_mid
    v1 = v + dt * a_mid
    return x1, v1

def rk4(x: Vec, v: Vec, t: float, dt: float, a: ForceFn) -> State:
    """Classic RK4 on coupled (x', v') = (v, a)."""
    a1 = a(t, x, v)
    kx1, kv1 = v, a1

    x2 = x + 0.5 * dt * kx1
    v2 = v + 0.5 * dt * kv1
    a2 = a(t + 0.5 * dt, x2, v2)
    kx2, kv2 = v2, a2

    x3 = x + 0.5 * dt * kx2
    v3 = v + 0.5 * dt * kv2
    a3 = a(t + 0.5 * dt, x3, v3)
    kx3, kv3 = v3, a3

    x4 = x + dt * kx3
    v4 = v + dt * kv3
    a4 = a(t + dt, x4, v4)
    kx4, kv4 = v4, a4

    x1 = x + (dt / 6.0) * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
    v1 = v + (dt / 6.0) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
    return x1, v1

STEPPERS = {
    "euler": euler,
    "semi": semi_implicit_euler,
    "rk2": rk2,
    "rk4": rk4,
}

def step(method: str, state: State, t: float, dt: float, force: ForceFn) -> State:
    """
    One integrator step of length dt using the selected method.
    """
    x, v = state
    x = _ensure_vec(x); v = _ensure_vec(v)
    fn = STEPPERS.get(method.lower())
    if fn is None:
        raise ValueError(f"unknown method '{method}'")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    return fn(x, v, t, dt, force)

def _advance(method: str, x: Vec, v: Vec, t: float, h: float, force: ForceFn) -> State:
    """Advance by a sub-interval h with the same method."""
    return step(method, (x, v), t, h, force)

def integrate(
    method: str,
    x0: Vec,
    v0: Vec,
    t0: float,
    t_end: float,
    dt: float,
    force: ForceFn,
    detect_ground: bool = True,
    # Generic plane n·x = h_plane (default: z = 0)
    plane_n: Optional[Vec] = None,
    plane_h: float = 0.0,
    z_tol: float = 1e-9,
    bisection_iters: int = 3,
) -> StepResult:
    """
    Fixed-step integration with partial final step so returned time equals t_end,
    unless a plane hit occurs first. If a hit occurs, returns state at hit.
    Detects crossing in either direction across the plane.

    NOTE: n·x = h uses n and h **as provided** (no normalization). This lets you
    express planes like x+z=3 with n=[1,0,1], h=3 directly.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if t_end < t0:
        raise ValueError("t_end must be >= t0")

    n = _ensure_vec(np.array([0.0, 0.0, 1.0]) if plane_n is None else plane_n)
    n_norm2 = float(n.dot(n))
    if n_norm2 == 0.0:
        raise ValueError("plane normal must be non-zero")
    tol = z_tol * float(np.sqrt(n_norm2))

    def phi(x: Vec) -> float:
        # signed offset from plane (positive "above")
        return float(n.dot(x) - plane_h)

    x = _ensure_vec(x0).copy()
    v = _ensure_vec(v0).copy()
    t = float(t0)

    fn = STEPPERS.get(method.lower())
    if fn is None:
        raise ValueError(f"unknown method '{method}'")

    steps = 0
    while t < t_end:
        h = min(dt, t_end - t)
        f_prev = phi(x)
        x_next, v_next = fn(x, v, t, h, force)
        f_next = phi(x_next)

        if detect_ground:
            # Crossing if sign changes across [-tol, tol] band in either direction
            crosses_down = (f_prev >  tol) and (f_next <=  tol)
            crosses_up   = (f_prev < -tol) and (f_next >= -tol)
            if crosses_down or crosses_up:
                # Bisection in [0, h] for root of phi
                ta, xa, va = 0.0, x, v
                tb, xb, vb = h, x_next, v_next
                fa = phi(xa)
                fb = phi(xb)

                for _ in range(max(1, bisection_iters)):
                    tm = 0.5 * (ta + tb)
                    xm, vm = _advance(method, xa, va, t + ta, tm - ta, force)
                    fm = phi(xm)
                    # Choose sub-interval by sign; if ambiguous, prefer the band side that reduces |fm|
                    if (fa > 0 and fm >= 0) or (fa < 0 and fm <= 0):
                        ta, xa, va, fa = tm, xm, vm, fm
                    else:
                        tb, xb, vb, fb = tm, xm, vm, fm

                t_hit = t + tb
                x_hit, v_hit = xb.copy(), vb.copy()
                # Project residual to plane when n is not unit: x' = x - ((n·x - h)/||n||^2) n
                off = n.dot(x_hit) - plane_h
                x_hit = x_hit - (off / n_norm2) * n

                return StepResult(
                    t=t_hit, x=x_hit, v=v_hit, hit_ground=True,
                    t_hit=t_hit, x_hit=x_hit, v_hit=v_hit,
                    meta={"steps": steps + 1, "method": method, "dt": dt, "refined": True},
                )

        # No event this sub-step
        x, v = x_next, v_next
        t += h
        steps += 1

    # Finished exactly at t_end
    return StepResult(
        t=t, x=x, v=v, hit_ground=False,
        meta={"steps": steps, "method": method, "dt": dt, "refined": False},
    )
