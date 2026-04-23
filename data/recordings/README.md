# OPSS filter-comparison recordings

Live captures of the OPSS state-estimation pipeline running on the Jetson
with **all four filters in parallel** against the same RealSense detection
stream:

- `kalman` — linear constant-velocity KF (pixel-space)
- `ukf`    — CTRV nonlinear UKF (ported from the MonteCarlo repo)
- `pf`     — 3D particle filter, 300 particles (ported from the PF repo)
- `adaptive` — auto-switches KF / UKF / PF per tick by motion classifier:
    * recent missed detections (≥3 of last 10 ticks) → **PF**
    * sustained yaw rate (>0.6 rad/s) or innovation (>25 px) → **UKF**
    * otherwise → **Kalman**

The cobot wire feed is driven by `adaptive` (primary). Each recording
captures every filter's primary track per tick PLUS the cobot's commanded
joint state PLUS pipeline-wide stats, all on a shared `tick` index for
joining.

## Runs

| Run id | Intent | Duration | Ticks | YOLO det rate | Adaptive split (K/U/P) | Notes |
|---|---|---|---|---|---|---|
| `20260423_190846` | "Linear" flight (Flight 1) | 84.6 s | 389 | 41% | 13 / 74 / 13 % | Hand-held flight; adaptive flagged most of it as nonlinear (UKF dominant) — likely natural hand wobble. Cobot J1 swept 65°, J2 swept 49°. **Mixed-regime in practice.** |
| `20260423_191637` | Turning flight (Flight 2) | 65.1 s | 283 | 20% | 23 / 33 / 44 % | Drone left frame more (lower det rate). Adaptive shifted hard toward PF because the miss-to-PF rule kicked in. Useful for showing adaptive responding to **degraded detection**, not just motion regime. |

## Files per run

```
metadata_<run>.json        run config snapshot (one-shot at start)
                             - pipeline status (FPS, latencies, tracker config)
                             - active trackers + primary
                             - YOLO model file size + mtime
                             - Jetson uname

kf_log_<run>.csv           Kalman primary track per tick
ukf_log_<run>.csv          CTRV UKF primary track per tick
pf_log_<run>.csv           Particle filter primary track per tick
adaptive_log_<run>.csv     Adaptive primary track + which underlying filter
                            it picked (column: filter_used_by_adaptive)
cobot_log_<run>.csv        MyCobot joint state, parsed live from
                            `docker logs opss-cobot` ([MOVE]/[COAST] lines)
pipeline_stats_<run>.csv   pipeline FPS, latencies, valid/invalid counts,
                            per-filter active track counts
```

Sampling rate: **10 Hz** for all CSVs (recorder polls the vision API at
this rate; cobot rows hold the latest parsed log line at sample time).

## CSV schemas

### Filter logs (`kf_*`, `ukf_*`, `pf_*`, `adaptive_*`)

| Column | Meaning |
|---|---|
| `tick` | recorder tick index, 0-based, common across all CSVs in a run |
| `wall_time_unix` | wall-clock at sample (Unix seconds, float) |
| `wall_time_iso` | wall-clock UTC ISO 8601 |
| `dt_since_start_s` | seconds since recorder start |
| `raw_det_count` | how many raw YOLO detections this tick |
| `raw_x_px`, `raw_y_px` | center of the highest-confidence raw detection (px, capture-resolution) |
| `raw_conf` | highest raw detection confidence |
| `raw_depth_m` | depth at that pixel from RealSense (m) |
| `track_count` | how many tracks this filter is currently maintaining |
| `track_id` | id of the highest-confidence track this tick (filter-local) |
| `x`, `y`, `z` | filtered position (units in `frame`) |
| `vx`, `vy`, `vz` | filtered velocity |
| `pos_uncertainty`, `vel_uncertainty` | filter-reported scalar uncertainty |
| `confidence` | passed through from detection that updated this track |
| `frame` | coord frame: `pixel_xy_metric_z` (xy in px, z in m) for all current trackers |
| `physics_plausible`, `physics_valid` | physics validator flags |
| `filter_used_by_adaptive` | only populated in `adaptive_*` — the underlying filter (`kalman`/`ukf`/`pf`) the adaptive selector picked this tick |

### Cobot log (`cobot_*`)

| Column | Meaning |
|---|---|
| `tick` | shared with filter logs |
| `wall_time_unix`, `wall_time_iso`, `dt_since_start_s` | same as above |
| `log_line_age_s` | how stale the latest parsed cobot log line was at sample time (None if no line yet) |
| `mode` | `MOVE` (commanded velocity nonzero) or `COAST` (drone-out + decay) |
| `j1_deg`, `j2_deg` | commanded joint angles |
| `v_j1_deg_per_tick`, `v_j2_deg_per_tick` | commanded angular velocities (deg per cobot tick = ~1/30 s) |
| `cx_norm`, `cy_norm` | normalized target center the cobot is acting on (`x/640`, `y/480`); `nan` when no detection |
| `raw_log_line` | original `[MOVE]`/`[COAST]` log line for trace |

### Pipeline stats (`pipeline_stats_*`)

Per-tick snapshot of `/pipeline/status`: `pipeline_fps`, total/per-stage
latencies (`latency_total_ms`, `latency_detect_ms`, `latency_track_ms`,
`latency_validate_fuse_ms`), cumulative valid/invalid state counts, and
per-filter active track counts (`tracks_kalman`, `tracks_ukf`, `tracks_pf`,
`tracks_adaptive`).

## Joining for analysis

All CSVs in one run share the `tick` column. To compare filters tick-by-tick:

```python
import pandas as pd
RUN = "20260423_190846"
kf  = pd.read_csv(f"kf_log_{RUN}.csv")
ukf = pd.read_csv(f"ukf_log_{RUN}.csv")
pf  = pd.read_csv(f"pf_log_{RUN}.csv")
ada = pd.read_csv(f"adaptive_log_{RUN}.csv")
cobot = pd.read_csv(f"cobot_log_{RUN}.csv")

# Same-tick filter divergence (e.g., KF vs UKF)
merged = kf.merge(ukf, on="tick", suffixes=("_kf", "_ukf"))
merged["dx"] = merged["x_kf"] - merged["x_ukf"]
```

## Reproducing

The recorder is `_recorder.py` in this directory. It polls the live
Jetson at 10 Hz over Tailscale and tails the cobot container's docker
logs over SSH. Start it with `python3 _recorder.py`, stop with Ctrl+C.

`_discarded/` holds aborted/empty runs, kept for audit but not part of
the reportable dataset.
