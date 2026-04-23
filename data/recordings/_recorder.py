"""
OPSS filter-comparison recorder.

Outputs (all go in this directory, suffixed with the run start timestamp):
  - metadata_<ts>.json       run config snapshot (one-shot, at start)
  - kf_log_<ts>.csv          Kalman filter primary track per tick
  - ukf_log_<ts>.csv         CTRV UKF primary track per tick
  - pf_log_<ts>.csv          Particle filter primary track per tick
  - adaptive_log_<ts>.csv    Adaptive selector primary track + which
                             underlying filter it chose (KF/UKF/PF)
  - cobot_log_<ts>.csv       MyCobot joint state from `docker logs opss-cobot`
                             (parsed [MOVE]/[COAST] lines, latest at sample)
  - pipeline_stats_<ts>.csv  pipeline FPS, latencies, valid/invalid counts

Sampling:
  Filter / pipeline-stats CSVs poll the Jetson at 10 Hz over Tailscale.
  Cobot CSV uses the LATEST parsed log line from the docker tail at each
  10 Hz tick; the cobot itself emits at 30 Hz so we get the freshest line.

Stop with Ctrl+C; rows are flushed on each tick so partials are safe.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request as u
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HOST = "http://100.106.218.71:8000"
JETSON_SSH = "afp@100.106.218.71"
POLL_HZ = 10.0
HERE = Path(__file__).resolve().parent

# ----- columns -----

FILTER_COLUMNS = [
    "tick",
    "wall_time_unix",
    "wall_time_iso",
    "dt_since_start_s",
    "raw_det_count",
    "raw_x_px",
    "raw_y_px",
    "raw_conf",
    "raw_depth_m",
    "track_count",
    "track_id",
    "x", "y", "z",
    "vx", "vy", "vz",
    "pos_uncertainty",
    "vel_uncertainty",
    "confidence",
    "frame",
    "physics_plausible",
    "physics_valid",
    "filter_used_by_adaptive",  # only meaningful in adaptive_log; "" elsewhere
]

COBOT_COLUMNS = [
    "tick",
    "wall_time_unix",
    "wall_time_iso",
    "dt_since_start_s",
    "log_line_age_s",     # how stale this cobot row is at sample time
    "mode",               # MOVE / COAST / (empty)
    "j1_deg",
    "j2_deg",
    "v_j1_deg_per_tick",
    "v_j2_deg_per_tick",
    "cx_norm",
    "cy_norm",
    "raw_log_line",
]

PIPELINE_COLUMNS = [
    "tick",
    "wall_time_unix",
    "wall_time_iso",
    "dt_since_start_s",
    "running",
    "pipeline_fps",
    "frames_processed",
    "detections_total",
    "tracks_active",
    "valid_states_cum",
    "invalid_states_cum",
    "latency_total_ms",
    "latency_detect_ms",
    "latency_track_ms",
    "latency_validate_fuse_ms",
    "tracks_kalman",
    "tracks_ukf",
    "tracks_pf",
    "tracks_adaptive",
]

# ----- helpers -----

def _get(path: str, timeout: float = 2.0):
    return json.loads(u.urlopen(HOST + path, timeout=timeout).read())


def _row_filter(tick: int, t0: float, det_payload: dict, fused_list: list) -> dict:
    now = time.time()
    iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
    det = det_payload.get("detections") or []
    raw_x = raw_y = raw_conf = raw_depth = None
    if det:
        c = det[0].get("center", {})
        raw_x = c.get("x")
        raw_y = c.get("y")
        raw_conf = det[0].get("confidence")
        raw_depth = det[0].get("depth")
    base = {
        "tick": tick,
        "wall_time_unix": now,
        "wall_time_iso": iso,
        "dt_since_start_s": now - t0,
        "raw_det_count": len(det),
        "raw_x_px": raw_x,
        "raw_y_px": raw_y,
        "raw_conf": raw_conf,
        "raw_depth_m": raw_depth,
        "track_count": len(fused_list or []),
    }
    if not fused_list:
        base.update({
            "track_id": None,
            "x": None, "y": None, "z": None,
            "vx": None, "vy": None, "vz": None,
            "pos_uncertainty": None,
            "vel_uncertainty": None,
            "confidence": None,
            "frame": None,
            "physics_plausible": None,
            "physics_valid": None,
            "filter_used_by_adaptive": "",
        })
        return base
    prim = max(fused_list, key=lambda s: s.get("confidence", 0.0))
    p = prim.get("position") or {}
    v = prim.get("velocity") or {}
    validation = prim.get("validation") or {}
    chosen = (prim.get("bbox") or {}).get("filter_used", "") or ""
    unc = prim.get("uncertainty") or {}
    base.update({
        "track_id": prim.get("track_id"),
        "x": p.get("x"), "y": p.get("y"), "z": p.get("z"),
        "vx": v.get("vx"), "vy": v.get("vy"), "vz": v.get("vz"),
        "pos_uncertainty": unc.get("position"),
        "vel_uncertainty": unc.get("velocity"),
        "confidence": prim.get("confidence"),
        "frame": prim.get("frame"),
        "physics_plausible": validation.get("physics_plausible"),
        "physics_valid": validation.get("physics_valid"),
        "filter_used_by_adaptive": chosen,
    })
    return base


def _row_pipeline(tick: int, t0: float, status: dict) -> dict:
    now = time.time()
    iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
    s = status.get("stats") or {}
    per_filter = s.get("tracks_per_filter") or {}
    return {
        "tick": tick,
        "wall_time_unix": now,
        "wall_time_iso": iso,
        "dt_since_start_s": now - t0,
        "running": status.get("running"),
        "pipeline_fps": s.get("pipeline_fps"),
        "frames_processed": s.get("frames_processed"),
        "detections_total": s.get("detections_total"),
        "tracks_active": s.get("tracks_active"),
        "valid_states_cum": s.get("valid_states"),
        "invalid_states_cum": s.get("invalid_states"),
        "latency_total_ms": s.get("latency_total_ms"),
        "latency_detect_ms": s.get("latency_detect_ms"),
        "latency_track_ms": s.get("latency_track_ms"),
        "latency_validate_fuse_ms": s.get("latency_validate_fuse_ms"),
        "tracks_kalman":   per_filter.get("kalman"),
        "tracks_ukf":      per_filter.get("ukf"),
        "tracks_pf":       per_filter.get("pf"),
        "tracks_adaptive": per_filter.get("adaptive"),
    }


# ----- cobot tail (parses `docker logs -f opss-cobot` over SSH) -----

_COBOT_RE = re.compile(
    r"\[(MOVE|COAST)\]\s+j1=([+-]?\d+\.\d+)\s+j2=([+-]?\d+\.\d+)"
    r"\s+v=\(([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)"
    r"\s+\(cx=([+-]?\d+\.\d+|nan),\s*cy=([+-]?\d+\.\d+|nan)\)"
)


@dataclass
class _CobotState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen_at: float = 0.0
    raw: str = ""
    mode: str = ""
    j1: float = float("nan")
    j2: float = float("nan")
    vj1: float = float("nan")
    vj2: float = float("nan")
    cx: float = float("nan")
    cy: float = float("nan")


def _cobot_tail_thread(state: _CobotState, stop_event: threading.Event) -> None:
    cmd = ["ssh", "-o", "ServerAliveInterval=10", JETSON_SSH,
           "docker logs -f --tail=1 opss-cobot 2>&1"]
    while not stop_event.is_set():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True,
            )
            for line in proc.stdout:
                if stop_event.is_set():
                    break
                m = _COBOT_RE.search(line)
                if not m:
                    continue
                mode, j1, j2, vj1, vj2, cx, cy = m.groups()
                with state.lock:
                    state.last_seen_at = time.time()
                    state.raw = line.rstrip("\n")
                    state.mode = mode
                    state.j1 = float(j1)
                    state.j2 = float(j2)
                    state.vj1 = float(vj1)
                    state.vj2 = float(vj2)
                    state.cx = float("nan") if cx == "nan" else float(cx)
                    state.cy = float("nan") if cy == "nan" else float(cy)
            try:
                proc.terminate()
            except Exception:
                pass
        except Exception as e:
            print(f"[REC] cobot tail subprocess error: {e}; respawning in 2s", flush=True)
            time.sleep(2.0)


def _row_cobot(tick: int, t0: float, state: _CobotState) -> dict:
    now = time.time()
    iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
    with state.lock:
        seen = state.last_seen_at
        return {
            "tick": tick,
            "wall_time_unix": now,
            "wall_time_iso": iso,
            "dt_since_start_s": now - t0,
            "log_line_age_s": (now - seen) if seen > 0 else None,
            "mode": state.mode,
            "j1_deg": state.j1 if state.j1 == state.j1 else None,
            "j2_deg": state.j2 if state.j2 == state.j2 else None,
            "v_j1_deg_per_tick": state.vj1 if state.vj1 == state.vj1 else None,
            "v_j2_deg_per_tick": state.vj2 if state.vj2 == state.vj2 else None,
            "cx_norm": state.cx if state.cx == state.cx else None,
            "cy_norm": state.cy if state.cy == state.cy else None,
            "raw_log_line": state.raw,
        }


# ----- metadata snapshot -----

def _write_metadata(path: Path, ts: str) -> None:
    snap = {
        "recording_started_unix": time.time(),
        "recording_started_iso": datetime.utcnow().isoformat() + "Z",
        "recording_id": ts,
        "host": HOST,
        "poll_hz": POLL_HZ,
    }
    try:
        snap["pipeline_status"] = _get("/pipeline/status")
    except Exception as e:
        snap["pipeline_status_error"] = str(e)
    try:
        snap["trackers"] = _get("/trackers")
    except Exception as e:
        snap["trackers_error"] = str(e)
    try:
        # Pull active model file size + first 8 bytes (header) as a weak hash
        # surrogate by reading from the Jetson via SSH.
        info = subprocess.check_output(
            ["ssh", JETSON_SSH,
             "stat -c '%s %y' ~/Projects/opss-finale/vision/user_best.pt 2>/dev/null"],
            text=True, timeout=8,
        ).strip()
        snap["model_user_best_pt"] = info
    except Exception as e:
        snap["model_info_error"] = str(e)
    try:
        snap["jetson_uname"] = subprocess.check_output(
            ["ssh", JETSON_SSH, "uname -a"], text=True, timeout=5,
        ).strip()
    except Exception:
        pass
    path.write_text(json.dumps(snap, indent=2, default=str))


# ----- main -----

def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "metadata":      HERE / f"metadata_{ts}.json",
        "kalman":        HERE / f"kf_log_{ts}.csv",
        "ukf":           HERE / f"ukf_log_{ts}.csv",
        "pf":            HERE / f"pf_log_{ts}.csv",
        "adaptive":      HERE / f"adaptive_log_{ts}.csv",
        "cobot":         HERE / f"cobot_log_{ts}.csv",
        "pipeline":      HERE / f"pipeline_stats_{ts}.csv",
    }
    print(f"[REC] Run id: {ts}")
    print(f"[REC] Output dir: {HERE}")
    for k, p in paths.items():
        print(f"  {k:9s} -> {p.name}")
    print(f"[REC] Polling {HOST} at {POLL_HZ} Hz. Ctrl+C to stop.")

    _write_metadata(paths["metadata"], ts)
    print(f"[REC] Wrote metadata snapshot.")

    # Open all CSVs + write headers
    files = {}
    writers = {}
    for k in ("kalman", "ukf", "pf", "adaptive"):
        files[k] = open(paths[k], "w", newline="")
        writers[k] = csv.DictWriter(files[k], fieldnames=FILTER_COLUMNS)
        writers[k].writeheader()
    files["cobot"] = open(paths["cobot"], "w", newline="")
    writers["cobot"] = csv.DictWriter(files["cobot"], fieldnames=COBOT_COLUMNS)
    writers["cobot"].writeheader()
    files["pipeline"] = open(paths["pipeline"], "w", newline="")
    writers["pipeline"] = csv.DictWriter(files["pipeline"], fieldnames=PIPELINE_COLUMNS)
    writers["pipeline"].writeheader()
    for f in files.values():
        f.flush()

    # Cobot tail thread
    cobot_state = _CobotState()
    stop_event = threading.Event()
    cobot_thread = threading.Thread(
        target=_cobot_tail_thread, args=(cobot_state, stop_event),
        daemon=True, name="cobot-tail",
    )
    cobot_thread.start()

    period = 1.0 / POLL_HZ
    t0 = time.time()
    tick = 0
    last_print = t0
    try:
        while True:
            try:
                det = _get("/detections/latest")
                fused = {
                    name: _get(f"/fused/{name}/latest").get("fused_states", [])
                    for name in ("kalman", "ukf", "pf", "adaptive")
                }
                status = _get("/pipeline/status")
            except Exception as e:
                print(f"[REC] tick {tick}: poll error {e}", flush=True)
                time.sleep(period)
                tick += 1
                continue

            for name in ("kalman", "ukf", "pf", "adaptive"):
                writers[name].writerow(_row_filter(tick, t0, det, fused[name]))
                files[name].flush()
            writers["cobot"].writerow(_row_cobot(tick, t0, cobot_state))
            files["cobot"].flush()
            writers["pipeline"].writerow(_row_pipeline(tick, t0, status))
            files["pipeline"].flush()

            now = time.time()
            if now - last_print >= 2.0:
                counts = {k: len(v) for k, v in fused.items()}
                cobot_age = (now - cobot_state.last_seen_at) if cobot_state.last_seen_at > 0 else None
                age_str = "n/a" if cobot_age is None else f"{cobot_age:.2f}s"
                print(f"[REC] tick={tick:5d} t={now-t0:6.1f}s "
                      f"raw={len(det.get('detections',[]))} tracks={counts} "
                      f"cobot[{cobot_state.mode or '-'} age={age_str}]",
                      flush=True)
                last_print = now

            tick += 1
            sleep_for = period - ((time.time() - t0) % period)
            if 0 < sleep_for < period:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[REC] Stopping…")
    finally:
        stop_event.set()
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
        elapsed = time.time() - t0
        print(f"[REC] Done. {tick} ticks over {elapsed:.1f}s ({tick/max(elapsed,1e-6):.1f} Hz effective).")
        for k, p in paths.items():
            try:
                sz = p.stat().st_size
                print(f"  {k:9s} {p.name}  ({sz} bytes)")
            except Exception:
                print(f"  {k:9s} {p.name}  (missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
