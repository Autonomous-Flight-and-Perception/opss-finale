"""
Cobot Communication Module
Handles communication with the MyCobot280 robot arm via Unix socket.

CANONICAL CONTROL-OUTPUT CONTRACT (schema ``opss.cobot.v1``)
===========================================================

Transport:
    AF_UNIX SOCK_DGRAM at ``/tmp/opss_cobot.sock``. JSON-encoded datagrams.

Canonical producer:
    The live pipeline loop calls :meth:`UnixSocketBroadcaster.send_control_output`
    every pipeline tick. This is the ONLY method that forms part of the
    canonical control contract. All other send methods on this class are
    legacy / debug and are not called by the live pipeline.

Cadence / heartbeat:
    Datagrams are emitted at pipeline FPS (rate-limited to
    ``CobotConfig.send_rate_hz``, default 30 Hz). Every tick produces a
    datagram regardless of target count. This is how the consumer
    distinguishes three states:
      - healthy + targets:  ``targets`` non-empty, recent ``timestamp_ns``
      - healthy + no targets: ``targets`` empty, recent ``timestamp_ns``
      - pipeline dead:       no datagram within the staleness window
                              (recommended: ~200 ms at 30 Hz cadence)

Envelope shape::

    {
      "schema":       "opss.cobot.v1",
      "timestamp_ns": <int>,          # publish-time wall-clock nanoseconds
      "pipeline": {
        "healthy":    <bool>,         # pipeline liveness flag
        "fps":        <float>,        # latest rolling pipeline FPS
        "tracker":    "kalman"|"ukf_nn",
        "frame":      "pixel"|"pixel_xy_metric_z"|"camera_metric"|"world_metric"
      },
      "targets":      [ <per-target dict>, ... ],  # sorted confidence-desc
      "count":        <int>,
    }

Per-target shape is produced by ``FusedState.to_control_output``.
``targets[0]`` is the primary target when the list is non-empty.

Frame / unit semantics:
    ``pipeline.frame`` is the authoritative frame for all targets in the
    datagram. Per-target ``frame`` + ``units`` are also included and MUST
    match the envelope. The four legal frame tags are defined in
    ``opss.state.kalman.ObjectState``:

      - "pixel":              image-plane pixels (no depth)
      - "pixel_xy_metric_z":  x,y pixels; z meters (Kalman hybrid)
      - "camera_metric":      meters, camera frame (identity extrinsics)
      - "world_metric":       meters, world frame (gravity along -z)

    The consumer MUST read ``pipeline.frame`` (or equivalently each target's
    ``frame``) to interpret position/velocity units correctly. No field is
    assumed to be in a fixed frame.

Target identity:
    ``track_id`` is stable across frames per the tracker contract.
    ``targets`` is ordered confidence-descending; ``targets[0]`` is the
    canonical primary target for single-target actuators.

Validity:
    Each target carries an aggregate ``valid`` boolean AND a per-component
    ``validation`` block. Consumers should gate on ``valid`` for simple
    cases and read ``validation.*`` when richer logic is needed.

Legacy / debug methods on this broadcaster:
    - :meth:`send_state`, :meth:`send_states`:
          Retained for programmatic / test use. They produce an older
          envelope shape (no ``schema``, no ``pipeline`` block) and are NOT
          used by the live pipeline loop. Prefer ``send_control_output``.
    - :meth:`send_raw_detections`:
          Debug-only. Raw YOLO detections in pixel space at capture
          resolution. Not part of the canonical control contract and not
          called by the live pipeline.
"""
import socket
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import threading


# Canonical wire schema identifier. Bumped on backward-incompatible changes.
CONTROL_SCHEMA_VERSION = "opss.cobot.v1"


@dataclass
class CobotConfig:
    """Cobot communication configuration"""
    socket_path: str = "/tmp/opss_cobot.sock"
    reconnect_interval: float = 5.0
    send_rate_hz: float = 30.0


class UnixSocketBroadcaster:
    """
    Non-blocking Unix socket broadcaster for cobot communication.

    Canonical entry point is :meth:`send_control_output`. See the module
    docstring for the wire schema (``opss.cobot.v1``).
    """

    def __init__(self, config: Optional[CobotConfig] = None):
        self.config = config or CobotConfig()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.connected = False
        self._last_send_time = 0.0
        self._min_send_interval = 1.0 / self.config.send_rate_hz

        print(f"[COBOT] Broadcaster initialized: {self.config.socket_path}")

    def send_control_output(
        self,
        fused_states: List['FusedState'],
        pipeline_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        CANONICAL control-output broadcast (schema ``opss.cobot.v1``).

        This is the single canonical method used by the live pipeline to
        deliver control data to the cobot. It is designed to be called
        unconditionally every pipeline tick — including when
        ``fused_states`` is empty — so that datagram cadence doubles as a
        heartbeat.

        Args:
            fused_states:  List of ``FusedState`` emitted by B2₃ fusion
                           for this tick. MAY be empty (== no targets
                           right now; this is a valid, meaningful payload).
            pipeline_info: Optional dict with pipeline-level metadata. Keys
                           are propagated verbatim into the envelope's
                           ``pipeline`` block; missing keys fall back to
                           safe defaults. Expected keys:
                             - healthy (bool, default True)
                             - fps     (float, default 0.0)
                             - tracker (str,   default "unknown")
                             - frame   (str,   default derived from
                                        first FusedState or "unknown")

        Returns:
            True if the datagram was written to the socket, False if the
            tick was rate-limited or the socket peer is not bound. A
            dropped tick is not an error — the next tick will send.

        Heartbeat / no-target / stale-data semantics:
            - non-empty targets       -> canonical control update
            - empty targets           -> canonical "no targets" heartbeat
            - no datagram at all      -> pipeline considered dead by the
                                          consumer (staleness window: ~200 ms
                                          at 30 Hz nominal cadence)
        """
        now = time.time()
        if now - self._last_send_time < self._min_send_interval:
            return False  # rate-limited; next tick will send

        # Sort confidence-desc so targets[0] is the canonical primary.
        sorted_states = sorted(
            fused_states,
            key=lambda s: getattr(s, "confidence", 0.0),
            reverse=True,
        )

        # Derive authoritative envelope frame. Prefer caller-supplied value;
        # otherwise take it from the first state; fall back to "unknown" when
        # the list is empty and the caller didn't tell us.
        pipeline_info = dict(pipeline_info or {})
        if "frame" not in pipeline_info:
            if sorted_states:
                pipeline_info["frame"] = getattr(sorted_states[0], "frame", "unknown")
            else:
                pipeline_info["frame"] = "unknown"

        pipeline_block = {
            "healthy": bool(pipeline_info.get("healthy", True)),
            "fps": float(pipeline_info.get("fps", 0.0)),
            "tracker": str(pipeline_info.get("tracker", "unknown")),
            "frame": str(pipeline_info["frame"]),
        }

        data = {
            "schema": CONTROL_SCHEMA_VERSION,
            "timestamp_ns": time.time_ns(),
            "pipeline": pipeline_block,
            "targets": [s.to_control_output() for s in sorted_states],
            "count": len(sorted_states),
        }
        return self._send_data(data)

    def send_state(self, fused_state: 'FusedState') -> bool:
        """
        [LEGACY — not canonical] Send a single fused state.

        Retained for programmatic / test use. The live pipeline uses
        :meth:`send_control_output` instead. Produces a bare per-target
        dict with no ``schema`` / ``pipeline`` envelope.
        """
        now = time.time()
        if now - self._last_send_time < self._min_send_interval:
            return False  # Rate limited

        return self._send_data(fused_state.to_control_output())

    def send_states(self, fused_states: List['FusedState']) -> bool:
        """
        [LEGACY — not canonical] Send multiple fused states.

        Retained for programmatic / test use. The live pipeline uses
        :meth:`send_control_output` instead. Empty list is dropped here
        (no heartbeat); ``send_control_output`` is the heartbeat-capable
        canonical entry point.
        """
        if not fused_states:
            return False

        now = time.time()
        if now - self._last_send_time < self._min_send_interval:
            return False

        data = {
            "timestamp": time.time_ns(),
            "states": [s.to_control_output() for s in fused_states],
            "count": len(fused_states)
        }
        return self._send_data(data)

    def send_raw_detections(self, detections: List[Dict], frame_size: Dict) -> bool:
        """
        [DEBUG — not canonical] Send raw YOLO detections (pixel-space).

        This is the pre-canonicalization payload — raw detections at
        capture resolution with no identity, no velocity, and no frame
        tag. It is NOT part of the canonical cobot control contract and
        is NOT called by the live pipeline. Retained only as a debug /
        diagnostic method.

        Canonical control output: :meth:`send_control_output`.
        """
        data = {
            "timestamp": time.time_ns(),
            "detections": detections,
            "frame_size": frame_size,
            "count": len(detections),
        }
        return self._send_data(data)

    def _send_data(self, data: Dict) -> bool:
        """Internal method to send JSON data over Unix socket"""
        try:
            encoded = json.dumps(data).encode('utf-8')
            self.sock.sendto(encoded, self.config.socket_path)

            self._last_send_time = time.time()

            if not self.connected:
                self.connected = True
                print("[COBOT] Connected via Unix socket")

            return True

        except FileNotFoundError:
            if self.connected:
                print("[COBOT] Disconnected (socket not found)")
                self.connected = False
            return False

        except Exception as e:
            if self.connected:
                print(f"[COBOT] Send error: {e}")
                self.connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self.connected


class CobotStateReceiver:
    """
    Receives state updates from the cobot (optional feedback channel).
    Can be used for closed-loop control verification.
    """

    def __init__(self, listen_path: str = "/tmp/opss_cobot_feedback.sock"):
        self.listen_path = listen_path
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_feedback: Optional[Dict] = None
        self._lock = threading.Lock()

    def start(self):
        """Start listening for cobot feedback"""
        import os

        # Remove existing socket
        try:
            os.unlink(self.listen_path)
        except FileNotFoundError:
            pass

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"[COBOT] Feedback receiver started: {self.listen_path}")

    def stop(self):
        """Stop the feedback receiver"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _listen_loop(self):
        """Background thread to receive feedback"""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(self.listen_path)
        sock.settimeout(0.5)

        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
                feedback = json.loads(data.decode('utf-8'))

                with self._lock:
                    self._latest_feedback = feedback

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[COBOT] Feedback error: {e}")

        sock.close()

    def get_latest_feedback(self) -> Optional[Dict]:
        """Get the most recent feedback from cobot"""
        with self._lock:
            return self._latest_feedback


# Singleton broadcaster
_broadcaster: Optional[UnixSocketBroadcaster] = None


def get_broadcaster(config: Optional[CobotConfig] = None) -> UnixSocketBroadcaster:
    """Get or create the singleton broadcaster"""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = UnixSocketBroadcaster(config)
    return _broadcaster
