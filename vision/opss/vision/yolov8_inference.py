import numpy as np
import cv2
from typing import List, Dict, Tuple
from pathlib import Path
import threading

# Lazy torch import — avoid crashing at module load on Jetson (libcudnn mismatch)
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class YOLOv8Detector:
    def __init__(self, model_path: str = "user_best.pt", conf_threshold: float = 0.30):
        print(f"[YOLO] Initializing YOLOv8 detector...")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            print(f"[YOLO] ERROR: ultralytics not installed!")
            print(f"[YOLO] Run: pip install ultralytics")
            raise e

        torch = _get_torch()

        self.conf_threshold = conf_threshold
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[YOLO] Using device: {self.device}")
        
        if self.device == "cpu":
            print(f"[YOLO] WARNING: Running on CPU - will be slow!")
        
        # Load model
        try:
            self.model = YOLO(model_path)
            print(f"[YOLO] Model loaded: {model_path}")
        except Exception as e:
            print(f"[YOLO] ERROR: Failed to load model {model_path}")
            print(f"[YOLO] Error: {e}")
            raise e
        
        # Move to GPU if available
        torch = _get_torch()
        if torch.cuda.is_available():
            try:
                self.model.to(self.device)
                print("[YOLO] Model moved to GPU")
                
                # Enable half precision on MODEL (not in predict call)
                try:
                    self.model.model.half()
                    print("[YOLO] Half-precision (FP16) enabled - 2x faster!")
                except Exception as e:
                    print(f"[YOLO] Half-precision not supported: {e}")
            except Exception as e:
                print(f"[YOLO] Failed to move to GPU: {e}")
        
        # Warm up the model with inference resolution (640x360 for 16:9)
        print("[YOLO] Warming up model...")
        try:
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)  # Match inference resolution
            _ = self.model(
                dummy,
                conf=self.conf_threshold,
                classes=[0],
                verbose=False,
                device=self.device,
                imgsz=640,
                max_det=20
            )
            print("[YOLO] Model ready (optimized for 640x360 inference)")
        except Exception as e:
            print(f"[YOLO] Warmup failed: {e}")
    
    def detect_persons_only(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        CORRECT box parsing for Ultralytics YOLO
        """
        try:
            torch = _get_torch()
            # Run inference with torch.inference_mode()
            with torch.inference_mode():
                r = self.model(
                    image,
                    conf=self.conf_threshold,
                    classes=[0],  # Person only
                    verbose=False,
                    device=self.device,
                    imgsz=640,
                    max_det=20
                    # NOTE: No 'half=' parameter - FP16 is set on model itself
                )[0]
            
            detections = []
            annotated = image.copy()
            
            b = r.boxes  # Boxes object
            if b is not None and len(b) > 0:
                # CORRECT: Extract all boxes at once
                xyxy = b.xyxy.detach().cpu().numpy().astype(int)   # (N, 4)
                confs = b.conf.detach().cpu().numpy()              # (N,)
                
                # Iterate over arrays
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    detections.append({
                        "class": "drone",
                        "confidence": float(conf),
                        "corners": {
                            "top_left": {"x": int(x1), "y": int(y1)},
                            "top_right": {"x": int(x2), "y": int(y1)},
                            "bottom_left": {"x": int(x1), "y": int(y2)},
                            "bottom_right": {"x": int(x2), "y": int(y2)},
                        },
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "center": {"x": int((x1 + x2) // 2), "y": int((y1 + y2) // 2)},
                    })
                    
                    # Draw on frame
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"drone: {conf:.2f}",
                        (int(x1), max(int(y1) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
            
            return annotated, detections
        
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return image, []


# Singleton pattern for detector
_detector = None
_detector_lock = threading.Lock()


def process_frame_bgr(bgr: np.ndarray, thresh: float = 0.30, persons_only: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Thread-safe singleton pattern
    """
    global _detector, _detector_lock
    
    # Initialize detector on first call
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                try:
                    _detector = YOLOv8Detector(conf_threshold=thresh)
                except Exception as e:
                    print(f"[YOLO] Failed to initialize: {e}")
                    return bgr, []
    
    # Run detection
    try:
        return _detector.detect_persons_only(bgr)
    except Exception as e:
        print(f"[YOLO] process_frame_bgr error: {e}")
        import traceback
        traceback.print_exc()
        return bgr, []


__all__ = ['YOLOv8Detector', 'process_frame_bgr']
