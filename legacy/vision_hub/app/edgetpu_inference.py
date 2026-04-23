from pathlib import Path
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify, detect

_DEFAULT_MODEL = "app/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"

class EdgeTPUModel:
    def __init__(self, model_path: str = _DEFAULT_MODEL):
        self.model_path = str(model_path or _DEFAULT_MODEL)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(self.model_path)
        self.interp = make_interpreter(self.model_path)
        self.interp.allocate_tensors()
        self.w, self.h = common.input_size(self.interp)

    def _set_input(self, rgb_u8: np.ndarray):
        if rgb_u8.dtype != np.uint8:
            rgb_u8 = rgb_u8.astype(np.uint8, copy=False)
        if rgb_u8.shape[:2] != (self.h, self.w):
            try:
                from cv2 import resize, INTER_NEAREST
                rgb_u8 = resize(rgb_u8, (self.w, self.h), interpolation=INTER_NEAREST)
            except Exception:
                rgb_u8 = rgb_u8[:self.h, :self.w, :3]
                if rgb_u8.shape[0] != self.h or rgb_u8.shape[1] != self.w:
                    pad = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                    pad[:rgb_u8.shape[0], :rgb_u8.shape[1]] = rgb_u8
                    rgb_u8 = pad
        common.set_input(self.interp, rgb_u8)

    def run_classify(self, rgb_u8: np.ndarray, top_k=5, threshold=0.0):
        self._set_input(rgb_u8)
        self.interp.invoke()
        out = classify.get_classes(self.interp, top_k=top_k, score_threshold=threshold)
        return [(c.id, float(c.score)) for c in out]

    def run_detect(self, rgb_u8: np.ndarray, threshold=0.3, limit=50):
        self._set_input(rgb_u8)
        self.interp.invoke()
        objs = detect.get_objects(self.interp, score_threshold=threshold)
        res = []
        for o in objs[:limit]:
            b = o.bbox
            res.append({
                "id": int(getattr(o, "id", -1)),
                "score": float(o.score),
                "bbox": [int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)],
            })
        return res

# Back-compat for existing code
class EdgeTPUInference(EdgeTPUModel):
    pass
