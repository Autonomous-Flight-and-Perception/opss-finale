import cv2
from app.edgetpu_inference import EdgeTPUModel

_MODEL_PATH = "app/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
_TPU = EdgeTPUModel(_MODEL_PATH)  # exposes _TPU.w, _TPU.h

def process_frame_bgr(bgr, thresh: float = 0.55):
    """
    Process frame and draw bounding boxes.
    
    Returns:
        tuple: (annotated_frame, detections_list)
            - annotated_frame: BGR image with drawn boxes
            - detections_list: List of detection dicts with corner coordinates
    """
    ih, iw = _TPU.h, _TPU.w
    fh, fw = bgr.shape[:2]
    sx, sy = fw / float(iw), fh / float(ih)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    objs = _TPU.run_detect(rgb, threshold=thresh)
    
    detections = []
    
    for o in objs:
        if o["id"] != 0:  # COCO person only
            continue
        x1,y1,x2,y2 = o["bbox"]
        x1 = max(0, min(fw-1, int(x1*sx)))
        y1 = max(0, min(fh-1, int(y1*sy)))
        x2 = max(0, min(fw-1, int(x2*sx)))
        y2 = max(0, min(fh-1, int(y2*sy)))
        
        # Store detection with all 4 corners
        detections.append({
            "class": "person",
            "confidence": float(o["score"]),
            "corners": {
                "top_left": {"x": x1, "y": y1},
                "top_right": {"x": x2, "y": y1},
                "bottom_left": {"x": x1, "y": y2},
                "bottom_right": {"x": x2, "y": y2}
            },
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center": {
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2)
            }
        })
        
        # Draw on frame
        cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(bgr,f'person:{o["score"]:.2f}',(x1,max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    
    return bgr, detections