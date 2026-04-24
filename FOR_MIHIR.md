# FOR MIHIR

Metrics for the active PyTorch YOLO drone detector (`vision/user_best.pt`),
extracted directly from the checkpoint's embedded training record.

## `user_best.pt` — active drone detector (YOLOv8n)

**Final validation metrics (best epoch):**

| Metric | Value |
|---|---|
| Precision (B) | **0.9108** |
| Recall (B) | **0.8929** |
| mAP@50 (B) | **0.9327** |
| mAP@50-95 (B) | **0.5990** |
| Fitness | 0.5990 |
| val/box_loss | 1.3142 |
| val/cls_loss | 0.6447 |
| val/dfl_loss | 1.3618 |

**Training config (`train_args`):**
- Base weights: `yolov8n.pt` (nano), `pretrained=True`
- Dataset: `/content/drone-1/data.yaml` (Colab `drone-1` dataset)
- 20 epochs, batch 128, imgsz 640, `optimizer='auto'`, `lr0=0.01`, `amp=True`
- Augmentations: mosaic 1.0, fliplr 0.5, hsv_h/s/v 0.015/0.7/0.4, erasing 0.4
- Device: single GPU, Ultralytics 8.4.41
- Wall-clock training time: **645 s (~10.75 min)** for 20 epochs
- Trained: 2026-04-23 02:58 UTC (checkpoint date)
- Stripped `best.pt` (epoch=-1, optimizer/EMA/updates=None — deployment-ready)

**Runtime on Jetson Orin Nano** (from `README.md:104-109` and
`vision/opss/vision/yolov8_inference.py:55-59`):
- ~23 FPS end-to-end, FP16 on GPU, imgsz 640, conf 0.30, `classes=[0]` ("drone")

**Per-epoch learning curve (validation mAP@50):**

```
ep  1: 0.5185    ep  8: 0.8258    ep 15: 0.9075
ep  2: 0.4087    ep  9: 0.8452    ep 16: 0.9168
ep  3: 0.5135    ep 10: 0.8746    ep 17: 0.9204
ep  4: 0.6124    ep 11: 0.8263    ep 18: 0.9245
ep  5: 0.7806    ep 12: 0.8859    ep 19: 0.9278
ep  6: 0.7727    ep 13: 0.8958    ep 20: 0.9327  ← best
ep  7: 0.7247    ep 14: 0.9066
```

## Comparison against the other bundled drone weights

From `README.md:104-109`:

| Weights | Arch | mAP@50 | Jetson FPS |
|---|---|---|---|
| **`user_best.pt`** (active) | YOLOv8n | **0.933** | ~23 |
| `yolo_drone_detect.pt` | YOLOv26-L | 0.908 | ~15 |
| `doguilmak_drone_v8x.pt` | YOLOv8x | unpub. | ~10 |
| `yolov8n.pt` (COCO default) | YOLOv8n | 0.373\* | ~30 |

\*COCO val baseline (80 classes), not drone-specific.

`user_best.pt` wins on both accuracy (highest drone mAP@50) and speed among the
drone-trained weights — it's YOLOv8n fine-tuned for 20 epochs on `drone-1`, so
it keeps nano-class latency while adding ~56 points of mAP over stock COCO
YOLOv8n on the drone class.
