## EdgeTPU Runtime (Python 3.9)
- Activate: `source venv39/bin/activate`
- Install stack: `pip install -r constraints-edgetpu.txt --no-deps`
- Detect: `python -m app.edgetpu_cli --mode detect --model app/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --image sample.jpg --thresh 0.2`
- Classify: `python -m app.edgetpu_cli --mode classify --model app/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite --image sample.jpg`
