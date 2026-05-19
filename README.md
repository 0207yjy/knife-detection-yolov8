# CCTV Knife Detection — YOLOv8 Nano

Real-time knife detection fine-tuned on the [CCTV Knife Detection Dataset](https://www.kaggle.com/datasets/simuletic/cctv-knife-detection-dataset), optimized for Raspberry Pi deployment.

| Class | ID |
|-------|----|
| person | 0 |
| knife  | 1 |

## Model

- Base: **YOLOv8 Nano** (`yolov8n.pt`)
- Input size: **320×320** (RPi-friendly)
- Export format: **ONNX** (runs on RPi via `onnxruntime`, no PyTorch needed)

## Dataset

114 CCTV scenario images, split 80/20 train/val.  
License: CC-BY-NC-SA-4.0

---

## Training (PC / Server)

```bash
pip install -r requirements.txt
python train.py --data dataset.yaml --epochs 100 --imgsz 320 --batch 16 --device 0
```

Results saved to `runs/train/knife_yolov8n/`.

## Export to ONNX

```bash
python export.py --weights runs/train/knife_yolov8n/weights/best.pt --imgsz 320
# → runs/train/knife_yolov8n/weights/best.onnx
```

Copy `best.onnx` to your Raspberry Pi.

---

## Raspberry Pi Setup

```bash
# Install dependencies (no PyTorch)
pip install -r requirements_rpi.txt

# USB webcam
python detect.py --weights best.onnx --source 0

# Pi Camera v2/v3 (requires picamera2)
python detect.py --weights best.onnx --source picamera

# Video file
python detect.py --weights best.onnx --source video.mp4

# Press Q to quit
```

### Recommended Pi hardware
- Raspberry Pi 4 (2 GB+) — ~8–12 FPS at 320px
- Raspberry Pi 5 — ~20 FPS at 320px
- Pi Camera v2 or v3 (or any USB webcam)

### Expected performance (RPi 4, 320px ONNX)

| Metric | Value |
|--------|-------|
| Inference | ~80–120 ms |
| FPS | ~8–12 |
| RAM | ~200 MB |

---

## Project Structure

```
knife-detection-yolov8/
├── data/
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── dataset.yaml          # dataset config
├── train.py              # fine-tuning script
├── export.py             # export to ONNX
├── detect.py             # RPi real-time inference
├── requirements.txt      # training deps
└── requirements_rpi.txt  # RPi runtime deps
```
