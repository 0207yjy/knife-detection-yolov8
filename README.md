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

---

## Hailo-8 AI HAT (Hardware Accelerated)

For maximum FPS on Raspberry Pi, deploy with a [Hailo-8 AI HAT](https://www.raspberrypi.com/products/ai-hat/) (26 TOPS).

### Step 1 — Convert ONNX → HEF (run on PC, not RPi)

Install [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/software-downloads/) (requires free registration):

```bash
pip install hailo_dataflow_compiler-*.whl
bash convert_to_hef.sh          # best.onnx → best.hef
```

The script runs three stages automatically:
1. **Parse** — ONNX → Hailo Archive (`.har`)
2. **Optimize** — INT8 quantization using calibration images from `data/images/train/`
3. **Compile** — `.har` → `best.hef` targeting `hailo8`

### Step 2 — Copy HEF to Raspberry Pi

```bash
scp best.hef pi@<rpi-ip>:~/knife-detection-yolov8/
```

### Step 3 — Install Hailo runtime on RPi

Follow the [Hailo RPi5 examples setup guide](https://github.com/hailo-ai/hailo-rpi5-examples):

```bash
sudo apt install hailo-all       # installs driver + hailort
pip install hailo_platform-*.whl # Python bindings (from developer zone)
```

### Step 4 — Run inference

```bash
python detect_hailo.py --weights best.hef --source 0       # USB webcam
python detect_hailo.py --weights best.hef --source picamera
```

### Expected performance (RPi 5 + Hailo-8 HAT)

| Metric | ONNX (CPU) | Hailo-8 HEF |
|--------|-----------|-------------|
| Inference | ~80–120 ms | ~5–10 ms |
| FPS | ~8–12 | ~60–80 |

---

## Project Structure

```
knife-detection-yolov8/
├── data/
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── dataset.yaml           # dataset config
├── train.py               # fine-tuning script
├── export.py              # export to ONNX
├── detect.py              # RPi inference (ONNX / CPU)
├── detect_hailo.py        # RPi inference (Hailo-8 HEF)
├── convert_to_hef.sh      # ONNX → HEF conversion (run on PC)
├── requirements.txt       # training deps
└── requirements_rpi.txt   # RPi runtime deps
```
