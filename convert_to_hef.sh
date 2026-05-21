#!/usr/bin/env bash
# Convert best.onnx → best.hef for Hailo-8 using Hailo Dataflow Compiler (DFC).
# Run this on a PC/server — NOT on the Raspberry Pi itself.
#
# Prerequisites:
#   1. Register at https://hailo.ai/developer-zone/
#   2. Download and install hailo_dataflow_compiler wheel:
#      pip install hailo_dataflow_compiler-*.whl
#
# After conversion, copy best.hef to your RPi and run:
#   python detect_hailo.py --weights best.hef --source 0

set -euo pipefail

NET="knife_yolov8n"
HW="hailo8"
ONNX="${1:-best.onnx}"
CALIB="${2:-./data/images/train}"
CALIB_SIZE=64

echo "=== Hailo-8 HEF Conversion ==="
echo "  ONNX:  $ONNX"
echo "  Calib: $CALIB ($CALIB_SIZE images)"
echo ""

echo "[1/3] Parse ONNX → HAR"
hailo parser onnx "$ONNX" \
  --net-name "$NET" \
  --hw-arch "$HW"

echo "[2/3] Optimize (INT8 post-training quantization)"
hailo optimize "${NET}.har" \
  --hw-arch "$HW" \
  --calib-set-path "$CALIB" \
  --calib-set-size "$CALIB_SIZE"

echo "[3/3] Compile → HEF"
hailo compile "${NET}_optimized.har" \
  --hw-arch "$HW"

mv "${NET}_optimized.hef" best.hef
echo ""
echo "Done → best.hef  (copy to Raspberry Pi)"
