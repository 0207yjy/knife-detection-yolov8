"""
Export trained YOLOv8 model to ONNX format for Raspberry Pi deployment.
ONNX with opset 12 runs on RPi via onnxruntime without GPU.
"""
from ultralytics import YOLO
import argparse


def export(args):
    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"Exported to: {args.weights.replace('.pt', '.onnx')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/knife_yolov8n/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    export(args)
