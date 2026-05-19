"""
YOLOv8 Nano fine-tuning for CCTV knife detection.
Optimized for Raspberry Pi deployment (small model, 320px input).
"""
from ultralytics import YOLO
import argparse
import os

def train(args):
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=30,
        save=True,
        cache=True,
        augment=True,
        degrees=10,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        close_mosaic=10,
        amp=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="knife_yolov8n")
    args = parser.parse_args()

    train(args)
