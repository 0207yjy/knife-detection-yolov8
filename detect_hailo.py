"""
Real-time knife detection for Raspberry Pi + Hailo-8 AI HAT.
Runs best.hef on the Hailo-8 chip for hardware-accelerated inference.

Prerequisites on RPi:
    1. Install Hailo AI HAT software stack from hailo.ai/developer-zone/
       (installs hailo_platform Python package and PCIe driver)
    2. Copy best.hef from your PC after running convert_to_hef.sh

Usage:
    python detect_hailo.py --weights best.hef --source 0          # USB webcam
    python detect_hailo.py --weights best.hef --source picamera   # Pi Camera
    python detect_hailo.py --weights best.hef --source video.mp4  # video file
"""
import argparse
import sys
import time

import cv2
import numpy as np

from detect import CLASSES, COLORS, draw, letterbox, open_picamera, postprocess


def run_hailo(args):
    try:
        import hailo_platform as hp
    except ImportError:
        sys.exit(
            "hailo_platform not found.\n"
            "Install the Hailo AI HAT software stack from hailo.ai/developer-zone/"
        )

    hef = hp.HEF(args.weights)

    with hp.VDevice() as target:
        cfg_params = hp.ConfigureParams.create_from_hef(
            hef=hef, interface=hp.HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, cfg_params)[0]
        ng_params = network_group.create_params()

        in_params = hp.InputVStreamParams.make(
            network_group, quantized=False, format_type=hp.FormatType.FLOAT32
        )
        out_params = hp.OutputVStreamParams.make(
            network_group, quantized=False, format_type=hp.FormatType.FLOAT32
        )
        input_name = list(in_params.keys())[0]

        if args.source == "picamera":
            cap, cap_type = open_picamera(args.imgsz)
        else:
            src = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(src)
            cap_type = "cv2"
            if not cap.isOpened():
                sys.exit(f"Cannot open source: {args.source}")

        fps_counter, fps, t_start = 0, 0.0, time.time()

        with hp.InferVStreams(network_group, in_params, out_params) as pipeline:
            with network_group.activate(ng_params):
                while True:
                    if cap_type == "picamera2":
                        frame = cap.capture_array()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break

                    orig_h, orig_w = frame.shape[:2]
                    img, scale, pad_w, pad_h = letterbox(frame, args.imgsz)

                    # BGR→RGB, normalize to [0,1], add batch dim → NHWC float32
                    blob = img[:, :, ::-1].astype(np.float32) / 255.0
                    blob = blob[np.newaxis]  # (1, 320, 320, 3)

                    outputs = pipeline.infer({input_name: blob})

                    # Hailo output may be (1, H, W, C); collapse to (1, N, 6)
                    raw = list(outputs.values())[0]
                    if raw.ndim == 4:
                        raw = raw.reshape(1, -1, raw.shape[-1])
                    # Transpose inner axes to match ONNX (1, 6, 8400) → reuse postprocess
                    raw = raw.transpose(0, 2, 1)

                    dets = postprocess(
                        [raw], orig_h, orig_w, scale, pad_w, pad_h,
                        args.conf, args.iou
                    )
                    frame = draw(frame, dets)

                    fps_counter += 1
                    if fps_counter % 10 == 0:
                        fps = 10 / (time.time() - t_start)
                        t_start = time.time()
                    cv2.putText(
                        frame, f"FPS: {fps:.1f} [Hailo-8]",
                        (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2
                    )

                    if args.show:
                        cv2.imshow("Knife Detection (Hailo-8)", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        if cap_type == "picamera2":
            cap.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best.hef")
    parser.add_argument("--source", default="0",
                        help="0=webcam, picamera, or path to video")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--show", action="store_true", default=True)
    args = parser.parse_args()

    run_hailo(args)
