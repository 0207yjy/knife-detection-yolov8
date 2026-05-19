"""
Real-time knife detection for Raspberry Pi using ONNX model.
Supports Pi Camera (libcamera), USB webcam, or video file.

Usage:
    python detect.py --weights best.onnx --source 0          # USB webcam
    python detect.py --weights best.onnx --source picamera   # Pi Camera v2/v3
    python detect.py --weights best.onnx --source video.mp4  # video file
    python detect.py --weights best.pt   --source 0          # .pt model (requires GPU/CPU torch)
"""
import cv2
import numpy as np
import argparse
import time
import sys
import os

CLASSES = ["person", "knife"]
COLORS = [(0, 255, 0), (0, 0, 255)]  # green=person, red=knife


def letterbox(img, new_shape=320):
    h, w = img.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))
    pad_h = (new_shape - nh) // 2
    pad_w = (new_shape - nw) // 2
    img = cv2.copyMakeBorder(img, pad_h, new_shape - nh - pad_h,
                             pad_w, new_shape - nw - pad_w,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, scale, pad_w, pad_h


def postprocess(outputs, orig_h, orig_w, scale, pad_w, pad_h,
                conf_thres=0.4, iou_thres=0.45):
    preds = outputs[0]
    if preds.ndim == 3:
        preds = preds[0]          # (8400, 6) for nc=2
    preds = preds.T               # (6, 8400)

    boxes = preds[:4].T           # cx,cy,w,h
    scores = preds[4:].T          # (8400, nc)
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    mask = confs > conf_thres
    boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]

    if len(boxes) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2 in padded-image coords
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # remove letterbox padding and scale back to original
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    bboxes = np.stack([x1, y1, x2, y2], axis=1)
    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), confs.tolist(), conf_thres, iou_thres
    )
    if len(indices) == 0:
        return []

    results = []
    for i in indices.flatten():
        results.append({
            "box": bboxes[i].astype(int).tolist(),
            "conf": float(confs[i]),
            "class_id": int(class_ids[i]),
            "label": CLASSES[int(class_ids[i])],
        })
    return results


def draw(frame, detections):
    knife_detected = False
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["class_id"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if d["class_id"] == 1:
            knife_detected = True

    if knife_detected:
        cv2.putText(frame, "!! KNIFE DETECTED !!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return frame


def open_picamera(imgsz):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"size": (imgsz, imgsz), "format": "RGB888"}
        ))
        cam.start()
        return cam, "picamera2"
    except ImportError:
        print("picamera2 not found, falling back to /dev/video0")
        return cv2.VideoCapture(0), "cv2"


def run_onnx(args):
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        args.weights,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    imgsz = args.imgsz

    if args.source == "picamera":
        cap, cap_type = open_picamera(imgsz)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        cap_type = "cv2"
        if not cap.isOpened():
            sys.exit(f"Cannot open source: {args.source}")

    fps_counter, fps, t_start = 0, 0.0, time.time()

    while True:
        if cap_type == "picamera2":
            frame = cap.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        orig_h, orig_w = frame.shape[:2]
        img, scale, pad_w, pad_h = letterbox(frame, imgsz)
        blob = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, norm
        blob = np.transpose(blob, (2, 0, 1))[None]         # NCHW

        outputs = sess.run(None, {input_name: blob})
        dets = postprocess(outputs, orig_h, orig_w, scale, pad_w, pad_h,
                           args.conf, args.iou)
        frame = draw(frame, dets)

        fps_counter += 1
        if fps_counter % 10 == 0:
            fps = 10 / (time.time() - t_start)
            t_start = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if args.show:
            cv2.imshow("Knife Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save_video:
            pass  # handled below

    if cap_type == "picamera2":
        cap.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()


def run_pt(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    src = int(args.source) if args.source.isdigit() else args.source
    model.predict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=args.save_video,
        stream=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best.onnx",
                        help=".onnx (RPi) or .pt (torch)")
    parser.add_argument("--source", default="0",
                        help="0=webcam, picamera, or path to video")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    if args.weights.endswith(".onnx"):
        run_onnx(args)
    else:
        run_pt(args)
