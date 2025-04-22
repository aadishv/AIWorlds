import sys
import time
import json
import signal
import threading
import multiprocessing as mp
from multiprocessing import Manager, Queue, Event
import queue
import colorsys

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template

# your TRT inference wrapper
from inference import InferenceEngine

app = Flask(__name__)

# Globals for the Flask thread
global_frame = None  # latest BGR frame
shared_det = None  # will be set to Manager().dict()

# -----------------------------------------------------------------------------
# MJPEG STREAM + DRAW
# -----------------------------------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(_gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _gen_frames():
    global global_frame, shared_det

    class_colors = {0: (255, 0, 0), 1: (255, 255, 0),
                    2: (0, 255, 0), 3: (0, 0, 255)}
    last_classes = None

    while True:
        if global_frame is None:
            time.sleep(0.01)
            continue

        frame = global_frame.copy()

        # pick up classes once from shared_det
        cls_list = shared_det.get("classes")
        if cls_list and cls_list != last_classes:
            class_colors.clear()
            for i, _cls in enumerate(cls_list):
                hue = i / len(cls_list)
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                # convert to BGR 0-255
                # class_colors[i] = (int(b*255), int(g*255), int(r*255))
            last_classes = cls_list

        dets = shared_det.get("detections", [])

        # draw boxes + labels
        for det in dets:
            h, w = frame.shape[:2]                 # e.g. 480, 640
            model_h, model_w = 640, 640            # your inference input size

            scale_x = w / model_w                  # 640/640 = 1.0
            scale_y = h / model_h                  # 480/640 = 0.75

            x1 = int(det["x1"] * scale_x)
            y1 = int(det["y1"] * scale_y)
            x2 = int(det["x2"] * scale_x)
            y2 = int(det["y2"] * scale_y)

            cls_idx = int(det["class_index"])
            label = f"{det['class_name']}:{det['confidence']:.2f}"
            color = class_colors.get(cls_idx, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, 2, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                          (x1, y1 - th - 4),
                          (x1 + tw + 2, y1),
                          color, thickness=cv2.FILLED)
            cv2.putText(frame, label, (x1+1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        jpg = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(1/30)

# -----------------------------------------------------------------------------
# CAMERA CAPTURE
# -----------------------------------------------------------------------------


def capture_loop(frame_queue: Queue, stop_event: Event):
    global global_frame
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("📸 RealSense capture started.")

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue
            img = np.asanyarray(c.get_data())  # BGR

            global_frame = img
            if not frame_queue.full():
                frame_queue.put(img)
    finally:
        pipeline.stop()
        print("📸 RealSense stopped.")

# -----------------------------------------------------------------------------
# INFERENCE (separate process)
# -----------------------------------------------------------------------------


def inference_loop(frame_queue: Queue,
                   stop_event: Event,
                   shared_det,
                   engine_path: str,
                   output_json: str):
    print("🤖 Loading TensorRT engine…")
    engine = InferenceEngine(engine_path)
    print("✅ Engine loaded, starting inference.")

    # publish class names once
    shared_det["classes"] = engine.nms_params["classes"]

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            dets = engine.run(frame) or []
            t1 = time.time()

            # write JSON
            out = {
                "timestamp": time.time(),
                "inference_time_s": t1 - t0,
                "detections": dets
            }
            with open(output_json, 'w') as f:
                json.dump(out, f, indent=2)

            # share for drawing
            shared_det["detections"] = dets

    finally:
        engine.close()
        print("🤖 Inference loop stopped.")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main():
    global shared_det

    engine_path = "best.engine"
    output_json = "latest_detections.json"

    mgr = Manager()
    shared_det = mgr.dict()
    frame_q = mp.Queue(maxsize=1)
    stop_evt = mp.Event()

    # inference in its own process (isolated CUDA context)
    inf_proc = mp.Process(
        target=inference_loop,
        args=(frame_q, stop_evt, shared_det, engine_path, output_json),
        daemon=True
    )
    inf_proc.start()

    # camera capture in a thread
    cap_thr = threading.Thread(
        target=capture_loop,
        args=(frame_q, stop_evt),
        daemon=True
    )
    cap_thr.start()

    # clean shutdown
    def _shutdown(sig, frame):
        print("🛑 Shutting down…")
        stop_evt.set()
        inf_proc.join(1.0)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # start Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == '__main__':
    main()
