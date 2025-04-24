import threading
import time
import signal
import sys
import logging
import numpy as np
import copy
from flask import Flask, render_template, Response, jsonify, stream_with_context, Response
from flask_cors import CORS
import cv2
import json

from camera import Camera, Processing
from inference import InferenceEngine
import comms
import localization

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

shutdown = threading.Event()

def handle_sigterm(signum, frame):
    shutdown.set()

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

"""
what they send us: {x?, y?, theta}
what we send them: {
stuff: [{x, y, width, height, class(str), depth, confidence}]
flag(str)
pose {x, y, theta}
}
"""

class Worker:
    def __init__(self, engine_path, row):
        # camera stuff
        self.camera = Camera()
        self.camera.start()
        self.processing = Processing(self.camera.depth_scale)
        self.current_detections_raw = []
        self.current_detections = []
        self.current_frames = (np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint8))
        # inference stuff
        self.engine = InferenceEngine(engine_path)
        # serial stuff
        self.measurement_row = row
        self.localization = None

    def serial_callback(self, data):
        # HANDLE LOCALIZATION
        if (self.localization is None) or ('x' in data and 'y' in data and 'theta' in data):
            x = data.get('x', 0)
            y = data.get('y', 0)
            theta = data.get('theta', 0)
            self.localization = localization.Localization((x, y, theta))
            pose = data
        elif self.localization:
            theta = data.get('theta', 0)
            measurement = self.current_frames[1][self.measurement_row]
            self.localization.update(measurement, theta)
            pose = {
                'x': self.localization.pose[0],
                'y': self.localization.pose[1],
                'theta': self.localization.pose[2]
            }
        # HANDLE OBJECTS
        objects = self.current_detections
        # HANDLE FLAGS
        flag = ""
        return json.dumps({
            'pose': pose,
            'stuff': objects,
            'flag': flag
        }, indent=None, separators=(',', ':'))

    def serial_worker(self):
        print("serial worker")
        self.localization = None
        self.serial = comms.EventDrivenSerial(comms.prompt_user_for_port(), 9600, self.serial_callback)
        self.serial.serial_worker()

    def camera_worker(self):
        print("camera worker")
        try:
            while True:
                # camera stuff
                frames = self.camera.get_frames()
                depth_image, color_image, depth_map = self.processing.process_frames(
                    frames)
                self.current_frames = (color_image, depth_map)
                # processing stuff
                detections = copy.deepcopy(self.current_detections_raw)
                for detection in detections:
                    # 480, 640
                    detection['y'] *= 480 / 640
                    depth_value = self.processing.get_depth(
                        detection, depth_image)
                    if np.isnan(depth_value):
                        detection['depth'] = -1
                    else:
                        detection['depth'] = depth_value
                self.current_detections = [
                    i for i in detections if
                        i['class'] not in ['red', 'blue'] or
                        i['confidence'] > 0.6
                ]
                print(self.serial_callback({}))
        finally:
            self.camera.stop()

    def inference_worker(self):
        print("inference worker")
        self.engine.cuda_ctx.push()
        try:
            while True:
                img = self.current_frames[0]
                self.current_detections_raw = self.engine.run(img)
                time.sleep(0.01)
        finally:
            # pop when you exit, so you don’t leak
            self.engine.cuda_ctx.pop()
            self.engine.close()

    def app_worker(self):
        app = Flask("3151App")
        CORS(app, resources={r"/*": {"origins": "*"}})

        @app.route('/<type>.mjpg')
        def mjpeg_stream(type):
            """
            MJPEG stream of color frames with overlaid detections.
            """
            def generate():
                boundary = b'--frame'
                while True:
                    # Grab the latest frame + detections
                    color_img, depth_map = self.current_frames
                    img = (color_img if type == 'color' else depth_map if type == 'depth' else np.zeros_like(color_img)).copy()

                    # Draw detections onto img
                    for d in self.current_detections:
                        x, y = d['x'], d['y']
                        w, h = d['width'], d['height']
                        cls = d['class']
                        conf = d['confidence']
                        depth = d.get('depth', None)

                        # top‑left corner
                        x0 = int(x - w/2)
                        y0 = int(y - h/2)
                        x1 = int(x + w/2)
                        y1 = int(y + h/2)

                        # choose a color
                        color = (0, 0, 255)
                        if cls == 'blue':
                            color = (255, 0, 0)
                        elif cls == 'goal':
                            color = (0, 215, 255)
                        elif cls == 'red':
                            color = (0, 0, 255)
                        elif cls == 'bot':
                            color = (0, 0, 0)

                        # rectangle
                        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                        # label text
                        label = f"{cls} {conf:.2f}"
                        if depth is not None and depth >= 0:
                            label += f" d={depth:.2f}m"
                        # putText above box
                        t_w, t_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        txt_y = y0 - 5 if y0 - 5 > 10 else y0 + t_h + 5
                        cv2.rectangle(img, (x0, txt_y - t_h - 4), (x0 + t_w + 4, txt_y + 2), color, -1)
                        cv2.putText(img, label, (x0+2, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # JPEG‐encode
                    success, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not success:
                        continue
                    frame = jpg.tobytes()

                    # yield multipart chunk
                    yield boundary + b'\r\n' \
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

                    # throttle to ~30fps
                    time.sleep(1/30)

            return Response(
                stream_with_context(generate()),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/events')
        def sse_events():
            """
            Server‑Sent Events stream of raw detection JSON.
            """
            def generate():
                while True:
                    data = {"stuff": self.current_detections}
                    payload = f"data: {json.dumps(data)}\n\n"
                    yield payload
                    time.sleep(1/30)

            headers = {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
            return Response(stream_with_context(generate()), headers=headers)

        # A simple landing page to consume them
        @app.route('/')
        def index():
            with open('visualization.html', 'r') as f:
                return f.read()
        # Run Flask
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

worker = Worker("/home/aadish/AIWorlds/comp2/yolov5s-best.engine", 240)

threads = []
# camera worker
t1 = threading.Thread(target=worker.camera_worker, daemon=True)
threads.append(t1)
# inference worker
t2 = threading.Thread(target=worker.inference_worker, daemon=True)
threads.append(t2)
# app worker
t3 = threading.Thread(target=worker.app_worker, daemon=True)
threads.append(t3)

# start them all
for t in threads:
    t.start()

# now block here until CTRL‑C or SIGTERM
shutdown.wait()

# clean up
model.close()
# if your threads check shutdown flag, they can exit cleanly
for t in threads:
    t.join(timeout=1)
