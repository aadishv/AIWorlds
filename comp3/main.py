from camera import CameraWorker
from inference import InferenceWorker
from post import Processing
import threading
import time
import signal
import logging
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import cv2
import json
import atexit

class App:
    def __init__(self):
        print("test1")
        self.camera = CameraWorker()
        self.inference = InferenceWorker(self)
        self.most_recent_result = {}

    def service_simulator(self):
        print('test2')
        self.post = Processing(self, (0, 0, 0))
        while True:
            try:
                self.most_recent_result = self.post.update(5)
                print(self.most_recent_result)
            except Exception as e:
                print(f"Error in service_simulator: {e}")
            time.sleep(1.0 / 30.0)

    def run_dashboard(self):
        # stop excessive logging of requests
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

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
                    color_img, depth_img = self.camera.frames
                    # Convert the raw depth (float or uint16) → 0–255 uint8
                    depth_8u = cv2.normalize(
                        depth_img, None,
                        alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U
                    )
                    depth_map = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    img = (color_img if type == 'color' else depth_map if type ==
                           'depth' else np.zeros_like(color_img)).copy()

                    # Draw detections onto img
                    for d in self.most_recent_result['stuff']:
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
                        t_w, t_h = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        txt_y = y0 - 5 if y0 - 5 > 10 else y0 + t_h + 5
                        cv2.rectangle(img, (x0, txt_y - t_h - 4),
                                      (x0 + t_w + 4, txt_y + 2), color, -1)
                        cv2.putText(
                            img, label, (x0+2, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # JPEG‐encode
                    success, jpg = cv2.imencode(
                        '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not success:
                        continue
                    frame = jpg.tobytes()

                    # yield multipart chunk
                    yield boundary + b'\r\n' \
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

                    # throttle to ~30fps
                    time.sleep(1/100)

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
                    data = self.most_recent_result
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

    def close(self):
        self.camera.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ != "main":
    exit()
with App() as app:
    try:
        threads = []
        t1 = threading.Thread(target=app.camera.worker, daemon=True)
        threads.append(t1)
        t2 = threading.Thread(target=app.inference.worker, daemon=True)
        threads.append(t2)
        t3 = threading.Thread(target=app.service_simulator, daemon=True)
        threads.append(t3)
        t4 = threading.Thread(target=app.run_dashboard, daemon=True)
        threads.append(t4)
        # all of the crazy lifecycle management stuff
        shutdown = threading.Event()

        def handle_sigterm(signum, frame):
            shutdown.set()

        signal.signal(signal.SIGINT, handle_sigterm)
        signal.signal(signal.SIGTERM, handle_sigterm)

        atexit.register(app.close)

        for t in threads:
            t.start()

        shutdown.wait()

        for t in threads:
            t.join(timeout=1)
            t.join(timeout=1)
    finally:
        app.close()
