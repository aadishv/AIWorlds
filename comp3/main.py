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
        self.camera = CameraWorker()
        self.inference = InferenceWorker(self)
        self.most_recent_result = {}

    def service_simulator(self):
        self.post = Processing(self, (0, 0, 0), self.camera._processing.fl)
        while True:
            start = time.time()
            self.most_recent_result = self.post.update(5)
            time.sleep(max(0, 1.0 / 30.0 - (time.time() - start)))

    def run_dashboard(self):
        # stop excessive logging of requests
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        app = Flask("3151App")
        CORS(app, resources={r"/*": {"origins": "*"}})

        def generate_frames(type):
            """Generator function to capture frames and yield them for streaming."""
            while True:
                color_img, depth_img = self.camera.frames
                depth_8u = cv2.normalize(
                    -depth_img, None,
                    alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U
                )
                depth_map = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                frame = (color_img if type == 'color' else depth_map if type ==
                         'depth' else np.zeros_like(color_img)).copy()
                # Encode the frame as JPEG
                frame = cv2.resize(frame, (320, 240))
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Failed to encode frame")
                    continue  # Skip if encoding failed

                # Convert the buffer to bytes
                frame_bytes = buffer.tobytes()

                # Yield the frame in the multipart format
                # This format is understood by browsers for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Optional: Control frame rate (e.g., limit to 15 FPS)
            # time.sleep(1/15)

        @app.route('/<type>.mjpg')
        def video_feed(type):
            """Video streaming route. Access this in your browser."""
            # The Response object takes the generator function and streams its output
            return Response(generate_frames(type),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

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
        self.camera.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with App() as app:
        try:
            threads = []
            t1 = threading.Thread(target=app.camera.worker,
                                  name="camera_worker", daemon=True)
            threads.append(t1)
            t2 = threading.Thread(
                target=app.inference.worker, name="inference_worker", daemon=True)
            threads.append(t2)
            t3 = threading.Thread(
                target=app.service_simulator, name="service_simulator", daemon=True)
            threads.append(t3)
            t4 = threading.Thread(target=app.run_dashboard,
                                  name="dashboard_worker", daemon=True)
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
