import threading
import time
import signal
import logging
import numpy as np
import copy
from flask import Flask, stream_with_context, Response
from flask_cors import CORS
import cv2
import json

from camera import Camera, Processing
from inference import InferenceEngine
import comms
import localization

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
        self.current_frames = (
            np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint8))
        # inference stuff
        self.engine = InferenceEngine(engine_path)
        # serial stuff
        self.measurement_row = row
        self.localization = None

    def _serial_callback(self, data):
        print("serial callback")
        def keep_ascii(s): return "".join(
            c for c in s if ord(c) < 128 and ord(c) > 0)
        # HANDLE FLAGS
        flag = ""
        # HANDLE LOCALIZATION
        data = keep_ascii(data)
        try:
            data = json.loads(data)
            # print(data, "worked")
        except json.JSONDecodeError:
            # print(
            #     ' '.join(list(map(lambda a: f'|{a}|{ord(a)}|', data.lstrip()))))
            # print(f"Error decoding JSON: {e}")
            return None
        if self.localization is None:
            x = data.get('x', 0)
            y = data.get('y', 0)
            theta = data.get('theta', 0)
            self.localization = localization.Localization((x, y, theta))
            pose = data
        else: # self.localization is not None:
            theta = data.get('theta', 0)
            measurement = self.current_frames[2][self.measurement_row]
            self.localization.update(measurement, theta)
            pose = {
                'x': self.localization.pose[0],
                'y': self.localization.pose[1],
                'theta': self.localization.pose[2]
            }

            # update flag given measurement
            percentile = np.percentile(measurement[measurement > 0], 20) * self.processing.depth_scale
            print(percentile * 3.28)
            if percentile * 3.28 < 1: # that is, at least 20% of nonzero readings are less than 1 foot away
                flag = "STOP"
        # HANDLE OBJECTS
        objects = self.current_detections
        return json.dumps({
            'pose': pose,
            'stuff': objects,
            'flag': flag
        }, indent=None, separators=(',', ':'))

    def serial_worker(self):
        print("serial worker")

        self.localization = None
        self.serial = comms.EventDrivenSerial(
            "/dev/ttyACM1", 115200, self._serial_callback)  # comms.prompt_user_for_port(comms.list_serial_ports())
        self.serial.serial_worker()

    def camera_worker(self):
        print("camera worker")
        try:
            while True:
                # camera stuff
                frames = self.camera.get_frames()
                depth_image, color_image, depth_map = self.processing.process_frames(
                    frames)
                self.current_frames = (color_image, depth_map, depth_image)
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
                try:
                    self._serial_callback('{"imu": 150.0}')
                except Exception as e:
                    print(f"Error sending serial data: {e}")
        finally:
            self.camera.stop()

    def inference_worker(self):
        print("inference worker")
        self.engine.cuda_ctx.push()
        MIN_LATENCY = 1.0 / 20.0
        try:
            while True:
                img = self.current_frames[0]
                start_time = time.time()
                self.current_detections_raw = self.engine.run(img)
                time_elapsed = time.time() - start_time
                if time_elapsed < MIN_LATENCY: # cap at 20 fps
                    time.sleep(MIN_LATENCY - time_elapsed)

        finally:
            # pop when you exit, so you don’t leak
            self.engine.cuda_ctx.pop()
            self.engine.close()

    def app_worker(self):
        # stop excessive logging of requests
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        def generate(type):
            boundary = b'--frame'
            while True:
                # Grab the latest frame + detections
                color_img, depth_map = self.current_frames
                img = (color_img if type == 'color' else depth_map if type ==
                       'depth' else np.zeros_like(color_img)).copy()

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
                time.sleep(1/30)



        app = Flask("3151App")
        CORS(app, resources={r"/*": {"origins": "*"}})

        @app.route('/<type>.mjpg')
        def mjpeg_stream(type):
            """
            MJPEG stream of color frames with overlaid detections.
            """
            return Response(
                stream_with_context(generate(type)),
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


"""
List of workers:
    Camera worker
        * Flagship worker, always running at 30 fps
        * Gets most recent camera frames
    Inference worker
        * Runs below 10 fps
        * Blocking computations on GPU for YOLOv5s/n model
    App worker
        * Generally idle
        * Serves Flask dashboard at localhost:5000 or [ip]:5000
    Serial worker
        * Reads serial data from USB port
        * Responds in JSON
"""
if __name__ == "main":
    shutdown = threading.Event()

    def handle_sigterm(signum, frame):
        shutdown.set()


    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    # initialize worker
    worker = Worker("/home/aadish/AIWorlds/comp2/yolov5n-best.engine", 240)

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
    # serial worker
    t4 = threading.Thread(target=worker.serial_worker, daemon=True)
    threads.append(t4)

    # start them all
    for t in threads:
        t.start()

    # now block here until CTRL‑C or SIGTERM
    shutdown.wait()

    # clean up
    worker.model.close()
    # if your threads check shutdown flag, they can exit cleanly
    for t in threads:
        t.join(timeout=1)
