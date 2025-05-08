# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Flask MJPEG streaming server using a CSI camera (such as the Raspberry Pi Version 2)
# connected to a NVIDIA Jetson Nano Developer Kit using OpenCV and GStreamer.
# Streams at 10 FPS and minimizes lag by always sending the latest frame.

from flask import Flask, Response
import cv2
import threading
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1 sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

app = Flask(__name__)

class CameraStream:
    def __init__(self, pipeline):
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            # Sleep a bit to avoid busy-waiting if camera is disconnected
            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

camera_stream = CameraStream(gstreamer_pipeline(framerate=10))

def gen_mjpeg():
    min_interval = 1.0 / 10  # 10 fps
    while True:
        start_time = time.time()
        frame = camera_stream.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        elapsed = time.time() - start_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

@app.route('/')
def video_feed():
    return Response(gen_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import atexit
@atexit.register
def cleanup():
    camera_stream.release()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
