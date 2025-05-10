from flask import Flask, Response, send_from_directory
from flask_cors import CORS, cross_origin
import cv2
import os

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})

# Serve the standalone HTML file


@app.route('/')
def index():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'index.html')


def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def gen_frames():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/video_feed')
@cross_origin(origins="*")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve app.js from root for browser access


@app.route('/app.js')
def serve_app_js():
    return send_from_directory(app.root_path, 'app.js')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
