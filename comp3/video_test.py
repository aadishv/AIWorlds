import cv2
import time
import logging
from flask import Flask, Response
from flask_cors import CORS


def generate_frames():
    """Generator function to capture frames from /dev/video0 and yield them for streaming."""
    # Open the video device
    cap = cv2.VideoCapture(0)  # 0 corresponds to /dev/video0

    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue

            # Convert to bytes and yield
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Control frame rate
            time.sleep(1/30)
    finally:
        # Release the capture when done
        cap.release()


def create_app():
    # Disable excessive logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route('/video.mjpg')
    def video_feed():
        """Video streaming route. Access this in a browser or other client."""
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return """
        <html>
          <head>
            <title>Video Stream</title>
          </head>
          <body>
            <h1>Video Stream from /dev/video0</h1>
            <img src="/video.mjpg" width="640" height="480" />
          </body>
        </html>
        """

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
