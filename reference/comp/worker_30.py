
import random
import threading
import time
import requests
from utils import schema
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from flask_cors import CORS


# Global variables to hold the latest image and a lock for thread safety
latest_image = None
latest_depth = None
image_lock = threading.Lock()
depth_lock = threading.Lock()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


most_recent_serialization = None


def run_serialization_prep():
    try:
        response = requests.get(schema.INFERENCE_5_URL + '/detections')
        response.raise_for_status()
        raw = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching detections: {e}")
        return
    obj = {}
    # TODO: DO THIS!
    obj['points'] = [random.random() * 100 for _ in range(20)]
    # TODO: DO THIS! (add depth data)
    obj['dets'] = raw


def main_loop():
    global latest_image, latest_depth
    pipeline = None
    try:
        # 1) Start RealSense
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        align_to = rs.stream.color
        align = rs.align(align_to)
        pipeline.start(cfg)
        print("📸 RealSense started.")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Update the global latest_image and latest_depth
            with image_lock:
                latest_image = img.copy()
            with depth_lock:
                latest_depth = depth.copy()
            # Run the existing serialization prep
            run_serialization_prep()

            # small sleep to prevent a tight loop if anything goes wrong
            time.sleep(0.001)

    except Exception as e:
        print("Error in camera loop:", e)

    finally:
        if pipeline:
            pipeline.stop()
        print("🛑 Worker30 stopped.")


@app.route('/image.jpg')
def get_image():
    """
    Returns the most recently fetched RGB image as a JPEG.
    """
    with image_lock:
        if latest_image is None:
            # No image has been captured yet
            return Response(status=503, response="No image available")

        # Encode as JPEG
        success, jpeg = cv2.imencode('.jpg', latest_image)
        if not success:
            return Response(status=500, response="Failed to encode image")

        return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/depth.jpg')
def get_depth():
    """
    Returns the most recently fetched depth image as a JPEG.
    """
    with depth_lock:
        if latest_depth is None:
            # No depth image has been captured yet
            return Response(status=503, response="No depth image available")

        # Encode as JPEG

        success, jpeg = cv2.imencode('.jpg', latest_depth)
        if not success:
            return Response(status=500, response="Failed to encode depth image")

        return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/')
def get_worker30():
    return open('visualization.html').read()


if __name__ == "__main__":
    # Start the camera loop in a daemon thread
    cam_thread = threading.Thread(target=main_loop, daemon=True)
    cam_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
