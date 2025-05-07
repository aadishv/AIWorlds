from camera import CameraWorker
from inference import InferenceWorker
from post import Processing
import threading
import time
import signal
import logging
from flask import Flask, Response, stream_with_context, redirect, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import json
import atexit
import serial
import sys
import os


class App:
    def __init__(self):
        self.camera = CameraWorker()
        self.inference = InferenceWorker(self)
        self.most_recent_result = {}

    def service_simulator(self):
        self.post = Processing(self, (0, 0, 0), self.camera._processing.fl)
        while True:
            start = time.time()
            self.most_recent_result = self.post.update((0, 0, 10))
            time.sleep(max(0, 1.0 / 30.0 - (time.time() - start)))

    def run_dashboard(self):
        # stop excessive logging of requests
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        # Suppress the reloader's startup message
        os.environ['WERKZEUG_RUN_MAIN'] = 'true'
        # os.environ['FLASK_ENV'] = 'development'   # This setting can sometimes cause issues, better to control debug=False below
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *args, **kwargs: None  # Suppress server banner

        app = Flask("3151App")
        CORS(app, resources={r"/*": {"origins": "*"}})

        def generate_frames(type):
            """Generator function to capture frames and yield them for streaming."""
            while True:
                # Access the latest frames from the CameraWorker instance
                # Ensure self.camera.frames is accessed thread-safely if needed,
                # but for this simple case, reading should be mostly fine.
                color_img, depth_img = self.camera.frames

                if color_img is None or depth_img is None:
                    print(
                        f"Warning: generate_frames received None for {type} image.")
                    time.sleep(0.1)  # Wait a bit before trying again
                    continue

                frame_to_stream = None
                if type == 'color':
                    frame_to_stream = color_img.copy()  # Get a copy to avoid modifying the original
                elif type == 'depth':
                    # Convert depth to an 8-bit colormap for visualization
                    # Check for max value before normalization to avoid division by zero if image is all 0s
                    max_depth = depth_img.max()
                    if max_depth > 0:
                        depth_8u = cv2.normalize(
                            depth_img, None,
                            alpha=255, beta=0,  # Invert min/max to make closer objects brighter if desired, or keep alpha=0, beta=255 for far=bright
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U
                        )
                        # Optional: Apply mask for 0 depth values if you want them black
                        # depth_mask = (depth_img > 0).astype(np.uint8) * 255
                        # depth_8u = cv2.bitwise_and(depth_8u, depth_mask)

                        frame_to_stream = cv2.applyColorMap(
                            depth_8u, cv2.COLORMAP_JET)
                    else:
                        # Handle case where depth image is all zeros
                        # Return a black image same size as color
                        frame_to_stream = np.zeros_like(color_img)

                if frame_to_stream is None:
                    # Default to black if type is unknown
                    frame_to_stream = np.zeros_like(color_img)

                # Encode the frame as JPEG
                # Resize for dashboard display
                frame_to_stream = cv2.resize(frame_to_stream, (320, 240))
                ret, buffer = cv2.imencode('.jpg', frame_to_stream)
                if not ret:
                    print(f"Failed to encode frame for type {type}")
                    time.sleep(0.1)  # Wait a bit before trying again
                    continue  # Skip if encoding failed

                # Convert the buffer to bytes
                frame_bytes = buffer.tobytes()

                # Yield the frame in the multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Control frame rate (e.g., limit to 15 FPS)
                # Using a small sleep to prevent dashboard from overwhelming the system
                # based on desired stream FPS, not camera FPS
                time.sleep(1/15)  # Stream at max 15 FPS

        @app.route('/<type>.mjpg')
        def video_feed(type):
            """Video streaming route. Access this in your browser."""
            if type not in ['color', 'depth']:
                return "Invalid stream type", 404  # Handle invalid types

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
                    # Access the most recent result
                    # Use app.most_recent_result to access instance variable within generator
                    data = self.most_recent_result
                    # Ensure data is serializable (handle potential numpy types etc if necessary)
                    # Simple dump works for basic dict. For complex objects, need a custom encoder.
                    payload = f"data: {json.dumps(data)}\n\n"
                    yield payload
                    time.sleep(1/30)  # Update events at ~30 Hz

            headers = {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'  # Recommended for SSE to prevent buffering
            }
            return Response(stream_with_context(generate()), headers=headers)

        # --- New endpoint to update HSV configuration ---

        @app.route('/update_hsv', methods=['POST'])
        def update_hsv_config():
            """
            Receives JSON payload with HSV values and writes to config file.
            Expected JSON format: {"HUE": float, "SATURATION": float, "VALUE": float}
            """
            try:
                config_data = request.get_json()  # Get JSON data from the request body

                # Basic validation
                if config_data is None:
                    return jsonify({"status": "error", "message": "Invalid JSON received"}), 400

                # Check for required keys and attempt conversion to float
                required_keys = ["HUE", "SATURATION", "VALUE"]
                config_to_write = {}
                try:
                    for key in required_keys:
                        if key not in config_data:
                            return jsonify({"status": "error", "message": f"Missing key: '{key}'"}), 400
                        config_to_write[key] = float(
                            config_data[key])  # Ensure float type

                except ValueError:
                    return jsonify({"status": "error", "message": "Values for HUE, SATURATION, and VALUE must be numbers"}), 400
                except TypeError:
                    return jsonify({"status": "error", "message": "Invalid data type for HUE, SATURATION, or VALUE"}), 400

                # Get the JSON file path from the camera worker instance
                json_file_path = self.camera.json_path  # Access the attribute

                # Write the updated config to the file
                try:
                    # Use 'w' mode to overwrite the file
                    with open(json_file_path, 'w') as f:
                        # Use indent for readability
                        json.dump(config_to_write, f, indent=4)

                    # Return success response
                    return jsonify({"status": "success", "message": "HSV config updated", "data": config_to_write}), 200

                except IOError as e:
                    print(
                        f"Error writing to HSV config file {json_file_path}: {e}")
                    # Return internal server error if file writing fails
                    return jsonify({"status": "error", "message": f"Failed to write config file: {e}"}), 500

            except Exception as e:
                # Catch any other unexpected errors during request processing
                print(
                    f"An unexpected error occurred in /update_hsv endpoint: {e}")
                return jsonify({"status": "error", "message": f"An internal error occurred: {e}"}), 500
        # --- End of new endpoint ---

        # A simple landing page to consume them

        @app.route('/')
        def index():
            # Redirect to your dashboard URL - adjust if your dashboard is elsewhere
            return redirect('http://localhost:4321/tools/vairc')

        # Run Flask
        # Use debug=False in production/deployment
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

    def service_serial(self):
        port = "/dev/ttyACM1"
        baud = 115200
        ser = serial.Serial(port, baud, timeout=None)
        post = None
        first = True
        while True:
            line = ser.readline().decode("utf-8", "replace").strip()
            line = "".join(c for c in line if ord(c) < 128 and ord(c) > 0)
            try:
                data = json.loads(line)
            except Exception as e:
                print("error with", line, e)
                continue

            if first:
                print(data)
                post = Processing(
                    self, (data['x'], data['y'], data['theta']), self.camera._processing.fl)
                first = False
                # Optionally respond with an “ack” or immediately run an update
                continue

            # Subsequent messages contain only theta (or {"theta":…})
            x = data.get("x", 0)
            y = data.get("y", 0)
            theta = data.get("theta", 0)
            # returns the full {pose,stuff,flag,jetson}
            self.most_recent_result = post.update((x, y, theta))
            print(self.most_recent_result)
            v5_json = post.convert_to_v5(self.most_recent_result)
            ser.write((v5_json + "\n").encode("utf-8"))
            time.sleep(1.0 / 30.0)

    def close(self):
        self.camera.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # get first unnamed argument
    mode = 'sim'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ser':
            mode = 'ser'
    with App() as app:
        try:
            threads = [
                threading.Thread(target=app.camera.worker,
                                 name="camera_worker", daemon=False),
                threading.Thread(
                    target=app.inference.worker, name="inference_worker", daemon=False),
                (
                    threading.Thread(
                        target=app.service_simulator, name="service_simulator", daemon=False)
                    if mode == 'sim' else
                    threading.Thread(
                        target=app.service_serial, name="service_serial", daemon=False)
                ),
                threading.Thread(target=app.run_dashboard,
                                 name="dashboard_worker", daemon=False)
            ]
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
