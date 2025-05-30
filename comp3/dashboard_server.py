import logging
import os
import sys
import time
import json
import numpy as np
import cv2
from flask import Flask, Response, stream_with_context, redirect, request, jsonify
from flask_cors import CORS


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


class DashboardServer:
    def __init__(self, app_instance):
        self.app_instance = app_instance
        self.flask_app = Flask("3151App")
        CORS(self.flask_app, resources={r"/*": {"origins": "*"}})
        self._setup_routes()

    def _generate_frames(self, type):
        while True:
            color_img, depth_img = self.app_instance.camera.frames

            if color_img is None or depth_img is None:
                print(
                    f"Warning: generate_frames received None for {type} image.")
                time.sleep(0.1)
                continue

            frame_to_stream = None
            if type == 'color':
                frame_to_stream = color_img.copy()
            elif type == 'depth':
                max_depth = depth_img.max()
                if max_depth > 0:
                    depth_8u = cv2.normalize(
                        depth_img, None,
                        alpha=255, beta=0,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U
                    )
                    frame_to_stream = cv2.applyColorMap(
                        depth_8u, cv2.COLORMAP_JET)
                else:
                    frame_to_stream = np.zeros_like(color_img)

            if frame_to_stream is None:
                frame_to_stream = np.zeros_like(color_img)

            frame_to_stream = cv2.resize(frame_to_stream, (320, 240))
            ret, buffer = cv2.imencode('.jpg', frame_to_stream)
            if not ret:
                print(f"Failed to encode frame for type {type}")
                time.sleep(0.1)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1/15)

    def _update_hsv_config(self):
        try:
            config_data = request.get_json()
            if config_data is None:
                return jsonify({"status": "error", "message": "Invalid JSON received"}), 400
            required_keys = ["HUE", "SATURATION", "VALUE"]
            config_to_write = {}
            try:
                for key in required_keys:
                    if key not in config_data:
                        return jsonify({"status": "error", "message": f"Missing key: '{key}'"}), 400
                    config_to_write[key] = float(config_data[key])
            except ValueError:
                return jsonify({"status": "error", "message": "Values for HUE, SATURATION, and VALUE must be numbers"}), 400
            except TypeError:
                return jsonify({"status": "error", "message": "Invalid data type for HUE, SATURATION, or VALUE"}), 400
            json_file_path = self.app_instance.camera.json_path
            try:
                with open(json_file_path, 'w') as f:
                    json.dump(config_to_write, f, indent=4)
                return jsonify({"status": "success", "message": "HSV config updated", "data": config_to_write}), 200
            except IOError as e:
                print(

                    f"Error writing to HSV config file {json_file_path}: {e}")
                return jsonify({"status": "error", "message": f"Failed to write config file: {e}"}), 500
        except Exception as e:
            print(

                f"An unexpected error occurred in /update_hsv endpoint: {e}")
            return jsonify({"status": "error", "message": f"An internal error occurred: {e}"}), 500

    def _setup_routes(self):
        app = self.flask_app
        app_instance = self.app_instance

        def _sse_generate():
            while True:
                data = app_instance.most_recent_result
                payload = f"data: {json.dumps(data)}\n\n"
                yield payload
                time.sleep(1/30)



        def _index():
            return redirect('http://localhost:4321/tools/vairc')

        @app.route('/<type>.mjpg')
        def video_feed(type):
            if type not in ['color', 'depth']:
                return "Invalid stream type", 404
            return Response(self._generate_frames(type),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/events')
        def sse_events():
            headers = {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
            return Response(stream_with_context(_sse_generate()), headers=headers)

        @app.route('/update_hsv', methods=['POST'])
        def update_hsv_config():
            return self._update_hsv_config()

        def run_events(include_old):
            yield "\n" # for the page to load
            last_len = 0 if include_old else len(self.app_instance.v5_logs)
            while True:
                for stuff in self.app_instance.v5_logs[last_len:]:
                    yield stuff + "\n"
                last_len = len(self.app_instance.v5_logs)
                time.sleep(1.0 / 30.0)

        @app.route('/terminal')
        def v5_logger():
            headers = {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
            return Response(stream_with_context(run_events(True)), headers=headers)
        # does not include old stuff
        @app.route('/terminal2')
        def v5_logger_no_include():
            headers = {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
            return Response(stream_with_context(run_events(False)), headers=headers)

        @app.route('/')
        def index():
            return _index()

    def run(self):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        os.environ['WERKZEUG_RUN_MAIN'] = 'true'
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *args, **kwargs: None
        self.flask_app.run(host='0.0.0.0', port=5000,
                           threaded=True, debug=False)
