from camera import CameraWorker
from inference import InferenceWorker
from post import Processing
import threading
import time
import signal
import json
import atexit
import serial
import sys
from dashboard_server import DashboardServer


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

    def service_serial(self):
        port = "/dev/ttyACM1"
        baud = 115200
        ser = serial.Serial(port, baud, timeout=None)
        post = None
        first = True
        while True:
            try:
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
            except serial.SerialException as e:
                print("Encountered error in serial loop:", e)
                time.sleep(1.0)

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
                threading.Thread(target=lambda: DashboardServer(app).run(),
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
