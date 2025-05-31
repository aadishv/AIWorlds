import numpy as np
import copy
from constants import MEASUREMENT_ROW, ENGINE_PATH
from poseconv import locate_detection
import json
import datetime, time, os, cv2

class FakeDetection:  # use w/ duck-typing
    def __init__(self):
        self.pose = (0, 0, 0)


class Processing:
    def __init__(self, app, pose, focal_length):  # pose = (x, y, theta)
        self.app = app
        self.localization = FakeDetection()
        self.detections = []
        self.depth_scale = app.camera._camera.depth_scale
        self.fl = focal_length
        self.t = 0
        self.recording_path = None

    def record(self, object):
        if self.recording_path is None:
            self.recording_path = f"/home/vex/AIWorlds/recordings/{datetime.datetime.now().strftime('%m.%d::%H:%M:%S')}"
            self.color_path = f"{self.recording_path}/color"
            self.depth_path = f"{self.recording_path}/depth"
            self.log_path = f"{self.recording_path}/log"
            os.makedirs(self.recording_path, exist_ok=True)
            os.makedirs(self.color_path, exist_ok=True)
            os.makedirs(self.depth_path, exist_ok=True)
            os.makedirs(self.log_path, exist_ok=True)
        t = time.time()

        color_img, depth_img = self.app.camera.frames
        cv2.imwrite(f"{self.color_path}/{t}.jpg", color_img)
        max_depth = depth_img.max()
        if max_depth > 0:
            depth_8u = cv2.normalize(
                depth_img, None,
                alpha=255, beta=0,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
            depth_img = cv2.applyColorMap(
                depth_8u, cv2.COLORMAP_JET)
        else:
            depth_img = np.zeros_like(color_img)
        cv2.imwrite(f"{self.depth_path}/{t}.jpg", depth_img)
        with open(f"{self.log_path}/{t}.json", "w") as f:
            json.dump(object, f)

    def get_depth(self, detection, depth_img):
        try:
            height = detection["height"]
            width = detection["width"]
            top = max(detection["y"] - height / 2, 0)
            bottom = min(detection["y"] + height / 2, depth_img.shape[0])
            left = max(detection["x"] - width / 2, 0)
            right = min(detection["x"] + width / 2, depth_img.shape[1])
            depth_img = depth_img[int(top):int(
                bottom), int(left):int(right)].astype(float).flatten()  # * self.depth_scale
            return np.percentile(depth_img[depth_img > 0], 10) + (6.5/39.37) # add offset because skibid
        except Exception:
            return -1

    def get_jetson_info(self):
        info = {"cpu_temp": None, "gpu_temp": None, "uptime": None}

        # CPU temperature: typically thermal_zone0
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                # file is in millidegree Celsius
                t = int(f.read().strip())
                info["cpu_temp"] = t / 1000.0
        except Exception:
            pass

        # GPU temperature: often thermal_zone1
        try:
            with open("/sys/class/thermal/thermal_zone1/temp", "r") as f:
                t = int(f.read().strip())
                info["gpu_temp"] = t / 1000.0
        except Exception:
            pass

        # Uptime in seconds (first field of /proc/uptime)
        try:
            with open("/proc/uptime", "r") as f:
                uptime_secs = float(f.read().split()[0])
                info["uptime"] = uptime_secs
        except Exception:
            pass

        return info

    def convert_to_v5(self, data):
        result = copy.deepcopy(data)
        # Create a new object from scratch instead of modifying the existing one
        new_result = {
            'pose': {
                'x': result['pose']['x'],
                'y': result['pose']['y'],
                'theta': -result['pose']['theta']
            },
            'stuff': [],
            'flag': result['flag']
        }

        # Copy and transform each detection
        for det in result['stuff']:
            new_det = {
                'x': det['fx'],
                'y': det['fy'],
                'z': det['fz'],
                'class': det['class']
            }
            new_result['stuff'].append(new_det)

        result = json.dumps(new_result, separators=(",", ":"))
        return result

    def update(self, pose):  # update with new theta
        self.t += 1
        depth_image = self.app.camera.frames[1]
        # process detections
        detections = copy.deepcopy(self.app.inference.raw_detections)
        for detection in detections:
            detection['y'] *= 480 / 640
            if '0622' in ENGINE_PATH:  # a little trick
                if detection['class'] == 'goal':
                    detection['height'] *= 0.8
                    detection['width'] *= 0.8
            depth_value = self.get_depth(detection, depth_image)
            detection['depth'] = -1 if np.isnan(depth_value) else depth_value
        self.detections = [
            i for i in detections
            # if i['class'] not in ['red', 'blue'] or  ...
            if i['confidence'] > 0.7
        ]

        for det in self.detections:
            # Correctly extract pose components and convert theta to radians
            x, y, theta_deg = self.localization.pose
            theta_rad = np.deg2rad(theta_deg)

            # Pass individual arguments to locate_detection
            result = locate_detection(x, y, theta_rad, det)
            det['fx'], det['fy'], det['fz'] = result['x'], result['y'], result['z']
        # run localization
        measurement = self.app.camera.frames[1][MEASUREMENT_ROW] * 3.28 * 12
        self.localization.pose = pose

        flag = ""
        if measurement[measurement > 0].size > 0:
            percentile = np.percentile(
                measurement[measurement > 0], 10)
            if percentile < 12:
                flag = "STOP"

        # build output
        output = {
            'pose': {
                'x': self.localization.pose[0],
                'y': self.localization.pose[1],
                'theta': self.localization.pose[2],
            },
            'stuff': self.detections,
            'flag': flag,
            'jetson': self.get_jetson_info()
        }

        self.record(output)
        return output
