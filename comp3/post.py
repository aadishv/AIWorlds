import numpy as np
import os
import copy
from constants import MEASUREMENT_ROW
import random
from poseconv import locate_detection
from test import OliverLocalization
# def locate_detection(robot_x, robot_y, robot_theta_deg, detection):
import json

class Localization:
    def __init__(self, initial_pose):
        self.pose = (0, 0, np.pi / 2)  # initial_pose  # (x, y, θ)
        self.t = 0

    def update(self, measurement, imu):
        # Radius of the circle
        radius = 48

        # Calculate angular velocity to complete a circle in 300 steps
        angular_velocity = 2 * np.pi / 300

        # Update the angle
        theta = (self.t % 300 * angular_velocity)

        # Calculate new x and y coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Update the pose
        self.pose = (x, y, theta)
        self.t += 1


'''
example:
```py
localization = OliverLocalization(10, (0, 60, 0))

localization.update(depths, theta)
print(localization.pose)
```
'''


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
            return np.percentile(depth_img[depth_img > 0], 10)
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
        print(result)
        return result

    def update(self, theta):  # update with new theta
        depth_image = self.app.camera.frames[1]
        # process detections
        detections = copy.deepcopy(self.app.inference.raw_detections)
        for detection in detections:
            detection['y'] *= 480 / 640
            depth_value = self.get_depth(detection, depth_image)
            detection['depth'] = -1 if np.isnan(depth_value) else depth_value
        self.detections = [
            i for i in detections
            if i['class'] not in ['red', 'blue'] or i['confidence'] > 0.6
        ]
        for det in self.detections:
            result = locate_detection(*self.localization.pose, det)
            det['fx'], det['fy'], det['fz'] = result['x'], result['y'], result['z']
        # run localization
        measurement = self.app.camera.frames[1][MEASUREMENT_ROW] * 3.28 * 12
        if isinstance(self.localization, FakeDetection):
            self.localization = OliverLocalization(
                self.fl, (-(72-np.percentile(measurement, 50)), 0, 90))
        self.localization.update(measurement, 90)
        # collision detection
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
        _ = self.convert_to_v5(output)
        return output
