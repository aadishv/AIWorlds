import numpy as np
import os
import copy
from constants import MEASUREMENT_ROW
import random
from poseconv import locate_detection

# def locate_detection(robot_x, robot_y, robot_theta_deg, detection):

class Localization:
    def __init__(self, initial_pose):
        self.pose = initial_pose  # (x, y, θ)

    def update(self, measurement, imu):
        # add 0.01 to each part
        ny = self.pose[1] + 1
        self.pose = (
            self.pose[0],
            -3 * 24 if ny > 3 * 24 else ny,
            self.pose[2],
        )

class Processing:
    def __init__(self, app, pose):  # pose = (x, y, theta)
        self.app = app
        self.localization = Localization(pose)
        self.detections = []
        self.depth_scale = app.camera._camera.depth_scale

    def get_depth(self, detection, depth_img):
        try:
            height = detection["height"]
            width = detection["width"]
            top = max(detection["y"] - height / 2, 0)
            bottom = min(detection["y"] + height / 2, depth_img.shape[0])
            left = max(detection["x"] - width / 2, 0)
            right = min(detection["x"] + width / 2, depth_img.shape[1])
            depth_img = depth_img[int(top):int(
                bottom), int(left):int(right)].astype(float).flatten()
            p = np.percentile(depth_img[depth_img > 0], 10) * self.depth_scale
            return p
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
        measurement = self.app.camera.frames[1][MEASUREMENT_ROW]
        self.localization.update(measurement, theta)

        # collision detection
        flag = ""
        if measurement[measurement > 0].size > 0:
            percentile = np.percentile(
                measurement[measurement > 0], 10) * self.depth_scale
            if percentile * 3.28 < 1:
                print("collision!!!!")
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
        return output
