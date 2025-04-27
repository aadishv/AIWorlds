import numpy as np
from constants import MEASUREMENT_ROW
import json
import copy
# oliver do your ~~thing~~ localization :P


class Localization:
    def __init__(self, initial_pose):
        self.pose = initial_pose  # (x, y, θ)

    def update(self, measurement, imu):
        # MARK: - THIS PART OF THE CODE WRITTEN BY OLIVER

        # imu is the imu heading in centidegrees (from 0 to 360*100)
        # measurement is a numpy array of shape (640,) representing a row of depth measurements in meters
        # use imu and measurement to update self.pose
        # this function needs to run in less than 1/30 of a second
        pass

# a post-processing class for detections and ...other stuff..., designed to be used with an EventDrivenSerial-like interface


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
    def update(self, theta):  # update with new theta
        depth_image = self.app.camera.frames[1]
        # process detections
        detections = copy.deepcopy(self.app.inference.raw_detections)
        for detection in detections:
            # 480, 640
            detection['y'] *= 480 / 640
            depth_value = self.get_depth(
                detection, depth_image)
            if np.isnan(depth_value):
                detection['depth'] = -1
            else:
                detection['depth'] = depth_value
        self.detections = [
            i for i in detections if
            i['class'] not in ['red', 'blue'] or
            i['confidence'] > 0.6
        ]
        # run localization
        measurement = self.app.camera.frames[1][MEASUREMENT_ROW]
        self.localization.update(measurement, theta)
        # do collision detection
        flag = ""
        if (measurement[measurement > 0].shape[0] > 0):
            percentile = np.percentile(
                measurement[measurement > 0], 10) * self.depth_scale
            if percentile * 3.28 < 1:  # that is, at least 20% of nonzero readings are less than 1 foot away
                print("collision!!!!")
                flag = "STOP"
        # TODO: pose conversion

        # return all results
        output = {
            'pose': {
                'x': self.localization.pose[0],
                'y': self.localization.pose[1],
                'theta': self.localization.pose[2]
            },
            'stuff': self.detections,
            'flag': flag
        }
        return output
        # json.dumps(output, indent=None, separators=(',', ':'))
