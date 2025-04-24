import numpy as np

class Localization:
    def __init__(self, initial_pose):
        self.pose = initial_pose # (x, y, θ)
    def update(self, measurement, imu):
        # MARK: - THIS PART OF THE CODE WRITTEN BY OLIVER

        # imu is the imu heading in centidegrees (from 0 to 360*100)
        # measurement is a numpy array of shape (640,) representing a row of depth measurements in meters
        # use imu and measurement to update self.pose
        # this function needs to run in less than 1/30 of a second
        pass
