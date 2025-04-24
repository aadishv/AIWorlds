
import pyrealsense2 as rs
import cv2
import numpy as np

# from highstakes.py, JetsonExample


class Camera:
    # Class handles Camera object instantiation and data requests.
    def __init__(self):
        self.pipeline = rs.pipeline()  # Initialize RealSense pipeline
        self.config = rs.config()
        # Enable depth stream at 640x480 in z16 encoding at 30fps
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # Enable color stream at 640x480 in rgb8 encoding at 30fps
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def start(self):
        self.profile = self.pipeline.start(self.config)  # Start the pipeline
        # Obtain depth sensor and calculate depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frames(self):
        # Wait and fetch frames from the pipeline
        return self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()  # Stop the pipeline when finished


class Processing:
    # Class to handle camera data processing, preparing for inference, and running inference on camera image.
    def __init__(self, depth_scale):
        self.depth_scale = depth_scale
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.HUE = 1.2
        self.SATURATION = 1.2
        self.VALUE = 0.8

    # checked
    def process_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = hsv[..., 0] + self.HUE
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.SATURATION, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.VALUE, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # checked
    def updateHSV(self, newHSV):
        self.HUE = newHSV.h

        if (self.SATURATION >= 0):
            self.SATURATION = 1 + (newHSV.s) / 100
        else:
            self.SATURATION = (100 - abs(newHSV.s)) / 100

        if (self.VALUE >= 0):
            self.VALUE = 1 + (newHSV.v) / 100
        else:
            self.VALUE = (100 - abs(newHSV.v)) / 100

    # checked
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

    # checked
    def align_frames(self, frames):
        # Align depth frames to color frames
        aligned_frames = self.align.process(frames)
        # Get the aligned frames and validate them
        self.depth_frame_aligned = aligned_frames.get_depth_frame()
        self.color_frame_aligned = aligned_frames.get_color_frame()

        if not self.depth_frame_aligned or not self.color_frame_aligned:
            self.depth_frame_aligned = None
            self.color_frame_aligned = None

    # checked
    def process_frames(self, frames):
        # Align frames and extract color and depth images
        # Apply a color map to the depth image
        self.align_frames(frames)
        depth_image = np.asanyarray(self.depth_frame_aligned.get_data())
        # feed-forward factor, since it is always 3% off
        depth_image = depth_image / 1.03
        color_image = np.asanyarray(self.color_frame_aligned.get_data())
        # apply color correction to image
        color_image = self.process_image(color_image)
        depthImage = cv2.normalize(
            depth_image, None, alpha=0.01, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depthImage, cv2.COLORMAP_JET)

        return depth_image, color_image, depth_map
