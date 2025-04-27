try:
    import pyrealsense2 as rs
    _ = rs.pipeline()
except AttributeError:
    import pyrealsense2.pyrealsense2 as rs
    _ = rs.pipeline()
except Exception:
    print("Error importing pyrealsense2")
    exit(1)
import cv2
import numpy as np

class Camera:
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
        return self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()  # Stop the pipeline when finished

class Processing:
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

        return color_image, depth_image

# lightweight wrapper class to suit the Worker
class CameraWorker:
    def __init__(self):
        self._camera = Camera()
        self._camera.start()
        self._processing = Processing(self._camera.depth_scale)
        # color, depth
        self.frames = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8)
        )

    def worker(self):
        try:
            while True:
                frames = self._camera.get_frames()
                self.frames = self._processing.process_frames(frames)
        finally:
            self._camera.stop()
    # snippet for getting color map from image:
    # depth_map = cv2.applyColorMap(depthImage, cv2.COLORMAP_JET)
    def close(self):
        self._camera.stop()
