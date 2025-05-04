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
    def __init__(self, depth_scale, profile):
        """
        depth_scale : the meter‐per‐unit scale from depth_sensor.get_depth_scale()
        profile     : rs.pipeline_profile returned by pipeline.start()
        """
        self.depth_scale = depth_scale

        # grab the two streams' intrinsics & extrinsics
        ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        din = ds.get_intrinsics()      # depth camera intrinsics
        cin = cs.get_intrinsics()      # color camera intrinsics
        self.fl = cin.fx
        ext = ds.get_extrinsics_to(cs)  # depth → color

        w, h = cin.width, cin.height

        # build a LUT so that (u_color,v_color) → (u_depth,v_depth)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for v in range(h):
            for u in range(w):
                # back‑project this color‑pixel at unit depth into 3D color coords
                pt_c = rs.rs2_deproject_pixel_to_point(cin, [u, v], 1.0)
                # transform into depth camera coords
                Xd = ext.rotation[0]*pt_c[0] + ext.rotation[1] * \
                    pt_c[1] + ext.rotation[2]*pt_c[2] + ext.translation[0]
                Yd = ext.rotation[3]*pt_c[0] + ext.rotation[4] * \
                    pt_c[1] + ext.rotation[5]*pt_c[2] + ext.translation[1]
                Zd = ext.rotation[6]*pt_c[0] + ext.rotation[7] * \
                    pt_c[1] + ext.rotation[8]*pt_c[2] + ext.translation[2]
                # project back to depth image pixel
                px = rs.rs2_project_point_to_pixel(din, [Xd, Yd, Zd])
                map_x[v, u] = px[0]
                map_y[v, u] = px[1]

        self.map_x = map_x
        self.map_y = map_y

        # your HSV gains
        self.HUE = 1.2
        self.SATURATION = 1.2
        self.VALUE = 0.8

    def process_image(self, image):
        """Apply the same HSV‐based tweak as before."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = hsv[..., 0] + self.HUE
        hsv[..., 1] = np.clip(hsv[..., 1] * self.SATURATION, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self.VALUE,      0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_frames(self, frames):
        # pull raw frames
        dframe = frames.get_depth_frame()
        cframe = frames.get_color_frame()
        if not dframe or not cframe:
            return None, None

        # raw arrays
        depth_raw = np.asanyarray(dframe.get_data()).astype(np.float32)
        color = np.asanyarray(cframe.get_data())

        # convert to meters and apply your 3% feed‑forward
        depth_m = depth_raw * self.depth_scale / 1.03

        # fast multi‑threaded remap to color coords
        depth_aligned = cv2.remap(
            depth_m, self.map_x, self.map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )

        # now depth_aligned[v,u] is the distance in meters
        # (pixels with no depth become 0.0; you can mask or set to NaN if you prefer)

        # color‐correct the BGR image
        color_corr = self.process_image(color)

        return color_corr, depth_aligned

# lightweight wrapper class to suit the Worker


class CameraWorker:
    def __init__(self):
        self._camera = Camera()
        self._camera.start()
        self._processing = Processing(
            self._camera.depth_scale, self._camera.profile)
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

    def close(self):
        self._camera.stop()
