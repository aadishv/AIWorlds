try:
    import pyrealsense2 as rs
    # Try using the standard import and a simple test
    _ = rs.pipeline()
    # Also test rs.align which can sometimes cause issues
    _ = rs.align(rs.stream.color)
except AttributeError:
    # Fallback for some specific installations
    try:
        import pyrealsense2.pyrealsense2 as rs
        _ = rs.pipeline()
        _ = rs.align(rs.stream.color)
    except Exception as e:
        print(f"Error importing pyrealsense2: {e}")
        print("Please ensure pyrealsense2 is correctly installed.")
        exit(1)
except Exception as e:
    print(
        f"Error importing pyrealsense2 or initializing basic components: {e}")
    print("Please ensure pyrealsense2 is correctly installed and a camera is connected.")
    # Decide if you want to exit or handle gracefully, exiting for now
    exit(1)

import cv2
import numpy as np
import json
import os  # Import os to check file existence


# --- Existing Camera and Processing Classes (unchanged) ---

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
        # Start the pipeline, allowing configuration to resolve the devices and streams
        try:
            self.profile = self.pipeline.start(self.config)
            # Obtain depth sensor and calculate depth scale
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            return True
        except Exception as e:
            print(f"Failed to start camera pipeline: {e}")
            return False

    def get_frames(self):
        # Wait for a coherent set of frames: depth and color
        try:
            frames = self.pipeline.wait_for_frames()
            # Optional: Align frames (if you weren't using the remap method)
            # align = rs.align(rs.stream.color)
            # aligned_frames = align.process(frames)
            return frames  # Or aligned_frames if you switch to align
        except Exception as e:
            print(f"Error waiting for frames: {e}")
            return None

    def stop(self):
        # Stop the pipeline when finished
        try:
            self.pipeline.stop()
            print("Camera pipeline stopped.")
        except Exception as e:
            print(f"Error stopping pipeline: {e}")


class Processing:
    def __init__(self, depth_scale, profile):
        """
        depth_scale : the meter‐per‐unit scale from depth_sensor.get_depth_scale()
        profile     : rs.pipeline_profile returned by pipeline.start()
        """
        self.depth_scale = depth_scale

        # grab the two streams' intrinsics & extrinsics
        # Ensure we get the video stream profiles
        try:
            ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
            din = ds.get_intrinsics()      # depth camera intrinsics
            cin = cs.get_intrinsics()      # color camera intrinsics
            self.fl = cin.fx
            ext = ds.get_extrinsics_to(cs)  # depth → color
        except Exception as e:
            print(f"Error getting stream profiles/intrinsics/extrinsics: {e}")
            # Handle error - maybe raise or return False from init if allowed
            raise e  # Re-raise the exception to indicate failure

        w, h = cin.width, cin.height

        # build a LUT so that (u_color,v_color) → (u_depth,v_depth)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for v in range(h):
            for u in range(w):
                # back‑project this color‑pixel at unit depth into 3D color coords
                # rs2_deproject_pixel_to_point returns a float[3]
                pt_c = rs.rs2_deproject_pixel_to_point(
                    cin, [float(u), float(v)], 1.0)
                # transform into depth camera coords (using matrix multiplication concept)
                # Extrinsics rotation is a 3x3 matrix stored in a flat list of 9 elements
                Xd = ext.rotation[0]*pt_c[0] + ext.rotation[1] * \
                    pt_c[1] + ext.rotation[2]*pt_c[2] + ext.translation[0]
                Yd = ext.rotation[3]*pt_c[0] + ext.rotation[4] * \
                    pt_c[1] + ext.rotation[5]*pt_c[2] + ext.translation[1]
                Zd = ext.rotation[6]*pt_c[0] + ext.rotation[7] * \
                    pt_c[1] + ext.rotation[8]*pt_c[2] + ext.translation[2]
                # project back to depth image pixel
                # rs2_project_point_to_pixel returns a float[2]
                px = rs.rs2_project_point_to_pixel(din, [Xd, Yd, Zd])
                map_x[v, u] = px[0]
                map_y[v, u] = px[1]

        self.map_x = map_x
        self.map_y = map_y

        # --- HSV gains - Will be updated from JSON ---
        self.HUE = 0.0  # Initialize to neutral, will be set by JSON read
        self.SATURATION = 1.0  # Initialize to neutral
        self.VALUE = 1.0  # Initialize to neutral

    def process_image(self, image):
        """Apply the same HSV‐based tweak as before using current self.HUE, etc."""
        # Ensure image is valid before processing
        if image is None or image.size == 0:
            print("Warning: process_image received empty image.")
            return image

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Apply gains. Hue wraps around 180 in OpenCV's 8-bit representation.
            # Add HUE and then take modulo 180
            hsv[..., 0] = (hsv[..., 0].astype(np.float32) + self.HUE) % 180
            hsv[..., 1] = np.clip(hsv[..., 1].astype(
                np.float32) * self.SATURATION, 0, 255).astype(np.uint8)
            hsv[..., 2] = np.clip(hsv[..., 2].astype(
                np.float32) * self.VALUE,      0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            print(f"Error in process_image: {e}")
            return image  # Return original image on error

    def process_frames(self, frames):
        # pull raw frames
        dframe = frames.get_depth_frame()
        cframe = frames.get_color_frame()
        if not dframe or not cframe:
            print("Warning: process_frames received incomplete frames.")
            return None, None

        try:
            # raw arrays
            # Use np.copy() to ensure we have our own copy if needed, although asanyarray often copies
            depth_raw = np.asanyarray(dframe.get_data()).astype(np.float32)
            color = np.asanyarray(cframe.get_data())

            # convert to meters and apply your 3% feed‑forward (division by 1.03)
            depth_m = depth_raw * self.depth_scale / 1.03

            # fast multi‑threaded remap to color coords
            # Ensure map_x and map_y are correctly sized for the input depth image if needed,
            # but typically remap handles target size based on the maps.
            # The maps were built based on color image size (640x480), remapping depth to color view.
            depth_aligned = cv2.remap(
                depth_m, self.map_x, self.map_y,
                interpolation=cv2.INTER_NEAREST,  # INTER_LINEAR might also be an option
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0  # Pixels outside the depth image become 0.0
            )

            # now depth_aligned[v,u] is the distance in meters
            # (pixels with no depth become 0.0; you can mask or set to NaN if you prefer)

            # color‐correct the BGR image using the current HSV parameters
            color_corr = self.process_image(color)

            return color_corr, depth_aligned

        except Exception as e:
            print(f"Error in process_frames: {e}")
            return None, None


# lightweight wrapper class to suit the Worker
class CameraWorker:
    def __init__(self, json_path="hsv_config.json", update_interval=5):
        self.json_path = json_path
        self.update_interval = update_interval
        self._frame_counter = 0
        self._camera = Camera()

        # Start camera and check if successful before proceeding
        if not self._camera.start():
            # Handle the failure, e.g., raise an exception or exit
            # For this example, we'll raise an error to stop execution if camera fails
            raise RuntimeError("Failed to start camera pipeline.")

        self._processing = Processing(
            self._camera.depth_scale, self._camera.profile)

        # Set initial HSV parameters from the JSON file immediately
        self._update_hsv_parameters_from_json()

        # color, depth - Initialize with correct shapes and types
        # Assuming 640x480 based on config, uint8 for color, float32 for depth (meters)
        self.frames = (
            np.zeros((480, 640, 3), dtype=np.uint8),  # BGR color
            np.zeros((480, 640), dtype=np.float32)   # Depth in meters
        )

    def _update_hsv_parameters_from_json(self):
        """Reads HSV parameters from the JSON file and updates _processing."""
        if not os.path.exists(self.json_path):
            print(
                f"Warning: HSV config file not found at {self.json_path}. Using current parameters.")
            return

        try:
            with open(self.json_path, 'r') as f:
                config = json.load(f)

            updated = False
            if 'HUE' in config:
                self._processing.HUE = float(
                    config['HUE'])  # Ensure float type
                updated = True
            if 'SATURATION' in config:
                self._processing.SATURATION = float(
                    config['SATURATION'])  # Ensure float type
                updated = True
            if 'VALUE' in config:
                self._processing.VALUE = float(
                    config['VALUE'])  # Ensure float type
                updated = True

            if not updated:
                print(
                    f"Warning: HSV config file {self.json_path} did not contain HUE, SATURATION, or VALUE keys. Using current parameters.")

        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from {self.json_path}. File might be corrupted. Using current parameters.")
        except ValueError as e:
            print(
                f"Error converting values from {self.json_path}: {e}. Ensure HUE, SATURATION, VALUE are numbers. Using current parameters.")
        except Exception as e:
            print(
                f"An unexpected error occurred while reading {self.json_path}: {e}. Using current parameters.")

    def worker(self):
        """Continuously captures and processes frames."""
        try:
            print("Camera worker started.")
            while True:
                # Update HSV parameters periodically
                self._frame_counter += 1
                if self._frame_counter >= self.update_interval:
                    self._update_hsv_parameters_from_json()
                    self._frame_counter = 0  # Reset counter

                # Get frames
                frames = self._camera.get_frames()
                if frames is None:
                    # Handle frame acquisition failure - maybe add a small sleep or break
                    print("Failed to get frames, attempting to continue...")
                    continue  # Skip processing for this loop iteration

                # Process frames
                color_corr, depth_aligned = self._processing.process_frames(
                    frames)

                if color_corr is not None and depth_aligned is not None:
                    self.frames = (color_corr, depth_aligned)
                else:
                    print("Warning: Frame processing failed, frames not updated.")
                    # Could potentially re-initialize processing or camera here if failures persist

                # Add a small delay to avoid 100% CPU usage if get_frames is too fast or returns quickly on error
                # time.sleep(0.001) # Consider adding time import if needed

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping worker.")
        except Exception as e:
            print(f"An unexpected error occurred in worker loop: {e}")
        finally:
            self._camera.stop()
            print("Camera worker stopped.")

    def close(self):
        """External method to stop the camera."""
        # This might be called from another thread or the main program to shut down
        self._camera.stop()
