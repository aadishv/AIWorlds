import numpy as np
import copy
from constants import MEASUREMENT_ROW, ENGINE_PATH
from poseconv import locate_detection
import json


class Track:
    __slots__ = ("id", "bbox", "detection", "missed")

    def __init__(self, tid, det):
        # det is the dict from your inference: must contain 'x','y','width','height'
        self.id = tid
        # store the full detection dict so you can propagate other fields
        self.detection = det.copy()
        self.bbox = self._xywh_to_xyxy(det)
        self.missed = 0

    def update(self, det):
        self.detection = det.copy()
        self.bbox = self._xywh_to_xyxy(det)
        self.missed = 0

    @staticmethod
    def _xywh_to_xyxy(d):
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        return (x-w/2, y-h/2, x+w/2, y+h/2)

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)


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

        self.tracks = []     # list of Track()
        self.next_track_id = 0
        self.max_missed = 7     # how many frames to keep “gone” objects
        self.max_dist = 50.0  # max pixel‐distance to

    def _match_and_update_tracks(self, detections):
        """
        detections: list of dicts, each with 'x','y','width','height',…
        returns: list of active Track() objects after update
        """
        # 1) compute centroids of new detections
        det_centers = [((d["x"]), (d["y"])) for d in detections]
        used_det = set()
        # 2) for each existing track try to find nearest detection
        for trk in self.tracks:
            cx, cy = trk.center()
            best_i, best_dist = None, self.max_dist
            for i, (dx, dy) in enumerate(det_centers):
                if i in used_det:
                    continue
                dist = ((cx-dx)**2 + (cy-dy)**2)**0.5
                if dist < best_dist:
                    best_dist, best_i = dist, i
            if best_i is not None:
                # matched
                trk.update(detections[best_i])
                used_det.add(best_i)
            else:
                # no match this frame
                trk.missed += 1

        # 3) create new tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in used_det:
                self.tracks.append(Track(self.next_track_id, d))
                self.next_track_id += 1

        # 4) prune old tracks
        self.tracks = [
            trk for trk in self.tracks if trk.missed <= self.max_missed]

        return self.tracks

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
            if i['confidence'] > 0.3
        ]

        # Filter out detections with depth less than 0.5 meters
        tracks = self._match_and_update_tracks(self.detections)

        # replace your detections list with the alive tracks
        # you can carry over any per‐detection fields you like, here we take trk.detection
        self.detections = [trk.detection for trk in tracks]

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
        # if isinstance(self.localization, FakeDetection):
        #     self.localization = OliverLocalization(
        #         self.fl, (0, 0, self.t))
        # self.localization.update(np.reshape(self.app.camera.frames[1][MEASUREMENT_ROW], (1, 640)) * 3.28 * 12, self.t)
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
