import numpy as np

# Constants (tweak these to your camera & robot)
FOCAL_PX = 610.98       # focal length in px
CX_PX, CY_PX = 320.0, 240.0  # image center in px
CAMERA_H_M = 0.20         # camera height above robot origin (m)
M2IN = 39.3701      # meters → inches


def locate_detection(
    robot_x_in: float,
    robot_y_in: float,
    robot_theta_rad: float,   # now in radians, CCW positive
    detection: dict
) -> dict:
    """
    Given:
      robot_x_in, robot_y_in   : robot’s current X/Y in inches
      robot_theta_rad          : robot’s heading in radians (CCW +)
      detection: {
        "x":     centerX_px,   # image‐plane X in px
        "y":     centerY_px,   # image‐plane Y in px
        "depth": depth_m       # depth (forward) in meters
      }
    Returns a dict with:
      "x", "y": world‐frame horizontal coords in inches
      "z":      object height above robot base plane in inches
    """
    # 1) back‐project to camera frame (m)
    d = detection["depth"]
    px = detection["x"]
    py = detection["y"]

    Xc = d * (px - CX_PX) / FOCAL_PX
    Yc = d
    Zc = d * (py) / FOCAL_PX # CY_PX - py instead of (py)

    # 2) rotate by yaw about Z into world axes (still in meters)
    c, s = np.cos(robot_theta_rad), np.sin(robot_theta_rad)
    Rz = np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]
    ])
    cam_vec = np.array([Xc, Yc, Zc])
    rel_vec = Rz @ cam_vec   # [Xrel_m, Yrel_m, Zrel_m]

    # 3) convert X,Y to inches and translate by robot X,Y
    Xw_in = robot_x_in + rel_vec[0] * M2IN
    Yw_in = robot_y_in + rel_vec[1] * M2IN

    # 4) compute object height above robot plane in inches
    Zw_in = (rel_vec[2] - CAMERA_H_M) * M2IN

    return {"x": Xw_in, "y": Yw_in, "z": Zw_in}
