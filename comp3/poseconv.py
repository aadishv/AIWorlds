import math

# Constants (tweak these to your camera & robot)
FOCAL_PX     = 610.98      # focal length in pixels
CX_PX, CY_PX = 320.0, 240.0 # image center in pixels
CAMERA_H_M   = 0.20        # camera height above robot origin in meters

M2IN = 39.3701             # meters → inches

def locate_detection(
    robot_x_in, robot_y_in, robot_theta_deg,
    detection
):
    """
    robot_x_in, robot_y_in   : robot’s current X/Y in inches
    robot_theta_deg          : robot’s heading (yaw) in degrees
    detection: {
      "x": centerX_px,       # pixel
      "y": centerY_px,       # pixel
      "depth": depth_m       # meters
    }
    Returns { "x": Xw_in, "y": Yw_in, "z": Zw_in }
      • Xw_in, Yw_in are in inches
      • Zw_in is in inches (height above robot’s tracking plane)
    """

    # 1) Camera‐frame coords (in meters)
    d   = detection["depth"]   # [m]
    px  = detection["x"]
    py  = detection["y"]

    Xc = d * (px - CX_PX) / FOCAL_PX
    Yc = d
    Zc = d * (CY_PX - py) / FOCAL_PX

    # 2) Rotate by yaw θ into world axes (still in meters)
    th = math.radians(robot_theta_deg)
    ct, st = math.cos(th), math.sin(th)

    # pure‐yaw rotation about Z
    Xrel_m =  ct*Xc - st*Yc
    Yrel_m =  st*Xc + ct*Yc
    Zrel_m =  Zc

    # 3) Convert the planar offsets to inches and translate
    Xrel_in = Xrel_m * M2IN
    Yrel_in = Yrel_m * M2IN

    Xw_in = robot_x_in + Xrel_in
    Yw_in = robot_y_in + Yrel_in

    # 4) Compute object Z above robot’s tracking plane (in meters)
    Zw_m = Zrel_m - CAMERA_H_M
    Zw_in = Zw_m * M2IN

    return {"x": Xw_in, "y": Yw_in, "z": Zw_in}


# Example usage
if __name__ == "__main__":
    robot_pos = (  60.0,  48.0,  30.0 )  # x=60″, y=48″, θ=30°
    det = {"x": 350, "y": 180, "depth": 3.0}  # depth=3 m
    print(locate_detection(*robot_pos, det))
    # → {'x': … inches, 'y': … inches, 'z': … meters}
