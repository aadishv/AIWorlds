from localization import Localization
import numpy as np


# this a measurement taken from the pose [0, 62, 30]

# coordinate system: 0 is up, going counter-clockwise
# • 0 → top wall (y = +W)
# • 1 → left wall (x = −W)
# • 2 → bottom wall (y = −W)
# • 3 → right wall (x = +W)

class OliverLocalization:
    def __init__(self, fl, initial_pose=(0, 0, 0)):
        self.pose = initial_pose
        self.W = 3 * 24
        self.FOCAL_LENGTH = fl
        self.WALLS = {
            0: 0,  # top
            1: 90,  # left
            2: 180,  # bottom
            3: 270  # right
        }
    # o4-mini GENERATED

    def find_wall(self, x0, y0, theta_deg):
        """
        Raycast from (x0,y0) in direction theta_deg (0°→+Y axis, CCW positive).
        Returns wall index {0: top(y=+W), 1: left(x=-W), 2: bottom(y=-W), 3: right(x=+W)}.
        """
        # convert to radians
        th = np.deg2rad(theta_deg)
        # ray direction: dx/dy so that th=0→(0,1)
        dx = np.sin(th)
        dy = np.cos(th)

        # very small to avoid division by zero
        eps = 1e-9

        # compute the four possible ts
        t = np.empty(4, dtype=float)
        # 0: top  (y = +W)
        t[0] = (self.W - y0) / dy if abs(dy) > eps else np.inf
        # 1: left (x = -self.W)
        t[1] = (-self.W - x0) / dx if abs(dx) > eps else np.inf
        # 2: bottom(y = -self.W)
        t[2] = (-self.W - y0) / dy if abs(dy) > eps else np.inf
        # 3: right(x = +self.W)
        t[3] = (self.W - x0) / dx if abs(dx) > eps else np.inf

        # only forward‐facing intersections
        valid = t > 0

        # compute hit points
        xs = x0 + t * dx
        ys = y0 + t * dy

        # enforce that the hit lies within the segment of that wall
        # wall 0 & 2: y‑walls, x must lie in [−W, W]
        valid[0] &= (np.abs(xs[0]) <= self.W)
        valid[2] &= (np.abs(xs[2]) <= self.W)
        # wall 1 & 3: x‑walls, y must lie in [−W, W]
        valid[1] &= (np.abs(ys[1]) <= self.W)
        valid[3] &= (np.abs(ys[3]) <= self.W)

        # discard invalid
        t[~valid] = np.inf

        # pick the smallest positive t
        wall_index = int(np.argmin(t))
        return wall_index

    def run_points_w_theta(self, angles, depths, theta):
        x_constraints = []
        y_constraints = []

        for p_angle, p_depth in zip(angles, depths):  # point angle, point depth
            wall = self.find_wall(*self.pose[:2], p_angle)
            angle = p_angle - self.WALLS[wall]
            wall_amt = self.W if wall in [0, 3] else -self.W
            new_coord = wall_amt - np.cos(np.radians(angle)) * p_depth
            if wall in [0, 2]:
                y_constraints.append(new_coord)
            elif wall in [1, 3]:
                x_constraints.append(new_coord)
        x = np.percentile(x_constraints, 50) if x_constraints else self.pose[0]
        y = np.percentile(y_constraints, 50) if y_constraints else self.pose[1]
        self.pose = (x, y, theta)

    def horizontal_angle(self, offset, focal_length):  # returns degrees
        theta = np.arctan2(offset, focal_length)    # radians
        return np.degrees(theta)

    def update(self, measurements, theta):
        angles = []
        depths = []
        for i in range(1, 640):
            if measurements[i] <= 0:
                continue
            angles.append(self.horizontal_angle(i - 320, self.FOCAL_LENGTH))
            depths.append(measurements[i])
        self.run_points_w_theta(np.array(angles), np.array(depths), theta)
'''
example:
```py
localization = OliverLocalization(10, (0, 60, 0))

localization.update(depths, theta)
print(localization.pose)
```
'''
