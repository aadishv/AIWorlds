import numpy as np

class Localization:
    def __init__(self,
                 initial_pose,
                 focal_length,
                 sensor_offsets,
                 wall_coordinate=70.205,
                 acceptable_error=0.5):
        """
        initial_pose: tuple (x, y, θ) in same linear units & degrees
        focal_length: scalar, same linear unit as sensor_offsets
        sensor_offsets: tuple (x_off, y_off, orientation) where orientation is
                        the angle of the sensor relative to the chassis in degrees
        wall_coordinate: absolute coordinate of walls at 0/±wall_coordinate
        acceptable_error: depth matching tolerance
        """
        # pose = [x, y, heading(deg)]
        self.pose = np.array(initial_pose, dtype=float)
        self.fl = float(focal_length)
        self.x_off, self.y_off, self.sensor_orientation = sensor_offsets
        self.wall_coordinate = float(wall_coordinate)
        self.eps = float(acceptable_error)
        # Pre‐define candidate wall headings in degrees
        self.wall_angles = np.array([0.0, 90.0, 180.0, 270.0])
        self.first_it = True

    def update(self, offsets, depths, imu):
        """
        offsets: array
        depths: array
        imu: object with attribute `heading` in degrees
        Returns: updated pose (x, y, heading) as numpy array
        """

        # 1) compute angle offsets (deg) in one shot
        angles_off = np.degrees(np.arctan(offsets / self.fl))
        # cache old pose & imu heading
        prev_x, prev_y, _ = self.pose
        cur_heading = imu

        x_vals = []
        y_vals = []

        # 2) for each detected point, find which wall it hit and
        #    back out an (x or y) coordinate
        i = 0
        for θ_off, depth in zip(angles_off, depths):
            found = False
            for wall in self.wall_angles:
                # relative bearing from robot -> wall, including sensor orientation
                θ = np.deg2rad(cur_heading + self.sensor_orientation + θ_off - wall)

                # straight‐line distance from robot center to infinite wall
                if wall == 0.0:
                    d_wall = abs(self.wall_coordinate - prev_x)
                elif wall == 90.0:
                    d_wall = abs(self.wall_coordinate - prev_y)
                elif wall == 180.0:
                    d_wall = abs(-self.wall_coordinate - prev_x)
                else:  # 270
                    d_wall = abs(-self.wall_coordinate - prev_y)

                # predicted depth along sensor ray
                pred_depth = (d_wall
                              - (np.cos(θ)*self.x_off - np.sin(θ)*self.y_off)) \
                              / np.cos(θ)
                # if within tolerance, assign this point to that wall
                if abs(pred_depth - depth) <= self.eps or self.first_it:
                    # project the measured depth back to an (x or y)
                    dist = (np.cos(θ)*depth
                            + np.cos(θ)*self.x_off
                            - np.sin(θ)*self.y_off)
                    found = True
                    if wall == 0.0:        # +X wall
                        x_vals.append(dist - self.wall_coordinate)
                    elif wall == 90.0:     # +Y wall
                        y_vals.append(dist - self.wall_coordinate)
                    elif wall == 180.0:    # -X wall
                        x_vals.append(-dist + self.wall_coordinate)
                    else:                  # -Y wall
                        y_vals.append(-dist + self.wall_coordinate)
                    break
            if not found:
                i += 1
        new_x = np.mean(x_vals) if x_vals else prev_x
        new_y = np.mean(y_vals) if y_vals else prev_y

        # final pose: (x, y, heading)
        self.pose = np.array([new_x, new_y, cur_heading], dtype=float)
        self.first_it = False
        return self.pose
