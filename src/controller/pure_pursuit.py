import numpy as np

from src.utils.carla_utils_lane import VehicleStates


def get_target_point(lookahead, polyline):
    intersections = []
    for j in range(len(polyline) - 1):
        pt1 = polyline[j]
        pt2 = polyline[j + 1]
        intersections += circle_line_segment_intersection((0, 0), lookahead, pt1, pt2, full_line=False)
    filtered = [p for p in intersections if p[0] > 0]
    if len(filtered) == 0:
        return None
    return filtered[0]

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** .5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

param_Kp = 1
param_Ki = 0.05
param_Kd = 0.07
param_K_dd = 0.4


class PurePursuit:
    def __init__(self, k_dd=param_K_dd, wheel_base=2.875,
                 waypoint_shift=1.437):
        self.K_dd = k_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift

    def get_control(self, waypoints, speed):
        # transform x coordinates of waypoints such that coordinate origin is in rear wheel
        waypoints[:, 0] += self.waypoint_shift
        waypoints[:, 1] *= -1
        look_ahead_distance = np.clip(self.K_dd * speed, 3, 20)

        track_point = get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            return 0

        alpha = np.arctan2(track_point[1], track_point[0])

        # Change the steer output with the lateral controller.
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)

        # undo transform to waypoints
        waypoints[:, 0] -= self.waypoint_shift
        waypoints[:, 1] *= -1
        return steer


class PIDController:
    def __init__(self, k_p, k_i, k_d, set_point):
        self.Kp = k_p
        self.Ki = k_i
        self.Kd = k_d
        self.set_point = set_point
        self.int_term = 0
        self.derivative_term = 0
        self.last_error = None

    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.int_term += error * self.Ki * dt
        if self.last_error is not None:
            self.derivative_term = (error - self.last_error) / dt * self.Kd
        self.last_error = error
        return self.Kp * error + self.int_term + self.derivative_term


class PurePursuitPlusPID:
    def __init__(self, pure_pursuit=PurePursuit(), pid=PIDController(param_Kp, param_Ki, param_Kd, 0)):
        self.pure_pursuit = pure_pursuit
        self.pid = pid

    def get_control(self, vehicle, waypoints, desired_speed, dt):
        vehicle_states = VehicleStates(vehicle)
        vehicle_overall_speed_linear = vehicle_states.overall_linear_velocity
        self.pid.set_point = desired_speed
        throttle = self.pid.get_control(vehicle_overall_speed_linear, dt)
        steer = self.pure_pursuit.get_control(waypoints, vehicle_overall_speed_linear)
        return throttle, steer


if __name__ == "__main__":
    pp = PurePursuit()
    pid_ = PIDController(param_Kp, param_Ki, param_Kd, 0)
    controller = PurePursuitPlusPID(pure_pursuit=pp, pid=pid_)
