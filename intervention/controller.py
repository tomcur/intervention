import math
from typing import Tuple

import numpy as np
import carla

from . import physics
from .carla_utils.manager import TickState


def _least_square_circle_fit(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a circle to the given points.

    See e.g. https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    :param points: np.ndarray of size (N,2)
    :return: A tuple of an np.ndarray [cx, cy], and the circle radius
    """
    xs = points[:, 0]
    ys = points[:, 1]

    us = xs - np.mean(xs)
    vs = ys - np.mean(ys)

    Suu = np.sum(us ** 2)
    Suv = np.sum(us * vs)
    Svv = np.sum(vs ** 2)
    Suuu = np.sum(us ** 3)
    Suvv = np.sum(us * vs * vs)
    Svvv = np.sum(vs ** 3)
    Svuu = np.sum(vs * us * us)

    A = np.array([[Suu, Suv], [Suv, Svv]])

    b = np.array([1 / 2.0 * Suuu + 1 / 2.0 * Suvv, 1 / 2.0 * Svvv + 1 / 2.0 * Svuu])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx * cx + cy * cy + (Suu + Svv) / len(xs))

    cx += np.mean(xs)
    cy += np.mean(ys)

    return np.array([cx, cy]), float(r)


def _project_point_on_circle(
    point: np.ndarray, circle_origin: np.ndarray, circle_radius: float
) -> np.ndarray:
    direction = point - circle_origin
    point_on_circle = (
        circle_origin + (direction / np.linalg.norm(direction)) * circle_radius
    )

    return point_on_circle


def _turning_radius_to(x: float, y: float) -> float:
    """
    Calculates the turning radius assuming we're driving a circle from (0, 0) to (x, y),
    with a line crossing points (0, 0) to (0, 1) being on a tangent with this circle.
    """
    # The circle is defined by
    # (x - h)^2 + (y - k)^2 = r^2
    # with k = 0
    # so, h^2 = r^2
    # and thus r^2 = (x - r)^2 + y^2
    # => r^2 = x^2 - 2xr + r^2 + y^2
    # => 0 = x^2 - 2xr + y^2
    # => 2xr = x^2 + y^2
    # => r = (x^2 + y^2) / (2x)
    if x == 0:
        return math.inf
    radius = (x ** 2 + y ** 2) / (2 * abs(x))
    return radius


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def _angle_between(vector1: np.ndarray, vector2: np.ndarray) -> np.float64:
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.
    """
    v1_u = _unit_vector(vector1)
    v2_u = _unit_vector(vector2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    assert isinstance(angle, np.float64)
    return angle


class PidController:
    def __init__(
        self,
        unit_time_per_step,
        proportional=1.0,
        integral=0.0,
        derivative=0.0,
        integral_discounting_per_step=0.0,
    ):
        """
        A basic PID controller with integral discounting.
        :param integral_discounting_per_step: The multiplier with which to discount
        the control error integral. Must be between 0.0 (no discounting) and 1.0 (only
        consider the latest error value).
        """
        self._coef_proportional = proportional
        self._coef_integral = integral
        self._coef_derivative = derivative
        self._integral_discounting = integral_discounting_per_step

        self._dt = unit_time_per_step
        self._previous_error = None
        self._error_integral = 0.0

    def step(self, error, update=True):
        """
        :param error: The input error
        :param update: When False, calculates control output without affecting
        controller state
        :return: The control output
        """
        error_integral = self._error_integral

        error_integral *= 1.0 - self._integral_discounting
        error_integral += error * self._dt

        if self._previous_error is not None:
            derivative = (error - self._previous_error) / self._dt
        else:
            derivative = 0.0

        control = 0.0
        control += self._coef_proportional * error
        control += self._coef_integral * self._error_integral
        control += self._coef_derivative * derivative

        if update:
            self._previous_error = error
            self._error_integral = error_integral

        return control


def _interpolate_waypoint_n_meters_ahead(
    waypoints: np.ndarray, meters: float
) -> (float, float):
    """
    Gets a waypoint that is `meters` from the origin (at `0, 0`) along the trajectory in
    `waypoints`. This linearly interpolates the trajectory between the waypoints.

    :param waypoints: should be an `np.ndarray` of form [[X1, Y2], [X2, Y2], ...]
    :param meters`: the distance from the origin along the trajectory
    """
    total_dist = np.float(0.0)
    prev_pos = np.array([0, 0])
    cur_pos = np.array([0, 0])

    for waypoint in waypoints[:]:
        dist = np.linalg.norm(waypoint - cur_pos)
        if total_dist + dist >= meters:
            unit = (waypoint - cur_pos) / dist
            [x, y] = (cur_pos + unit * (meters - total_dist)).tolist()
            return x, y
        else:
            total_dist += dist
            prev_pos = cur_pos
            cur_pos = waypoint

    unit = (cur_pos - prev_pos) / dist
    [x, y] = (cur_pos + unit * (meters - total_dist)).tolist()
    return x, y


class VehicleController:
    def __init__(
        self,
        vehicle_geometry: physics.VehicleGeometry,
        waypoint_step_gap: int = 5,
        unit_time_per_step: float = 0.1,
    ):
        self._waypoint_step_gap = waypoint_step_gap
        self._dt = unit_time_per_step
        self._kinematic_bicycle = physics.KinematicBicycle(vehicle_geometry)

        self._speed_control = PidController(
            unit_time_per_step,
            proportional=0.5,
            integral=0.05,
            derivative=0.02,
            integral_discounting_per_step=0.04,
        )
        self._brake_control = PidController(
            unit_time_per_step,
            proportional=0.5,
            integral=0.05,
            derivative=0.02,
            integral_discounting_per_step=0.04,
        )

    def step(
        self, state: TickState, waypoints: np.ndarray, update_pids=True
    ) -> carla.VehicleControl:
        """
        :param state: current tick's state data
        :param waypoints: should be an `np.ndarray` of form [[X1, Y2], [X2, Y2], ...]
        :param update_pids: whether to update the low-level PID controllers with this
        input
        """
        targets = np.insert(waypoints, 0, [0, 0], axis=0)

        deltas = np.linalg.norm(targets[:-1] - targets[1:], axis=1)
        target_speed = deltas[:3].mean() / (self._waypoint_step_gap * self._dt)

        acceleration = target_speed - state.speed

        logger.trace(f"Target speed {target_speed * 60 * 60 / 1000}")

        x, y = _interpolate_waypoint_n_meters_ahead(waypoints, 5.0)
        radius = _turning_radius_to(x, y)
        steering_angle = self._kinematic_bicycle.turning_radius_to_steering_angle(
            radius
        )
        if x < 0.0:
            steering_angle = -steering_angle

        # Hacky heuristic to allow agent to more easily come to a full stop
        if target_speed * 60.0 * 60.0 / 1000.0 < 3.5:
            throttle = 0.0
            steering_angle = 0.0
            brake = self._brake_control.step(state.speed, update=update_pids)
            brake = max(brake, 0.05)
        else:
            throttle = self._speed_control.step(acceleration, update=update_pids)
            brake = self._brake_control.step(-acceleration, update=update_pids)

        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steering_angle = np.clip(steering_angle, -1.0, 1.0)

        return carla.VehicleControl(
            throttle=throttle, steer=steering_angle, brake=brake
        )
