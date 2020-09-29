from typing import List, Tuple

import numpy as np
import carla

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

    return np.array([cx, cy]), r


def _project_point_on_circle(
    point: np.ndarray, circle_origin: np.ndarray, circle_radius: float
) -> np.ndarray:
    direction = point - circle_origin
    point_on_circle = (
        circle_origin + (direction / np.linalg.norm(direction)) * circle_radius
    )

    return point_on_circle


def _unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def _angle_between(vector1, vector2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.
    """
    v1_u = _unit_vector(vector1)
    v2_u = _unit_vector(vector2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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

    def step(self, error):
        self._error_integral *= 1.0 - self._integral_discounting
        self._error_integral += error * self._dt

        if self._previous_error is not None:
            derivative = (error - self._previous_error) / self._dt
        else:
            derivative = 0.0

        control = 0.0
        control += self._coef_proportional * error
        control += self._coef_integral * self._error_integral
        control += self._coef_derivative * derivative

        return control


class VehicleController:
    def __init__(self, waypoint_step_gap: int = 5, unit_time_per_step: float = 0.1):
        self._waypoint_step_gap = waypoint_step_gap
        self._dt = unit_time_per_step

        self._speed_control = PidController(
            unit_time_per_step, proportional=0.5, integral=0.05, derivative=0.02,
        )
        self._brake_control = PidController(
            unit_time_per_step, proportional=0.5, integral=0.05, derivative=0.02,
        )
        self._turn_control = PidController(
            unit_time_per_step,
            proportional=0.5,
            integral=0.05,
            derivative=0.02,
            integral_discounting_per_step=0.08,
        )

        self._waypoint_steps_for_steering = {
            1: 4,  # Left
            2: 4,  # Right
            3: 2,  # Straight
            4: 3,  # Lane follow
        }

    def step(self, state: TickState, waypoints: np.ndarray) -> carla.VehicleControl:
        # Swap Y and X coordinates, and add our current position (at origin)
        # In the Learning by Cheating codebase, the following is used instead.
        # targets = [(0, 0)]
        # for i in range(STEPS):
        #     pixel_dx, pixel_dy = world_pred[i]
        #     angle = np.arctan2(pixel_dx, pixel_dy)
        #     dist = np.linalg.norm([pixel_dx, pixel_dy])
        #     targets.append([dist * np.cos(angle), dist * np.sin(angle)])
        # targets = np.array(targets)

        waypoints = waypoints[..., [1, 0]]
        targets = np.insert(waypoints, 0, [0, 0], axis=0)

        # Calculate throttle and braking
        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (
            self._waypoint_step_gap * self._dt
        )

        acceleration = target_speed - state.speed

        throttle = self._speed_control.step(acceleration)
        brake = self._brake_control.step(-acceleration)

        # Calculate steering
        circle_origin, circle_radius = _least_square_circle_fit(waypoints)
        idx_point_to_turn_to = self._waypoint_steps_for_steering[state.command]
        point_to_turn_to = targets[idx_point_to_turn_to + 1]
        point = _project_point_on_circle(point_to_turn_to, circle_origin, circle_radius)

        angle = _angle_between([1.0, 0], point)
        if point[1] < 0:
            angle = -angle

        steering = self._turn_control.step(angle)

        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)

        return carla.VehicleControl(throttle=throttle, steer=steering, brake=brake)
