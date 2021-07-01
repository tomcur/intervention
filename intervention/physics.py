from dataclasses import dataclass

import numpy as np
import math


def _cot(x: float) -> float:
    return float(1.0 / np.tan(x))


def _arccot(x: float) -> float:
    return float(np.pi / 2.0 - np.arctan(x))


@dataclass
class VehicleGeometry:
    """

    The geometry is defined as

        front
    ┏━━━━━━━━━━━┓
    ┃     w     ┃
    █┈┈┈┈┈┬┈┈┈┈┈█
    ┃     ┊ l   ┃
    ┃   ┬ ╳     ┃
    ┃ δ ┊ ┊     ┃
    █┈┈┈┴┈┴┈┈┈┈┈█
    ┃           ┃
    ┗━━━━━━━━━━━┛
        rear

    where
    w = wheel track (distance between front wheels)
    l = wheel base (distance between front and rear axle)
    δ = rear axle offset from the vehicle origin point (the 0-coordinate as given by
    CARLA).
    ╳ marks the vehicle origin point

    """

    wheel_base: float
    wheel_track: float
    max_inner_wheel_angle: float
    rear_axle_longitudinal_offset: float

    def origin_turning_radius(self, x: float, y: float) -> float:
        """
        Calculates the turning radius of the vehicle origin point assuming we're driving
        a circle from (0, 0) to (x, y), the longitudinal axis (a line determined by
        (0, -rear_axle_longitudinal_offset) to
        (0, -rear_axle_longitudinal_offset+wheel_base)
        being a tangent to this circle.
        """
        if x == 0:
            return math.inf

        k = self.rear_axle_longitudinal_offset

        radius = math.sqrt(((x ** 2 + y ** 2 + 2 * y * k) / (2 * x)) ** 2 + k ** 2)

        return max(radius, self.rear_axle_longitudinal_offset+0.0001)

    def origin_radius_to_rear_radius(self, radius) -> float:
        """
        Calculate the longitudinal central rear turning radius based on the
        (longitudinally central) origin's turning radius.
        """
        rear_radius = math.sqrt(radius ** 2 - self._rear_axle_longitudinal_offset ** 2)
        return rear_radius


class KinematicBicycle:
    def __init__(self, vehicle_geometry: VehicleGeometry):
        # Using Formula (1) from V. Arvind, 'Optimizing the turning radius of a vehicle
        # using symmetric four wheel steering system', vol. 4, no. 12, p. 8, 2013.

        wheel_track_to_center = vehicle_geometry.wheel_track / 2.0
        self._max_wheel_angle = _arccot(
            wheel_track_to_center / vehicle_geometry.wheel_base
            + _cot(vehicle_geometry.max_inner_wheel_angle)
        )
        self._wheel_base = vehicle_geometry.wheel_base
        self._rear_axle_longitudinal_offset = (
            vehicle_geometry.rear_axle_longitudinal_offset
        )

    def turning_radius_to_steering_angle(self, radius: float) -> float:
        """
        Turns a turning radius (in meters) to a steering angle (between 0.0 and 1.0).
        """
        # Using Formula (1.29) from C. V. Samak, T. V. Samak, and S. Kandhasamy,
        # 'Control Strategies for Autonomous Vehicles’, arXiv:2011.08729 [cs, eess],
        # Nov. 2020, Accessed: Nov. 25, 2020. [Online].
        # Available: http://arxiv.org/abs/2011.08729.

        rear_radius = math.sqrt(radius ** 2 - self._rear_axle_longitudinal_offset ** 2)
        front_radius = math.sqrt(rear_radius ** 2 + self._wheel_base ** 2)

        if front_radius <= 0.0:
            return 1.0
        wheel_angle = np.arctan(self._wheel_base / front_radius)
        steer_angle = np.clip(wheel_angle / self._max_wheel_angle, 0.0, 1.0)
        return float(steer_angle)
