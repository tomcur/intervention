from dataclasses import dataclass

import numpy as np


def _cot(x: float) -> float:
    return float(1.0 / np.tan(x))


def _arccot(x: float) -> float:
    return float(np.pi / 2.0 - np.arctan(x))


@dataclass
class VehicleGeometry:
    wheel_base: float
    wheel_track: float
    max_inner_wheel_angle: float


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

    def turning_radius_to_steering_angle(self, radius: float) -> float:
        """
        Turns a turning radius (in meters) to a steering angle (between 0.0 and 1.0).
        """
        # Using Formula (1.29) from C. V. Samak, T. V. Samak, and S. Kandhasamy,
        # 'Control Strategies for Autonomous Vehicles’, arXiv:2011.08729 [cs, eess],
        # Nov. 2020, Accessed: Nov. 25, 2020. [Online].
        # Available: http://arxiv.org/abs/2011.08729.

        if radius <= 0.0:
            return 1.0
        wheel_angle = np.arctan(self._wheel_base / radius)
        steer_angle = np.clip(wheel_angle / self._max_wheel_angle, 0.0, 1.0)
        return float(steer_angle)
