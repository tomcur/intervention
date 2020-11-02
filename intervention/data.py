import abc
import csv
import dataclasses
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from typing import List, Optional, TextIO, Tuple, Union

import dataclass_csv
import numpy as np
from typing_extensions import Literal

import carla

from .carla_utils import TickState

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


@dataclasses.dataclass
@dataclass_csv.dateformat("%Y-%m-%dT%H:%M:%S.%f%z")
class EpisodeSummary:
    uuid: str = ""
    collection_start_datetime: datetime = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    collection_end_datetime: datetime = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    town: str = ""
    terminated: bool = False
    success: bool = False
    collisions: int = 0
    distance_travelled: float = 0.0
    interventions: int = 0
    ticks: int = 0
    ticks_per_second: float = 0.0

    def end(self):
        self.collection_end_datetime = datetime.now(timezone.utc)
        self.ticks_per_second = (
            self.ticks
            / (
                self.collection_end_datetime - self.collection_start_datetime
            ).total_seconds()
        )

    def as_csv_writeable_dict(self):
        values = self.__dict__
        for (key, value) in values.items():
            if isinstance(value, bool):
                values[key] = int(value)
            elif isinstance(value, datetime):
                values[key] = value.isoformat()
        return values


class FrameData(TypedDict):
    tick: int
    command: int
    controller: str
    rgb_filename: str
    student_output_filename: Optional[str]
    time_to_intervention: Optional[int]
    time_to_end: Optional[int]
    time_from_intervention: Optional[int]
    lane_invasion: int  # (int-encoded bool: 0 = False, 1 = True)
    collision: int  # (int-encoded bool: 0 = False, 1 = True)
    location_x: float
    location_y: float
    location_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    speed: float
    orientation_x: float
    orientation_y: float
    orientation_z: float


class Store:
    """Episode store."""

    @abc.abstractmethod
    def push_student_driving(
        self,
        step: int,
        model_output: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        """Add one example of student driving to the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def push_teacher_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        """Add one example of teacher driving to the store."""
        raise NotImplementedError


class BlackHoleStore(Store):
    """Episode store backed by a black hole."""

    def push_student_driving(
        self,
        step: int,
        model_output: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        pass

    def push_teacher_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        pass


class ZipStore(Store):
    """Episode store backed by zip-file."""

    STORE_NUM_TICKS_BEFORE_INTERVENTION = (5 * 5) + 15

    def __init__(self, archive: zipfile.ZipFile, csv_file: TextIO):
        self._archive = archive
        self._csv = csv
        self._recent_student_driving: List[
            Tuple[int, np.ndarray, carla.VehicleControl, TickState]
        ] = []
        self._teacher_in_control = False
        self._intervention_step = 0
        self._latest_step = 0

        self._csv_file = csv_file
        self._csv_writer = csv.DictWriter(
            csv_file, fieldnames=list(FrameData.__annotations__.keys()),
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def push_student_driving(
        self,
        step: int,
        model_output: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        self._teacher_in_control = False
        self._recent_student_driving.append((step, model_output, control, state))
        if len(self._recent_student_driving) > self.STORE_NUM_TICKS_BEFORE_INTERVENTION:
            self._recent_student_driving.pop(0)

    def push_teacher_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        if not self._teacher_in_control:
            self._intervention_step = step
            self._teacher_in_control = True
            self._store_student_driving(reason="intervention")

        rgb_filename = f"{step:05d}-rgb-teacher.bin"
        self._add_file(rgb_filename, state.rgb.tobytes(order="C"))
        orientation = state.rotation.get_forward_vector()
        self._csv_writer.writerow(
            FrameData(
                tick=step,
                command=state.command,
                controller="teacher",
                rgb_filename=rgb_filename,
                student_output_filename=None,
                time_to_intervention=None,
                time_to_end=None,
                time_from_intervention=step - self._intervention_step,
                lane_invasion=int(state.lane_invasion is not None),
                collision=int(state.collision is not None),
                location_x=state.location.x,
                location_y=state.location.y,
                location_z=state.location.z,
                velocity_x=state.velocity.x,
                velocity_y=state.velocity.y,
                velocity_z=state.velocity.z,
                speed=state.speed,
                orientation_x=orientation.x,
                orientation_y=orientation.y,
                orientation_z=orientation.z,
            )
        )

    def _add_file(self, filename: str, data: bytes) -> None:
        self._archive.writestr(filename, data)

    def _store_student_driving(
        self, reason: Union[Literal["intervention"], Literal["end"]]
    ) -> None:
        if len(self._recent_student_driving) == 0:
            return
        last_step, _, _, _ = self._recent_student_driving[-1]

        for (step, model_output, _control, state) in self._recent_student_driving:
            rgb_filename = f"{step:05d}-rgb-student.bin"
            self._add_file(rgb_filename, state.rgb.tobytes(order="C"))

            model_output_filename = f"{step:05d}-output-student.npy"
            buffer = BytesIO()
            np.save(buffer, model_output)
            self._add_file(model_output_filename, buffer.getvalue())

            time_to_intervention = None
            time_to_end = None

            if reason == "intervention":
                time_to_intervention = self._intervention_step - step
            elif reason == "end":
                time_to_end = last_step - step

            orientation = state.rotation.get_forward_vector()
            self._csv_writer.writerow(
                FrameData(
                    tick=step,
                    command=state.command,
                    controller="student",
                    rgb_filename=rgb_filename,
                    student_output_filename=model_output_filename,
                    time_to_intervention=time_to_intervention,
                    time_to_end=time_to_end,
                    time_from_intervention=None,
                    lane_invasion=int(state.lane_invasion is not None),
                    collision=int(state.collision is not None),
                    location_x=state.location.x,
                    location_y=state.location.y,
                    location_z=state.location.z,
                    velocity_x=state.velocity.x,
                    velocity_y=state.velocity.y,
                    velocity_z=state.velocity.z,
                    speed=state.speed,
                    orientation_x=orientation.x,
                    orientation_y=orientation.y,
                    orientation_z=orientation.z,
                )
            )
        self._recent_student_driving = []

    def stop(self) -> None:
        self._store_student_driving(reason="end")
