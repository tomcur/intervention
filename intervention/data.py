import abc
import csv
import dataclasses
import queue
import sys
import threading
import zipfile
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Deque, Optional, TextIO, Tuple, Union

import carla
import dataclass_csv
import numpy as np
from PIL import Image
from typing_extensions import Literal

from .carla_utils import ManagedEpisode, TickState

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


EndStatus = Union[Literal["success", "collision", "stuck", "off_course", "unknown"]]


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
    weather: str = ""
    end_status: str = "unknown"
    collisions: int = 0
    distance_travelled: float = 0.0
    student_distance_travelled: float = 0.0
    teacher_distance_travelled: float = 0.0
    route_length_completed: float = 0.0
    route_length: float = 0.0
    interventions: int = 0
    ticks: int = 0
    ticks_per_second: float = 0.0

    @classmethod
    def from_managed_episode(clss, managed_episode: ManagedEpisode) -> "EpisodeSummary":
        return clss(town=managed_episode.town, weather=managed_episode.weather)

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
    rgb_filename: Optional[str]
    teacher_waypoints_filename: Optional[str]
    student_waypoints_filename: Optional[str]
    student_image_targets_filename: Optional[str]
    student_image_heatmaps_filename: Optional[str]
    ticks_engaged: Optional[int]
    ticks_to_intervention: Optional[int]
    ticks_intervened: Optional[int]
    ticks_to_engagement: Optional[int]
    ticks_to_end: Optional[int]
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
        teacher_waypoints: Optional[np.ndarray],
        student_waypoints: np.ndarray,
        model_image_targets: np.ndarray,
        model_image_heatmaps: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        """Add one example of student driving to the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def push_teacher_driving(
        self,
        step: int,
        teacher_waypoints: np.ndarray,
        student_waypoints: Optional[np.ndarray],
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        """Add one example of teacher driving to the store."""
        raise NotImplementedError


class BlackHoleStore(Store):
    """Episode store backed by a black hole."""

    def push_student_driving(
        self,
        step: int,
        teacher_waypoints: Optional[np.ndarray],
        student_waypoints: np.ndarray,
        model_image_targets: np.ndarray,
        model_image_heatmaps: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        pass

    def push_teacher_driving(
        self,
        step: int,
        teacher_waypoints: np.ndarray,
        student_waypoints: Optional[np.ndarray],
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        pass


class ZipStoreBackend(Store):
    """
    Episode store backed by zip-file.
    """

    STORE_NUM_TICKS_BEFORE_INTERVENTION = (5 * 5) + 15

    def __init__(
        self, archive: zipfile.ZipFile, csv_file: TextIO, metrics_only: bool = False
    ):
        self._archive = archive
        self._csv = csv
        self._metrics_only = metrics_only

        self._frame_data_queue: Deque[Tuple[int, FrameData]] = deque()

        self._teacher_in_control = True
        self._intervention_tick = 0
        self._engagement_tick = 0

        self._csv_file = csv_file
        self._csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=list(FrameData.__annotations__.keys()),
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def push_student_driving(
        self,
        tick: int,
        teacher_waypoints: Optional[np.ndarray],
        student_waypoints: np.ndarray,
        model_image_targets: np.ndarray,
        model_image_heatmaps: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        if self._teacher_in_control:
            self._engagement_tick = tick
            self._teacher_in_control = False
            self._store_teacher_driving(reason="engagement")

        rgb_filename = None
        teacher_waypoints_filename = None
        student_waypoints_filename = None
        model_image_targets_filename = None
        model_image_heatmaps_filename = None

        if not self._metrics_only:
            rgb_filename = f"{tick:05d}-rgb-student.png"
            self._add_rgb_image(rgb_filename, state.rgb)

            if teacher_waypoints is not None:
                teacher_waypoints_filename = f"{tick:05d}-teacher-waypoints.npy"
                buffer = BytesIO()
                np.save(buffer, teacher_waypoints)
                self._add_file(teacher_waypoints_filename, buffer.getvalue())

            student_waypoints_filename = f"{tick:05d}-student-waypoints.npy"
            model_image_targets_filename = f"{tick:05d}-image-targets-student.npy"
            model_image_heatmaps_filename = f"{tick:05d}-image-heatmaps-student.npy"

            buffer = BytesIO()
            np.save(buffer, student_waypoints)
            self._add_file(student_waypoints_filename, buffer.getvalue())

            buffer = BytesIO()
            np.save(buffer, model_image_targets)
            self._add_file(model_image_targets_filename, buffer.getvalue())

            buffer = BytesIO()
            np.save(buffer, model_image_heatmaps)
            self._add_file(model_image_heatmaps_filename, buffer.getvalue())

        orientation = state.rotation.get_forward_vector()

        frame_data = FrameData(
            tick=tick,
            command=int(state.command),
            controller="student",
            rgb_filename=rgb_filename,
            teacher_waypoints_filename=teacher_waypoints_filename,
            student_waypoints_filename=student_waypoints_filename,
            student_image_targets_filename=model_image_targets_filename,
            student_image_heatmaps_filename=model_image_heatmaps_filename,
            ticks_engaged=tick - self._engagement_tick,
            ticks_to_intervention=None,
            ticks_intervened=None,
            ticks_to_engagement=None,
            ticks_to_end=None,
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
        self._frame_data_queue.append((tick, frame_data))

    def push_teacher_driving(
        self,
        tick: int,
        teacher_waypoints: np.ndarray,
        student_waypoints: Optional[np.ndarray],
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        if not self._teacher_in_control:
            self._intervention_tick = tick
            self._teacher_in_control = True
            self._store_student_driving(reason="intervention")

        rgb_filename = None
        teacher_waypoints_filename = None
        student_waypoints_filename = None

        if not self._metrics_only:
            rgb_filename = f"{tick:05d}-rgb-teacher.png"
            self._add_rgb_image(rgb_filename, state.rgb)

            teacher_waypoints_filename = f"{tick:05d}-teacher-waypoints.npy"

            buffer = BytesIO()
            np.save(buffer, teacher_waypoints)
            self._add_file(teacher_waypoints_filename, buffer.getvalue())

            if student_waypoints is not None:
                student_waypoints_filename = f"{tick:05d}-student-waypoints.npy"

                buffer = BytesIO()
                np.save(buffer, student_waypoints)
                self._add_file(student_waypoints_filename, buffer.getvalue())

        orientation = state.rotation.get_forward_vector()

        frame_data = FrameData(
            tick=tick,
            command=int(state.command),
            controller="teacher",
            rgb_filename=rgb_filename,
            teacher_waypoints_filename=teacher_waypoints_filename,
            student_waypoints_filename=student_waypoints_filename,
            student_image_targets_filename=None,
            student_image_heatmaps_filename=None,
            ticks_engaged=None,
            ticks_to_intervention=None,
            ticks_intervened=tick - self._intervention_tick,
            ticks_to_engagement=None,
            ticks_to_end=None,
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
        self._frame_data_queue.append((tick, frame_data))

    def _add_rgb_image(self, filename: str, rgb: np.ndarray) -> None:
        buffer = BytesIO()
        im = Image.fromarray(rgb)
        im.save(buffer, format="PNG")
        self._add_file(filename, buffer.getvalue())

    def _add_file(self, filename: str, data: bytes) -> None:
        self._archive.writestr(filename, data)

    def _store_teacher_driving(
        self, reason: Union[Literal["engagement"], Literal["end"]]
    ) -> None:
        if len(self._frame_data_queue) == 0:
            return
        last_tick, _ = self._frame_data_queue[-1]

        for (tick, frame_data) in self._frame_data_queue:
            if reason == "engagement":
                frame_data["ticks_to_engagement"] = last_tick - tick
            elif reason == "end":
                frame_data["ticks_to_end"] = last_tick - tick

            self._csv_writer.writerow(frame_data)
        self._frame_data_queue.clear()

    def _store_student_driving(
        self, reason: Union[Literal["intervention"], Literal["end"]]
    ) -> None:
        if len(self._frame_data_queue) == 0:
            return
        last_tick, _ = self._frame_data_queue[-1]

        for (tick, frame_data) in self._frame_data_queue:
            if reason == "intervention":
                frame_data["ticks_to_intervention"] = last_tick - tick
            elif reason == "end":
                frame_data["ticks_to_end"] = last_tick - tick

            self._csv_writer.writerow(frame_data)
        self._frame_data_queue.clear()

    def stop(self) -> None:
        if self._teacher_in_control:
            self._store_teacher_driving(reason="end")
        else:
            self._store_student_driving(reason="end")


def _zip_store_worker(queue: queue.Queue, zip_store_backend: ZipStoreBackend):
    while True:
        method, data = queue.get()
        if method == "push_student_driving":
            zip_store_backend.push_student_driving(*data)
        elif method == "push_teacher_driving":
            zip_store_backend.push_teacher_driving(*data)
        elif method == "stop":
            zip_store_backend.stop()
            break
        else:
            raise Exception("unknown method passed to _zip_store_worker")


class ZipStore(Store):
    """
    Episode store backed by zip-file. This spins up a thread running a ZipStoreBackend
    that proceses and stores the data.
    """

    def __init__(
        self, archive: zipfile.ZipFile, csv_file: TextIO, metrics_only: bool = False
    ):
        self._queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        zip_store_backend = ZipStoreBackend(
            archive, csv_file, metrics_only=metrics_only
        )
        self._worker_thread = threading.Thread(
            target=_zip_store_worker, args=(self._queue, zip_store_backend)
        )
        self._worker_thread.start()

    def push_student_driving(
        self,
        tick: int,
        teacher_waypoints: Optional[np.ndarray],
        student_waypoints: np.ndarray,
        model_image_targets: np.ndarray,
        model_image_heatmaps: np.ndarray,
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        self._queue.put(
            (
                "push_student_driving",
                (
                    tick,
                    teacher_waypoints,
                    student_waypoints,
                    model_image_targets,
                    model_image_heatmaps,
                    control,
                    state,
                ),
            )
        )

    def push_teacher_driving(
        self,
        tick: int,
        teacher_waypoints: np.ndarray,
        student_waypoints: Optional[np.ndarray],
        control: carla.VehicleControl,
        state: TickState,
    ) -> None:
        self._queue.put(
            (
                "push_teacher_driving",
                (tick, teacher_waypoints, student_waypoints, control, state),
            )
        )

    def stop(self) -> None:
        self._queue.put(("stop", None))
        self._worker_thread.join()
