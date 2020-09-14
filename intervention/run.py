from typing import Any, Tuple, Dict, List

import multiprocessing

import abc
import itertools
import zipfile
from pathlib import Path
import uuid
import csv
import numpy as np
import torch
from loguru import logger
import carla

from .carla_utils import connect, TickState
from . import visualization, exceptions

from .learning_by_cheating import image
from .learning_by_cheating import birdview


def controls_differ(observation, supervisor_control, model_control) -> bool:
    """Determines whether supervisor and model controls differ.
    This function is not guaranteed to be symmetric.
    """
    return (
        supervisor_control.hand_brake != model_control.hand_brake
        or supervisor_control.reverse != model_control.reverse
        or (
            # FIXME high threshold because of jittery throttle
            abs(supervisor_control.throttle - model_control.throttle)
            >= 0.95
        )
        or (abs(supervisor_control.steer - model_control.steer) >= 0.25)
        or (
            observation["speed"] > 5.0 * 1000 / 60 / 60
            and abs(supervisor_control.brake - model_control.brake) >= 0.95
        )
        or (
            observation["speed"] > 10.0 * 1000 / 60 / 60
            and abs(supervisor_control.brake - model_control.brake) >= 0.8
        )
        or (
            observation["speed"] > 30.0 * 1000 / 60 / 60
            and abs(supervisor_control.brake - model_control.brake) >= 0.5
        )
    )


def controls_difference(state: TickState, supervisor_control, model_control) -> float:
    diff = 0

    if supervisor_control.hand_brake != model_control.hand_brake:
        diff += 1
    if supervisor_control.reverse != model_control.reverse:
        diff += 1

    diff += 0.7 * abs(supervisor_control.throttle - model_control.throttle)
    diff += 1.5 * abs(supervisor_control.steer - model_control.steer)

    if state.speed > 30.0 * 1000 / 60 / 60:
        diff += abs(supervisor_control.brake - model_control.brake)
    elif state.speed > 10.0 * 1000 / 60 / 60:
        diff += 0.5 * abs(supervisor_control.brake - model_control.brake)
    elif state.speed > 5.0 * 1000 / 60 / 60:
        diff += 0.25 * abs(supervisor_control.brake - model_control.brake)
    else:
        diff += 0.1 * abs(supervisor_control.brake - model_control.brake)

    return diff


def control_stable(control) -> bool:
    """Determines whether a control is "stable". When a control is stable for a while,
    the controller is assumed to be comfortable handing back control to the model.
    """
    return (
        not control.reverse
        and not control.hand_brake
        and control.brake == 0
        and abs(control.steer) < 0.4
    )


class Comparer:
    def __init__(self, student, teacher):
        self.difference_integral = 0.0
        self._stable_frames = 0
        self.student_in_control = False
        self.student = student
        self.teacher = teacher
        self.student_control = None
        self.teacher_control = None

    def evaluate_and_compare(self, state: TickState) -> None:
        self.student_control = self.student.run_step(
            {
                "rgb": state.rgb,
                "velocity": np.float32(
                    [state.velocity.x, state.velocity.y, state.velocity.z]  # type: ignore
                ),
                # "command": int(command),
                "command": state.command,
            }
        )

        self.teacher_control, _ = self.teacher.run_step(
            {
                "birdview": state.birdview,
                "velocity": np.float32(
                    [state.velocity.x, state.velocity.y, state.velocity.z]  # type: ignore
                ),
                "command": state.command,
            },
            teaching=True,
        )

        logger.trace(self.student_control)
        logger.trace(self.teacher_control)

        if self.student_in_control:
            self.difference_integral *= 0.9
            self.difference_integral += controls_difference(
                state, self.teacher_control, self.student_control
            )
            if self.difference_integral >= 5.0:
                logger.trace("switching to teacher control")
                self.student_in_control = False
                self.difference_integral = 0
        else:
            if control_stable(self.teacher_control):
                self._stable_frames += 1
                if self._stable_frames >= 20:
                    logger.trace("switching to student control")
                    self.student_in_control = True
                    self._stable_frames = 0
            else:
                self._stable_frames = 0

    def switch_control(self):
        """Switch control from the student to the teacher or vice-versa."""
        self.student_in_control = not self.student_in_control


def _prepare_student_agent():
    image_model = image.ImagePolicyModelSS(backbone="resnet34")
    # image_model.load_state_dict(torch.load("../LearningByCheating/ckpts/image/model-10.th"))
    image_model.load_state_dict(torch.load("./model-64.th"))
    image_model.eval()
    student_agent_args = {
        "camera_args": {
            "fixed_offset": 4.0,
            "fov": 90,
            "h": 160,
            "w": 384,
            "world_y": 1.4,
        },
        "pid": {
            "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},
            "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},
            "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},
            "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},
        },
        "steer_points": {"1": 4, "2": 3, "3": 2, "4": 2},
        "model": image_model,
    }
    student_agent = image.ImageAgent(**student_agent_args)
    return student_agent


def _prepare_teacher_agent():
    teacher_model = birdview.BirdViewPolicyModelSS(backbone="resnet18")
    teacher_model.load_state_dict(
        torch.load("../LearningByCheating/ckpts/privileged/model-128.th")
    )
    teacher_model.eval()

    teacher_agent_args = {
        # "steer_points": {"1": 4, "2": 3, "3": 2, "4": 2},
        "model": teacher_model,
    }
    teacher_agent = birdview.BirdViewAgent(**teacher_agent_args)

    return teacher_agent


class Store:
    """Episode store."""

    @abc.abstractmethod
    def push_student_driving(self, step: int, control: Any, rgb: Any) -> None:
        """Add one example of student driving to the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def push_teacher_driving(self, step: int, control: Any, rgb: Any) -> None:
        """Add one example of teacher driving to the store."""
        raise NotImplementedError


class BlackHoleStore(Store):
    """Episode store backed by a black hole."""

    def push_student_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        pass

    def push_teacher_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        pass


class ZipStore(Store):
    """Episode store backed by zip-file."""

    def __init__(self, archive: zipfile.ZipFile, csv: csv.DictWriter):
        self._archive = archive
        self._csv = csv
        self._recent_student_driving: List[
            Tuple[int, carla.VehicleControl, TickState]
        ] = []
        self._teacher_in_control = False
        self._intervention_step = 0

    def push_student_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        self._teacher_in_control = False
        self._recent_student_driving.append((step, control, state))
        if len(self._recent_student_driving) > 20:
            self._recent_student_driving.pop(0)

    def push_teacher_driving(
        self, step: int, control: carla.VehicleControl, state: TickState
    ) -> None:
        if not self._teacher_in_control:
            self._intervention_step = step
            self._teacher_in_control = True
            self._store_student_driving()

        rgb_filename = f"{step:05d}-rgb-teacher.bin"
        self._add_file(rgb_filename, state.rgb.tobytes(order="C"))
        orientation = state.rotation.get_forward_vector()
        self._csv.writerow(
            {
                "tick": step,
                "controller": "teacher",
                "rgb_filename": rgb_filename,
                "time_to_intervention": None,
                "time_from_intervention": step - self._intervention_step,
                "location_x": state.location.x,
                "location_y": state.location.y,
                "location_z": state.location.z,
                "velocity_x": state.velocity.x,
                "velocity_y": state.velocity.y,
                "velocity_z": state.velocity.z,
                "speed": state.speed,
                "orientation_x": orientation.x,
                "orientation_y": orientation.y,
                "orientation_z": orientation.z,
            }
        )

    def _add_file(self, filename: str, data: bytes) -> None:
        self._archive.writestr(filename, data)

    def _store_student_driving(self) -> None:
        for (step, _control, state) in reversed(self._recent_student_driving):
            rgb_filename = f"{step:05d}-rgb-student.bin"
            self._add_file(rgb_filename, state.rgb.tobytes(order="C"))
            orientation = state.rotation.get_forward_vector()
            self._csv.writerow(
                {
                    "tick": step,
                    "controller": "student",
                    "rgb_filename": rgb_filename,
                    "time_to_intervention": self._intervention_step - step,
                    "time_from_intervention": None,
                    "location_x": state.location.x,
                    "location_y": state.location.y,
                    "location_z": state.location.z,
                    "velocity_x": state.velocity.x,
                    "velocity_y": state.velocity.y,
                    "velocity_z": state.velocity.z,
                    "speed": state.speed,
                    "orientation_x": orientation.x,
                    "orientation_y": orientation.y,
                    "orientation_z": orientation.z,
                }
            )
        self._recent_student_driving = []

    def stop(self) -> None:
        self._store_student_driving()


def run_manual() -> None:
    visualizer = visualization.Visualizer()

    managed_episode = connect()
    actions = []
    with managed_episode as episode:
        for step in itertools.count():
            state = episode.tick()

            control = carla.VehicleControl()
            if visualization.Action.THROTTLE in actions:
                control.throttle = 100.0
            if visualization.Action.BRAKE in actions:
                control.brake = 100.0
            if visualization.Action.LEFT in actions:
                control.steer = -100.0
            if visualization.Action.RIGHT in actions:
                control.steer = 100.0
            episode.apply_control(control)

            birdview_render = episode.render_birdview()
            actions = visualizer.render(
                state.rgb,
                "manual",
                0.0,
                control,
                control,
                birdview_render,
            )

            if state.route_completed:
                break

def run(store: Store) -> None:
    visualizer = visualization.Visualizer()

    managed_episode = connect()
    with managed_episode as episode:
        logger.debug("Creating agents.")
        student = _prepare_student_agent()
        teacher = _prepare_teacher_agent()
        comparer = Comparer(student, teacher)

        for step in itertools.count():
            state = episode.tick()
            logger.trace("command {}", state.command)
            comparer.evaluate_and_compare(state)

            if comparer.student_in_control:
                store.push_student_driving(step, comparer.student_control, state)
                episode.apply_control(comparer.student_control)
            else:
                store.push_teacher_driving(step, comparer.teacher_control, state)
                episode.apply_control(comparer.teacher_control)

            birdview_render = episode.render_birdview()
            actions = visualizer.render(
                state.rgb,
                "student" if comparer.student_in_control else "teacher",
                comparer.difference_integral,
                comparer.student_control,
                comparer.teacher_control,
                birdview_render,
            )
            if visualization.Action.SWITCH_CONTROL in actions:
                comparer.switch_control()


def demo() -> None:
    run(BlackHoleStore())

def manual() -> None:
    run_manual()


def run_example_episode(store: Store) -> None:
    """
    param store: the store for the episode information.
    """
    visualizer = visualization.Visualizer()

    managed_episode = connect()
    with managed_episode as episode:
        logger.debug("Creating teacher agent.")
        teacher = _prepare_teacher_agent()

        for step in itertools.count():
            state = episode.tick()

            teacher_control, _ = teacher.run_step(
                {
                    "birdview": state.birdview,
                    "velocity": np.float32(
                        [state.velocity.x, state.velocity.y, state.velocity.z]  # type: ignore
                    ),
                    "command": state.command,
                },
                teaching=True,
            )

            store.push_teacher_driving(step, teacher_control, state)
            episode.apply_control(teacher_control)

            birdview_render = episode.render_birdview()
            visualizer.render(
                state.rgb,
                "teacher",
                0,
                teacher_control,
                teacher_control,
                birdview_render,
            )

            if state.probably_stuck:
                raise exceptions.EpisodeStuck()

            if state.collision:
                raise exceptions.CollisionInEpisode()

            if state.route_completed:
                break



def process_wrapper(target, *args, **kwargs):
    queue = multiprocessing.Queue()

    def _wrapper(target, queue, *args, **kwargs):
        value = target(*args, **kwargs)
        queue.put(value)

    process = multiprocessing.Process(
        target=_wrapper, args=(target, queue) + args, kwargs=kwargs
    )
    process.start()
    process.join()
    return queue.get()


def collect_example_episodes(data_path: Path, num_episodes: int) -> None:
    for episode in range(num_episodes):
        logger.info(f"Collecting episode {episode+1}/{num_episodes}.")
        episode_id = uuid.uuid4()
        episode_dir = Path(str(episode_id))
        episode_dir.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(episode_dir / "images.zip", mode="w") as zip_archive:
                with open(
                    episode_dir / "episode.csv", mode="w", newline=""
                ) as csv_file:
                    csv_writer = csv.DictWriter(
                        csv_file,
                        fieldnames=[
                            "tick",
                            "controller",
                            "rgb_filename",
                            "time_to_intervention",
                            "time_from_intervention",
                            "location_x",
                            "location_y",
                            "location_z",
                            "velocity_x",
                            "velocity_y",
                            "velocity_z",
                            "speed",
                            "orientation_x",
                            "orientation_y",
                            "orientation_z",
                        ],
                    )
                    csv_writer.writeheader()
                    store = ZipStore(zip_archive, csv_writer)
                    # Run in process to circumvent Carla bug
                    process_wrapper(run_example_episode, store)
        except (exceptions.EpisodeStuck, exceptions.CollisionInEpisode) as exception:
            logger.info(f"Removing episode because of episode exception: {exception}.")
            (episode_dir / "images.zip").unlink()
            (episode_dir / "episode.csv").unlink()
            episode_dir.rmdir()


def collect() -> None:
    with zipfile.ZipFile("episode-x.zip", mode="w") as zip_archive:
        with open("episode.csv", mode="w", newline="") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "tick",
                    "controller",
                    "rgb_filename",
                    "time_to_intervention",
                    "time_from_intervention",
                    "location_x",
                    "location_y",
                    "location_z",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                    "speed",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                ],
            )
            csv_writer.writeheader()
            store = ZipStore(zip_archive, csv_writer)
            run(store)
