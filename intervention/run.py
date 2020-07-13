import abc
import numpy as np
import torch
from loguru import logger
import itertools
import zipfile
from io import BytesIO
import threading
import sys

from .carla_utils import connect, carla_image_to_np
from . import visualization

from .learning_by_cheating import image
from .learning_by_cheating import birdview
from .learning_by_cheating import roaming


from typing import Any, Optional, Tuple, Dict, List


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


def controls_difference(observation, supervisor_control, model_control) -> float:
    diff = 0

    if supervisor_control.hand_brake != model_control.hand_brake:
        diff += 1
    if supervisor_control.reverse != model_control.reverse:
        diff += 1

    diff += 0.7 * abs(supervisor_control.throttle - model_control.throttle)
    diff += 1.5 * abs(supervisor_control.steer - model_control.steer)

    if observation["speed"] > 30.0 * 1000 / 60 / 60:
        diff += abs(supervisor_control.brake - model_control.brake)
    elif observation["speed"] > 10.0 * 1000 / 60 / 60:
        diff += 0.5 * abs(supervisor_control.brake - model_control.brake)
    elif observation["speed"] > 5.0 * 1000 / 60 / 60:
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

    def evaluate_and_compare(self, state):
        self.student_control = self.student.run_step(
            {
                "rgb": state["rgb"],
                "velocity": state["velocity"],
                # "command": int(command),
                "command": state["command"],
            }
        )

        self.teacher_control, _ = self.teacher.run_step(state, teaching=True)

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

    def push_student_driving(self, step: int, control: Any, rgb: Any) -> None:
        pass

    def push_teacher_driving(self, step: int, control: Any, rgb: Any) -> None:
        pass


class ZipStore(Store):
    """Episode store backed by zip-file."""

    def __init__(self, archive: zipfile.ZipFile):
        self._archive = archive
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._recent_student_driving: List[Tuple[int, Any, Any]] = []
        self._teacher_in_control = False
        self._intervention_step = 0

    def push_student_driving(self, step: int, control: Any, rgb: Any) -> None:
        self._teacher_in_control = False
        self._recent_student_driving.append((step, control, rgb))
        if len(self._recent_student_driving) > 20:
            self._recent_student_driving.pop(0)

    def push_teacher_driving(self, step: int, control: Any, rgb: Any) -> None:
        if not self._teacher_in_control:
            self._intervention_step = step
            self._teacher_in_control = True
            self._store_student_driving()

        rgb_filename = f"{step:05d}-rgb-teacher.bin"
        self._add_file(rgb_filename, rgb.tobytes(order="C"))
        self._meta[step] = {
            "type": "teacher",
            "control": control,
            "timeFromIntervention": step - self._intervention_step,
            "rgbFile": rgb_filename,
        }

    @property
    def meta(self) -> Dict[int, Dict[str, Any]]:
        """The metadata of the episode."""
        return self._meta

    def _add_file(self, filename: str, data: bytes) -> None:
        self._archive.writestr(filename, data)

    def _store_student_driving(self) -> None:
        for (step, control, rgb) in reversed(self._recent_student_driving):
            rgb_filename = f"{step:05d}-rgb-student.bin"
            self._add_file(rgb_filename, rgb.tobytes(order="C"))
            self._meta[step] = {
                "type": "student",
                "control": control,
                "timeToIntervention": step - self._intervention_step,
                "rgbFile": rgb_filename,
            }
        self._recent_student_driving = []


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
            logger.trace("command {}", state["command"])
            comparer.evaluate_and_compare(state)

            if comparer.student_in_control:
                store.push_student_driving(step, comparer.student_control, state["rgb"])
                episode.apply_control(comparer.student_control)
            else:
                store.push_teacher_driving(step, comparer.teacher_control, state["rgb"])
                episode.apply_control(comparer.teacher_control)

            birdview_render = episode.render_birdview()
            actions = visualizer.render(
                state["rgb"],
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


def collect() -> None:
    with zipfile.ZipFile("episode-x.zip", mode="w") as zip_archive:
        store = ZipStore(zip_archive)
        run(store)
