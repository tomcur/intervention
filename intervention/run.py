from typing import Any, Optional, Tuple, Dict, List, TextIO

import multiprocessing

import os
import abc
import itertools
import zipfile
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import uuid
import csv
import numpy as np
import torch
from loguru import logger
import carla

from .carla_utils import connect, TickState
from . import visualization, exceptions, controller
from . import data
from . import process

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
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.difference_integral = 0.0

        self._stable_frames = 0
        self.student_in_control = False

    def evaluate_and_compare(
        self,
        state: TickState,
        teacher_control: carla.VehicleControl,
        student_control: carla.VehicleControl,
    ) -> None:
        if self.student_in_control:
            self.difference_integral *= 0.9
            self.difference_integral += controls_difference(
                state, teacher_control, student_control
            )
            if self.difference_integral >= self.threshold:
                logger.trace("switching to teacher control")
                self.student_in_control = False
                self.difference_integral = 0
        else:
            if control_stable(teacher_control):
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


def _prepare_teacher_agent(teacher_checkpoint: Path):
    teacher_model = birdview.BirdViewPolicyModelSS(backbone="resnet18")
    teacher_model.load_state_dict(
        torch.load(teacher_checkpoint, map_location=process.torch_device)
    )
    teacher_model.eval()
    teacher_model.to(process.torch_device)

    teacher_agent_args = {
        # "steer_points": {"1": 4, "2": 3, "3": 2, "4": 2},
        "model": teacher_model,
    }
    teacher_agent = birdview.BirdViewAgent(**teacher_agent_args)

    return teacher_agent


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

        self._csv_file = csv_file
        self._csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "tick",
                "command",
                "controller",
                "rgb_filename",
                "student_output_filename",
                "time_to_intervention",
                "time_from_intervention",
                "lane_invasion",
                "collision",
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
            self._store_student_driving()

        rgb_filename = f"{step:05d}-rgb-teacher.bin"
        self._add_file(rgb_filename, state.rgb.tobytes(order="C"))
        orientation = state.rotation.get_forward_vector()
        self._csv_writer.writerow(
            {
                "tick": step,
                "command": state.command,
                "controller": "teacher",
                "rgb_filename": rgb_filename,
                "time_to_intervention": None,
                "time_from_intervention": step - self._intervention_step,
                "lane_invasion": int(state.lane_invasion is not None),
                "collision": int(state.collision is not None),
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
        for (step, model_output, _control, state) in reversed(
            self._recent_student_driving
        ):
            rgb_filename = f"{step:05d}-rgb-student.bin"
            self._add_file(rgb_filename, state.rgb.tobytes(order="C"))

            model_output_filename = f"{step:05d}-output-student.npy"
            buffer = BytesIO()
            np.save(buffer, model_output)
            self._add_file(model_output_filename, buffer.getvalue())

            orientation = state.rotation.get_forward_vector()
            self._csv_writer.writerow(
                {
                    "tick": step,
                    "command": state.command,
                    "controller": "student",
                    "rgb_filename": rgb_filename,
                    "student_output_filename": model_output_filename,
                    "time_to_intervention": self._intervention_step - step,
                    "time_from_intervention": None,
                    "lane_invasion": int(state.lane_invasion is not None),
                    "collision": int(state.collision is not None),
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

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
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
            with visualizer as painter:
                painter.add_rgb(state.rgb)
                painter.add_control("manual", control)
                painter.add_birdview(birdview_render)

            if state.route_completed:
                break


def run_image_agent(store: Store) -> None:
    """
    param store: the store for the episode information.
    """
    from .models.image import Image, Agent

    visualizer = visualization.Visualizer()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    with managed_episode as episode:
        logger.debug("Creating agent.")
        model = Image()
        agent = Agent(model)
        vehicle_controller = controller.VehicleController()
        checkpoint = torch.load("../checkpoints-intervention/2020-10-03/24.pth")
        model.load_state_dict(checkpoint["model_state_dict"])

        for step in itertools.count():
            state = episode.tick()

            logger.trace("command {}", state.command)
            logger.trace("distance travelled {}", state.distance_travelled)

            target_waypoints, _network_output, target_heatmap = agent.step(state)
            control = vehicle_controller.step(state, target_waypoints)

            episode.apply_control(control)

            birdview_render = episode.render_birdview()

            with visualizer as painter:
                painter.add_rgb(state.rgb)
                painter.add_waypoints(target_waypoints)
                painter.add_control("student", control)
                painter.add_birdview(birdview_render)

            if state.probably_stuck:
                raise exceptions.EpisodeStuck()

            if state.collision:
                raise exceptions.CollisionInEpisode()

            if state.route_completed:
                break

            # prev_state = state


def demo_image_agent() -> None:
    run_image_agent(BlackHoleStore())


def manual() -> None:
    run_manual()


def explore_off_policy_dataset(episode_path: Path) -> None:
    from .train import dataset
    from . import coordinates

    visualizer = visualization.Visualizer()
    data = dataset.off_policy_data(episode_path)

    idx = 0
    while True:

        _transformed_image, image, meta = data[idx]

        next_waypoints = []
        for [image_x, image_y] in meta["next_locations_image_coordinates"]:
            next_waypoints.append(
                coordinates.image_coordinate_to_ego_coordinate(image_x, image_y)
            )

        with visualizer as painter:
            painter.add_rgb(image)
            painter.add_waypoints(next_waypoints)

        actions = visualizer.get_actions()
        if visualization.Action.NEXT in actions:
            idx += 1
        elif visualization.Action.PREVIOUS in actions:
            idx -= 1


def run_example_episode(store: Store, teacher_checkpoint: Path) -> data.EpisodeSummary:
    """
    param store: the store for the episode information.
    """
    town = process.rng.choice(["Town01", "Town02", "Town07"])

    visualizer = visualization.Visualizer()
    summary = data.EpisodeSummary(town=town)
    vehicle_controller = controller.VehicleController()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    with managed_episode as episode:
        managed_episode.town = town

        logger.debug("Creating teacher agent.")
        teacher = _prepare_teacher_agent(teacher_checkpoint)

        for step in itertools.count():
            state = episode.tick()
            summary.distance_travelled = state.distance_travelled
            summary.ticks += 1

            teacher_target_waypoints = teacher.run_step(
                {
                    "birdview": state.birdview,
                    "velocity": np.float32(
                        [state.velocity.x, state.velocity.y, state.velocity.z]  # type: ignore
                    ),
                    "command": state.command,
                },
                teaching=True,
            )
            teacher_control = vehicle_controller.step(state, teacher_target_waypoints,)

            store.push_teacher_driving(step, teacher_control, state)
            episode.apply_control(teacher_control)

            birdview_render = episode.render_birdview()
            with visualizer as painter:
                painter.add_rgb(state.rgb)
                painter.add_control("teacher", teacher_control)
                painter.add_waypoints(teacher_target_waypoints)
                painter.add_birdview(birdview_render)

            if state.probably_stuck:
                summary.terminated = True
                break

            if state.collision:
                summary.collisions += 1
                summary.terminated = True
                break

            if state.route_completed:
                summary.success = True
                break

    summary.end()
    return summary


def run_on_policy_episode(
    store: Store, student_checkpoint: Path, teacher_checkpoint: Path
) -> data.EpisodeSummary:
    """
    param store: the store for the episode information.
    """
    from .models.image import Image, Agent

    visualizer = visualization.Visualizer()
    summary = data.EpisodeSummary()
    comparer = Comparer()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    with managed_episode as episode:
        vehicle_controller = controller.VehicleController()

        logger.debug("Creating student agent.")
        student_model = Image()
        student_agent = Agent(student_model)
        student_checkpoint = torch.load(student_checkpoint)
        student_model.load_state_dict(student_checkpoint["model_state_dict"])

        logger.debug("Creating teacher agent.")
        teacher = _prepare_teacher_agent(teacher_checkpoint)

        for step in itertools.count():
            state = episode.tick()
            summary.distance_travelled = state.distance_travelled
            summary.ticks += 1

            teacher_target_waypoints = teacher.run_step(
                {
                    "birdview": state.birdview,
                    "velocity": np.float32(
                        [state.velocity.x, state.velocity.y, state.velocity.z]  # type: ignore
                    ),
                    "command": state.command,
                },
                teaching=True,
            )
            teacher_control = vehicle_controller.step(
                state,
                teacher_target_waypoints,
                update_pids=not comparer.student_in_control,
            )
            if not comparer.student_in_control:
                episode.apply_control(teacher_control)
                store.push_teacher_driving(step, teacher_control, state)

            (
                student_target_waypoints,
                model_output,
                _student_target_heatmap,
            ) = student_agent.step(state)
            student_control = vehicle_controller.step(
                state,
                student_target_waypoints,
                update_pids=comparer.student_in_control,
            )
            if comparer.student_in_control:
                episode.apply_control(student_control)
                store.push_student_driving(step, model_output, student_control, state)

            comparer.evaluate_and_compare(state, teacher_control, student_control)

            with visualizer as painter:
                painter.add_rgb(state.rgb)
                painter.add_waypoints(teacher_target_waypoints, color=(0, 145, 255))
                painter.add_waypoints(student_target_waypoints, color=(255, 145, 0))
                painter.add_control(
                    "student", student_control, grayout=not comparer.student_in_control
                )
                painter.add_control(
                    "teacher", teacher_control, grayout=comparer.student_in_control
                )
                painter.add_control_difference(
                    comparer.difference_integral, threshold=comparer.threshold
                )

                birdview_render = episode.render_birdview()
                painter.add_birdview(birdview_render)

            if state.probably_stuck:
                summary.terminated = True
                break

            if state.collision:
                summary.collisions += 1
                summary.terminated = True
                break

            if state.route_completed:
                summary.success = True
                break

    summary.end()
    return summary


def process_wrapper(target, *args, **kwargs):
    """
    Runs target (with *args and **kwargs) in a subprocess. Blocks until target is done.
    Returns the return value of target.
    """
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


def collect_example_episode(
    teacher_checkpoint: Path, episode_dir: Path, seed_sequence: np.random.SeedSequence,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = ZipStore(zip_archive, csv_file)
            summary = run_example_episode(store, teacher_checkpoint)
            store.stop()
            return summary


def collect_example_episodes(
    teacher_checkpoint: Path, data_path: Path, num_episodes: int,
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(f"Collecting episode {episode+1}/{num_episodes}: {episode_id}.")
            episode_dir = data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = process_wrapper(
                collect_example_episode, teacher_checkpoint, episode_dir, seed_sequence,
            )
            episode_summary.uuid = episode_id
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())

            # try:
            # except (exceptions.EpisodeStuck, exceptions.CollisionInEpisode) as exception:
            #     logger.info(f"Removing episode because of episode exception: {exception}.")
            #     (episode_dir / "images.zip").unlink()
            #     (episode_dir / "episode.csv").unlink()
            #     episode_dir.rmdir()


def collect_on_policy_episode(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    episode_dir: Path,
    seed_sequence: np.random.SeedSequence,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = ZipStore(zip_archive, csv_file)
            summary = run_on_policy_episode(
                store, student_checkpoint, teacher_checkpoint
            )
            store.stop()
            return summary


def collect_on_policy_episodes(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    data_path: Path,
    num_episodes: int,
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(f"Collecting episode {episode+1}/{num_episodes}: {episode_id}.")
            episode_dir = data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = process_wrapper(
                collect_on_policy_episode,
                student_checkpoint,
                teacher_checkpoint,
                episode_dir,
                seed_sequence,
            )
            episode_summary.uuid = episode_id
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())
