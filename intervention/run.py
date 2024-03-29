import itertools
import time
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
import torch
from loguru import logger

from . import controller, data, pdf, process, visualization
from .carla_utils import TickState, connect
from .carla_utils.agents.navigation.local_planner import RoadOption
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


def controls_difference(
    state: TickState,
    supervisor_control,
    model_control,
) -> float:
    """
    Heuristically calculate the difference between two vehicle controls, based on
    throttle, braking, and steering commands.

    This difference can be used to calculate whether models' outputs agree with each
    other.
    """
    diff = 0

    if supervisor_control.hand_brake != model_control.hand_brake:
        diff += 1
    if supervisor_control.reverse != model_control.reverse:
        diff += 1

    if state.speed > 15.0 * 1000 / 60 / 60:
        throttle_diff_weight = 0.10
        throttle_diff_thresh = 0.40

        brake_diff_weight = 0.80
        brake_diff_thresh = 0.1

        steer_diff_weight = 3.50
        steer_diff_thresh = 0.0

        next_x_offset_weight = 0.5
        next_x_offset_thresh = 0.15
    elif state.speed > 10.0 * 1000 / 60 / 60:
        throttle_diff_weight = 0.15
        throttle_diff_thresh = 0.40

        brake_diff_weight = 0.60
        brake_diff_thresh = 0.1

        steer_diff_weight = 2.00
        steer_diff_thresh = 0.0

        next_x_offset_weight = 0.5
        next_x_offset_thresh = 0.15
    elif state.speed > 5.0 * 1000 / 60 / 60:
        throttle_diff_weight = 0.30
        throttle_diff_thresh = 0.10

        brake_diff_weight = 0.50
        brake_diff_thresh = 0.05

        steer_diff_weight = 1.0
        steer_diff_thresh = 0.0

        next_x_offset_weight = 0.4
        next_x_offset_thresh = 0.15
    else:
        throttle_diff_weight = 0.40
        throttle_diff_thresh = 0.10

        brake_diff_weight = 0.60
        brake_diff_thresh = 0.20

        steer_diff_weight = 0.5
        steer_diff_thresh = 0.0

        next_x_offset_weight = 0.0
        next_x_offset_thresh = 0.0

    throttle_diff = abs(supervisor_control.throttle - model_control.throttle)
    brake_diff = abs(supervisor_control.brake - model_control.brake)
    steer_diff = abs(supervisor_control.steer - model_control.steer)
    next_x_offset = abs(supervisor_target_waypoints[0, 0])

    diff += throttle_diff_weight * max(throttle_diff - throttle_diff_thresh, 0.0)
    diff += brake_diff_weight * max(brake_diff - brake_diff_thresh, 0.0)
    diff += steer_diff_weight * max(steer_diff - steer_diff_thresh, 0.0)

    # We see the next X offset as a measure of how much the student is drifting out of
    # lane according to the teacher.
    diff += next_x_offset_weight * max(next_x_offset - next_x_offset_thresh, 0.0)

    return diff


def waypoints_difference(
    state: TickState,
    supervisor_target_waypoints,
    model_target_waypoints,
) -> float:
    """
    Heuristically calculate the difference between two vehicle controls, based on target
    waypoints.

    This difference can be used to calculate whether models' outputs agree with each
    other.
    """
    error_vectors = model_target_waypoints - supervisor_target_waypoints
    errors = np.linalg.norm(error_vectors, axis=1)

    # Completely ignore very small differences
    errors = np.maximum(errors - 0.15, 0.0)

    # Ignore small differences
    threshold = 0.30
    errors[errors < threshold] = 0.0

    relative_errors = (
        2.0
        * errors
        / np.maximum(
            np.linalg.norm(model_target_waypoints, axis=1)
            + np.linalg.norm(supervisor_target_waypoints, axis=1),
            1.0,
        )
    )
    # print(model_target_waypoints[1])
    # print(supervisor_target_waypoints[1])
    # print(error_vectors[1])
    # print(errors[1])
    # print(relative_errors[1])

    total_diff = relative_errors[1:2].mean()
    return total_diff


def control_stable(state: TickState, control) -> bool:
    """
    Determines whether a control is "stable". When a control is stable for a while,
    the controller is assumed to be comfortable handing back control to the model.
    """
    return (
        state.speed > 0.5
        and not control.reverse
        and not control.hand_brake
        # and control.throttle > 0.01
        and control.brake < 0.2
        and abs(control.steer) < 0.3
    )


class Comparer:
    def __init__(self, threshold: float = 6.0, only_student_driving: bool = False):
        self.threshold = threshold
        self.only_student_driving = only_student_driving
        self.difference_integral = 0.0

        self._stable_frames = 0
        self.student_in_control = True

    def evaluate_and_compare(
        self,
        state: TickState,
        teacher_target_waypoints: np.ndarray,
        teacher_control: carla.VehicleControl,
        student_target_waypoints: np.ndarray,
        student_control: carla.VehicleControl,
    ) -> None:
        if self.student_in_control:
            self.difference_integral *= 0.92
            # self.difference_integral += controls_difference(
            #     state,
            #     teacher_control,
            #     student_control,
            # )
            self.difference_integral += waypoints_difference(
                state,
                teacher_target_waypoints,
                student_target_waypoints,
            )
            if (
                self.difference_integral >= self.threshold
                and not self.only_student_driving
            ):
                logger.trace("switching to teacher control")
                self.student_in_control = False
                self.difference_integral = 0
        else:
            if control_stable(state, teacher_control):
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


def benchmark() -> None:
    visualizer = visualization.Visualizer()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    with managed_episode as episode:
        start_time = datetime.now()
        for step in itertools.count():
            state = episode.tick()

            control = carla.VehicleControl(throttle=0.2)
            episode.apply_control(control)

            current_time = datetime.now()
            seconds = (current_time - start_time).total_seconds()

            logger.info(f"average tps {step / seconds:.2f} @ Tick {step}")

            if state.route_completed:
                break


def run_manual() -> None:
    visualizer = visualization.Visualizer()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    with managed_episode as episode:
        for step in itertools.count():
            state = episode.tick()

            actions = visualizer.get_actions()
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


def run_student_episode(
    store: data.Store,
    episode_dir: Path,
    student_checkpoint_path: Path,
    print_to_pdf: bool,
) -> data.EpisodeSummary:
    """
    Run an episode of on-policy student driving (without teacher supervision).

    :param store: the store for the episode information.
    :param episode_dir: a directory where additional episode data can be stored.
    """
    from .models.image import Agent, Image

    visualizer = visualization.Visualizer()

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    managed_episode.town = process.rng.choice(process.towns)
    managed_episode.weather = process.rng.choice(process.weathers)

    managed_episode.attach_high_resolution_rgb_camera = print_to_pdf
    printing_to_pdf: bool = print_to_pdf

    summary = data.EpisodeSummary.from_managed_episode(managed_episode)
    with managed_episode as episode:
        summary.route_length = episode.route_length

        vehicle_geometry = episode.get_vehicle_geometry()
        vehicle_controller = controller.VehicleController(vehicle_geometry)

        logger.debug("Creating student agent.")
        student_model = Image().to(process.torch_device)
        student_agent = Agent(student_model)
        student_checkpoint = torch.load(
            student_checkpoint_path, map_location=process.torch_device
        )
        student_model.load_state_dict(student_checkpoint["model_state_dict"])
        student_model.eval()

        start_time = datetime.now()

        for step in itertools.count():
            actions = visualizer.get_actions()
            if visualization.Action.TOGGLE_PRINT_TO_PDF in actions:
                printing_to_pdf = (not printing_to_pdf) and print_to_pdf

            state = episode.tick()
            summary.ticks += 1

            distance_delta = state.distance_travelled - summary.distance_travelled
            summary.student_distance_travelled += distance_delta
            summary.distance_travelled = state.distance_travelled
            summary.route_length_completed = state.distance_travelled_along_route

            (
                student_target_waypoints,
                _student_target_heatmap,
                model_image_targets,
                model_image_heatmaps,
            ) = student_agent.step(state)
            student_control, student_turn_radius = vehicle_controller.step(
                state,
                student_target_waypoints,
            )
            episode.apply_control(student_control)
            store.push_student_driving(
                step,
                None,
                student_target_waypoints,
                model_image_targets,
                model_image_heatmaps,
                student_control,
                state,
            )

            if printing_to_pdf:
                pdf.print_multipage(
                    episode_dir / f"test-{summary.ticks}.pdf",
                    summary.ticks,
                    state.high_resolution_rgb,
                    model_image_heatmaps,
                )

            with visualizer as painter:
                painter.add_command(state.command)
                painter.add_rgb(state.rgb, outline_color=(255, 145, 0))
                painter.add_waypoints(
                    student_target_waypoints,
                    color=(255, 145, 0),
                )
                painter.add_turn_radius(
                    vehicle_geometry,
                    student_turn_radius,
                    "LEFT" if student_control.steer < 0 else "RIGHT",
                    color=(255, 145, 0),
                )
                painter.add_control(
                    "student",
                    student_control,
                )

                birdview_render = episode.render_birdview()
                painter.add_birdview(birdview_render)
            current_time = datetime.now()
            seconds = (current_time - start_time).total_seconds()
            logger.debug(f"average tps {step / seconds:.2f} @ Tick {step}")

            if state.probably_stuck:
                summary.end_status = "stuck"
                break

            if state.probably_off_course:
                summary.end_status = "off_course"
                break

            if state.collision:
                summary.collisions += 1
                summary.end_status = "collision"
                break

            if state.route_completed:
                summary.end_status = "success"
                break

    summary.end()
    return summary


def demo_teacher_agent(teacher_checkpoint: Path, user_input_planner: bool) -> None:
    """
    :param user_input_planner: whether to use user input for the route planner.
    """
    run_teacher_episode(
        data.BlackHoleStore(), teacher_checkpoint, user_input_planner=user_input_planner
    )


def manual() -> None:
    run_manual()


def explore_off_policy_dataset(episode_path: Path) -> None:
    from . import coordinates
    from .train import dataset

    visualizer = visualization.Visualizer(
        event_processor=visualization.dataset_explorer_event_processor
    )
    data = dataset.off_policy_data(episode_path)

    idx = 0
    rendered = False
    auto_next = False
    auto_next_time = 0.0
    while True:
        _transformed_image, image, meta = data[idx]

        next_waypoints = []
        for [image_x, image_y] in meta["next_locations_image_coordinates"]:
            next_waypoints.append(
                coordinates.image_coordinate_to_ego_coordinate(image_x, image_y)
            )

        if not rendered:
            rendered = True
            with visualizer as painter:
                painter.add_rgb(np.moveaxis(image, [0], [2]))
                painter.add_waypoints(next_waypoints)

        actions = visualizer.get_actions()
        if visualization.Action.NEXT in actions:
            idx += 1
            auto_next = False
            rendered = False
        elif visualization.Action.PREVIOUS in actions:
            idx -= 1
            auto_next = False
            rendered = False
        elif visualization.Action.PLAY in actions:
            auto_next = not auto_next
            auto_next_time = time.time()

        now = time.time()
        if auto_next and now - auto_next_time > 0.1:
            auto_next_time = now
            idx += 1
            rendered = False


def explore_on_policy_dataset(episode_path: Path) -> None:
    from . import coordinates
    from .train import dataset

    visualizer = visualization.Visualizer(
        event_processor=visualization.dataset_explorer_event_processor
    )
    data = dataset.intervention_data(episode_path).combined

    idx = 0
    rendered = False
    auto_next = False
    auto_next_time = 0.0
    while True:

        (data_type, data_point) = data[idx]

        if data_type in [
            dataset.DataType.NEGATIVE,
            dataset.DataType.SUPERVISION_SIGNAL,
        ]:
            (
                _transformed_image,
                image,
                _student_model_targets,
                _student_model_heatmaps,
                meta,
            ) = data_point
        elif data_type is dataset.DataType.IMITATION:
            _transformed_image, image, meta = data_point

        next_waypoints = []
        for [image_x, image_y] in meta["next_locations_image_coordinates"]:
            next_waypoints.append(
                coordinates.image_coordinate_to_ego_coordinate(image_x, image_y)
            )

        if not rendered:
            rendered = True
            with visualizer as painter:
                painter.add_rgb(np.moveaxis(image, [0], [2]))
                painter.add_annotation(
                    [
                        f"Episode:         x",
                        f"Tick:            #{meta['tick']}",
                        f"Speed:          {meta['speed'] / 1000 * 60 * 60:3.0f} km/h",
                        f"Controller:      {meta['controller']}",
                        f"Ticks to inter.: {meta['ticks_to_intervention']}",
                    ]
                )
                painter.add_waypoints(next_waypoints)

        actions = visualizer.get_actions()
        if visualization.Action.NEXT in actions:
            idx += 1
            auto_next = False
            rendered = False
        elif visualization.Action.PREVIOUS in actions:
            idx -= 1
            auto_next = False
            rendered = False
        elif visualization.Action.PLAY in actions:
            auto_next = not auto_next
            auto_next_time = time.time()

        now = time.time()
        if auto_next and now - auto_next_time > 0.1:
            auto_next_time = now
            idx += 1
            rendered = False


def run_teacher_episode(
    store: data.Store,
    teacher_checkpoint: Path,
    user_input_planner: bool = False,
) -> data.EpisodeSummary:
    """
    :param store: the store for the episode information.
    :param teacher_checkpoint: the checkpoint to load for the teacher.
    :param user_input_planner: whether to use user input for the route planner.
    """
    visualizer = visualization.Visualizer(
        event_processor=visualization.drive_command_event_processor,
    )

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    managed_episode.town = process.rng.choice(process.towns)
    managed_episode.weather = process.rng.choice(process.weathers)

    summary = data.EpisodeSummary.from_managed_episode(managed_episode)
    with managed_episode as episode:
        summary.route_length = episode.route_length

        vehicle_geometry = episode.get_vehicle_geometry()
        vehicle_controller = controller.VehicleController(vehicle_geometry)

        logger.debug("Creating teacher agent.")
        teacher = _prepare_teacher_agent(teacher_checkpoint)

        for step in itertools.count():
            state = episode.tick()

            if user_input_planner:
                state.command = RoadOption.LANEFOLLOW
                actions = visualizer.get_actions()
                if visualization.Action.GO_LEFT in actions:
                    state.command = RoadOption.LEFT
                elif visualization.Action.GO_RIGHT in actions:
                    state.command = RoadOption.RIGHT
                elif visualization.Action.GO_STRAIGHT in actions:
                    state.command = RoadOption.STRAIGHT

            summary.ticks += 1

            distance_delta = state.distance_travelled - summary.distance_travelled
            summary.teacher_distance_travelled += distance_delta
            summary.distance_travelled = state.distance_travelled
            summary.route_length_completed = state.distance_travelled_along_route

            teacher_target_waypoints = teacher.run_step(
                {
                    "birdview": state.birdview,
                    "velocity": np.float32(
                        [  # type: ignore
                            state.velocity.x,
                            state.velocity.y,
                            state.velocity.z,
                        ]
                    ),
                    "command": int(state.command),
                },
                teaching=True,
            )
            teacher_control, teacher_turn_radius = vehicle_controller.step(
                state,
                teacher_target_waypoints,
            )

            store.push_teacher_driving(
                step,
                teacher_target_waypoints,
                None,
                teacher_control,
                state,
            )
            episode.apply_control(teacher_control)

            birdview_render = episode.render_birdview()
            with visualizer as painter:
                painter.add_command(state.command)
                painter.add_rgb(state.rgb, outline_color=(0, 145, 255))
                painter.add_control("teacher", teacher_control)
                painter.add_waypoints(teacher_target_waypoints)
                painter.add_turn_radius(
                    vehicle_geometry,
                    teacher_turn_radius,
                    "LEFT" if teacher_control.steer < 0 else "RIGHT",
                    color=(0, 145, 255),
                )
                painter.add_birdview(birdview_render)

            if state.probably_stuck:
                summary.end_status = "stuck"
                break

            if state.probably_off_course:
                summary.end_status = "off_course"
                break

            if state.collision:
                summary.collisions += 1
                summary.end_status = "collision"
                break

            if state.route_completed:
                summary.end_status = "success"
                break

    summary.end()
    return summary


def run_intervention_episode(
    store: data.Store,
    student_checkpoint_path: Path,
    teacher_checkpoint_path: Path,
    only_student_driving: bool,
) -> data.EpisodeSummary:
    """
    Run an episode of on-policy student driving with teacher supervision and
    intervention.

    param store: the store for the episode information.
    """
    from .models.image import Agent, Image

    visualizer = visualization.Visualizer()
    comparer = Comparer(only_student_driving=only_student_driving)

    managed_episode = connect(
        carla_host=process.carla_host, carla_world_port=process.carla_world_port
    )
    managed_episode.town = process.rng.choice(process.towns)
    managed_episode.weather = process.rng.choice(process.weathers)

    summary = data.EpisodeSummary.from_managed_episode(managed_episode)
    with managed_episode as episode:
        summary.route_length = episode.route_length
        summary.average_prediction_l1_error = 0.0
        summary.average_prediction_l2_error = 0.0

        vehicle_geometry = episode.get_vehicle_geometry()
        vehicle_controller = controller.VehicleController(vehicle_geometry)

        logger.debug("Creating student agent.")
        student_model = Image().to(process.torch_device)
        student_agent = Agent(student_model)
        student_checkpoint = torch.load(
            student_checkpoint_path, map_location=process.torch_device
        )
        student_model.load_state_dict(student_checkpoint["model_state_dict"])
        student_model.eval()

        logger.debug("Creating teacher agent.")
        teacher = _prepare_teacher_agent(teacher_checkpoint_path)

        start_time = datetime.now()

        for step in itertools.count():
            state = episode.tick()
            summary.ticks += 1

            distance_delta = state.distance_travelled - summary.distance_travelled
            if comparer.student_in_control:
                summary.student_distance_travelled += distance_delta
            else:
                summary.teacher_distance_travelled += distance_delta
            summary.distance_travelled = state.distance_travelled
            summary.route_length_completed = state.distance_travelled_along_route

            teacher_target_waypoints = teacher.run_step(
                {
                    "birdview": state.birdview,
                    "velocity": np.float32(
                        [  # type: ignore
                            state.velocity.x,
                            state.velocity.y,
                            state.velocity.z,
                        ]
                    ),
                    "command": int(state.command),
                },
                teaching=True,
            )
            teacher_control, teacher_turn_radius = vehicle_controller.step(
                state,
                teacher_target_waypoints,
                update_pids=not comparer.student_in_control,
            )
            (
                student_target_waypoints,
                _student_target_heatmap,
                model_image_targets,
                model_image_heatmaps,
            ) = student_agent.step(state)

            prediction_l1_error = np.linalg.norm(
                student_target_waypoints - teacher_target_waypoints, ord=1, axis=1
            ).mean()
            prediction_l2_error = np.linalg.norm(
                student_target_waypoints - teacher_target_waypoints, ord=2, axis=1
            ).mean()

            summary.average_prediction_l1_error += (
                prediction_l1_error - summary.average_prediction_l1_error
            ) / summary.ticks
            summary.average_prediction_l2_error += (
                prediction_l2_error - summary.average_prediction_l2_error
            ) / summary.ticks

            student_in_control = comparer.student_in_control
            if not student_in_control:
                episode.apply_control(teacher_control)
                store.push_teacher_driving(
                    step,
                    teacher_target_waypoints,
                    student_target_waypoints,
                    teacher_control,
                    state,
                    prediction_l1_error,
                    prediction_l2_error,
                    heuristic_control_difference=comparer.difference_integral,
                )

            student_control, student_turn_radius = vehicle_controller.step(
                state,
                student_target_waypoints,
                update_pids=comparer.student_in_control,
            )
            if student_in_control:
                episode.apply_control(student_control)
                store.push_student_driving(
                    step,
                    teacher_target_waypoints,
                    student_target_waypoints,
                    model_image_targets,
                    model_image_heatmaps,
                    student_control,
                    state,
                    prediction_l1_error,
                    prediction_l2_error,
                    heuristic_control_difference=comparer.difference_integral,
                )

            comparer.evaluate_and_compare(
                state,
                teacher_target_waypoints,
                teacher_control,
                student_target_waypoints,
                student_control,
            )

            if student_in_control and not comparer.student_in_control:
                # Control was switched from student to teacher
                summary.interventions += 1

            with visualizer as painter:
                painter.add_command(state.command)
                painter.add_rgb(
                    state.rgb,
                    outline_color=(255, 145, 0)
                    if comparer.student_in_control
                    else (0, 145, 255),
                )
                painter.add_waypoints(
                    teacher_target_waypoints,
                    color=(0, 145, 255),
                    grayout=comparer.student_in_control,
                )
                painter.add_turn_radius(
                    vehicle_geometry,
                    teacher_turn_radius,
                    "LEFT" if teacher_control.steer < 0 else "RIGHT",
                    color=(0, 145, 255),
                    grayout=comparer.student_in_control,
                )
                painter.add_waypoints(
                    student_target_waypoints,
                    color=(255, 145, 0),
                    grayout=not comparer.student_in_control,
                )
                painter.add_turn_radius(
                    vehicle_geometry,
                    student_turn_radius,
                    "LEFT" if student_control.steer < 0 else "RIGHT",
                    color=(255, 145, 0),
                    grayout=not comparer.student_in_control,
                )
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
            current_time = datetime.now()
            seconds = (current_time - start_time).total_seconds()
            logger.debug(f"average tps {step / seconds:.2f} @ Tick {step}")

            if state.probably_stuck:
                summary.end_status = "stuck"
                break

            if state.probably_off_course:
                summary.end_status = "off_course"
                break

            if state.collision:
                summary.collisions += 1
                summary.end_status = "collision"
                break

            if state.route_completed:
                summary.end_status = "success"
                break

    summary.end()
    return summary
