import os
import collections
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
from loguru import logger
from typing_extensions import Literal

import carla

from .. import physics, process
from ..utils.carla_image import carla_image_to_np
from .agents.navigation.local_planner import LocalPlannerNew, RoadOption
from .map_utils import Renderer

#: The number of ticks without movement after which we consider the vehicle to be stuck
STUCK_TICKS: int = 90 * 10

#: The distance in meters to the next checkpoint at which we consider the vehicle to be
#: off course.
OFF_COURSE_CHECKPOINT_DISTANCE: float = 15.0

CarlaWeather = Literal[
    "Default",
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "SoftRainNoon",
    "MidRainyNoon",
    "HardRainNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "SoftRainSunset",
    "MidRainSunset",
    "HardRainSunset",
]

CarlaTown = Literal[
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
]


CARLA_WEATHERS = CarlaWeather.__args__  # type: ignore
CARLA_TOWNS = CarlaTown.__args__  # type: ignore


@dataclass
class TickState:
    location: carla.Location
    rotation: carla.Rotation
    distance_travelled: float
    distance_to_next_checkpoint: float
    distance_to_goal: float
    distance_travelled_along_route: float
    rgb: np.ndarray
    high_resolution_rgb: Optional[np.ndarray]
    lane_invasion: Optional[carla.LaneInvasionEvent]
    collision: Optional[carla.CollisionEvent]
    speed: float
    velocity: carla.Vector3D
    command: RoadOption
    birdview: np.ndarray
    route_completed: bool
    probably_stuck: bool
    probably_off_course: bool


class EgoVehicle:
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle: carla.Vehicle = vehicle
        self._rgb_queues: Dict[carla.Sensor, queue.Queue[np.ndarray]] = {}
        self._lane_invasion_queue: queue.Queue[carla.LaneInvasionEvent] = queue.Queue()
        self._collision_queue: queue.Queue[carla.CollisionEvent] = queue.Queue()

    def apply_control(self, control: carla.VehicleControl) -> None:
        self.vehicle.apply_control(control)

    def current_speed_and_velocity(self) -> Tuple[float, carla.Vector3D]:
        velocity = self.vehicle.get_velocity()
        return (np.linalg.norm([velocity.x, velocity.y, velocity.z]), velocity)

    def current_location(self) -> carla.Location:
        return self.vehicle.get_transform().location

    def current_rotation(self) -> carla.Rotation:
        return self.vehicle.get_transform().rotation

    def latest_rgb(self, camera_handle: carla.Sensor) -> np.ndarray:
        """
        Blocks until an image is available for the given camera handle.

        :param camera_handle: MUST be a carla `Sensor` object, as returned by
        `EgoVehicle.add_rgb_camera`.
        """
        rgb = None
        while rgb is None or not self._rgb_queues[camera_handle].empty():
            rgb = self._rgb_queues[camera_handle].get()
        return carla_image_to_np(rgb)

    def latest_lane_invasion(self) -> Optional[carla.LaneInvasionEvent]:
        event = None
        while not self._lane_invasion_queue.empty():
            event = self._lane_invasion_queue.get()
        return event

    def latest_collision(self) -> Optional[carla.CollisionEvent]:
        event = None
        while not self._collision_queue.empty():
            event = self._collision_queue.get()
        return event

    def add_rgb_camera(
        self,
        carla_world: carla.World,
        image_size_x: int = 384,
        image_size_y: int = 160,
        effects: bool = True,
    ) -> carla.Sensor:
        """
        :param effects: if false, disables some camera effects
        """

        def _create_listener(rgb_camera: carla.Sensor):
            def _enqueue_image(image):
                logger.trace("Received image: {}", image)
                self._rgb_queues[rgb_camera].put(image)

            return _enqueue_image

        blueprints = carla_world.get_blueprint_library()
        rgb_camera_bp = blueprints.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", f"{image_size_x}")
        rgb_camera_bp.set_attribute("image_size_y", f"{image_size_y}")
        rgb_camera_bp.set_attribute("fov", "90")
        if not effects:
            rgb_camera_bp.set_attribute("motion_blur_intensity", "0.0")
        rgb_camera = carla_world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self.vehicle,
        )
        assert isinstance(rgb_camera, carla.Sensor)

        self._rgb_queues[rgb_camera] = queue.Queue()
        rgb_camera.listen(_create_listener(rgb_camera))

        return rgb_camera

    def add_lane_invasion_detector(self, carla_world: carla.World) -> carla.Sensor:
        def _lane_invasion(event: carla.LaneInvasionEvent) -> None:
            logger.warning("Lane invasion: {}", event)
            self._lane_invasion_queue.put(event)

        blueprints = carla_world.get_blueprint_library()
        lane_invasion_detector_bp = blueprints.find("sensor.other.lane_invasion")
        lane_invasion_detector = carla_world.spawn_actor(
            lane_invasion_detector_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        assert isinstance(lane_invasion_detector, carla.Sensor)
        lane_invasion_detector.listen(_lane_invasion)  # type: ignore

        return lane_invasion_detector

    def add_collision_detector(self, carla_world: carla.World) -> carla.Sensor:
        def _collision(event: carla.CollisionEvent) -> None:
            logger.warning("Collision: {}", event)
            self._collision_queue.put(event)

        blueprints = carla_world.get_blueprint_library()
        collision_detector_bp = blueprints.find("sensor.other.collision")
        collision_detector = carla_world.spawn_actor(
            collision_detector_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        assert isinstance(collision_detector, carla.Sensor)
        collision_detector.listen(_collision)  # type: ignore

        return collision_detector

    def get_vehicle_geometry(self) -> physics.VehicleGeometry:
        vehicle_location = self.vehicle.get_location()
        vehicle_physics = self.vehicle.get_physics_control()

        wheel_front_left = vehicle_physics.wheels[0]
        wheel_front_right = vehicle_physics.wheels[1]
        wheel_rear_left = vehicle_physics.wheels[2]
        wheel_rear_right = vehicle_physics.wheels[3]

        vehicle_pos = np.array(
            [vehicle_location.x, vehicle_location.y, vehicle_location.z]
        )

        # The physics positions are world coordinates in centimeters, we convert to
        # meters.
        wheel_front_left_pos = (
            np.array(
                [
                    wheel_front_left.position.x,
                    wheel_front_left.position.y,
                    wheel_front_left.position.z,
                ]
            )
            / 100.0
        )
        wheel_front_right_pos = (
            np.array(
                [
                    wheel_front_right.position.x,
                    wheel_front_right.position.y,
                    wheel_front_right.position.z,
                ]
            )
            / 100.0
        )
        wheel_rear_left_pos = (
            np.array(
                [
                    wheel_rear_left.position.x,
                    wheel_rear_left.position.y,
                    wheel_rear_left.position.z,
                ]
            )
            / 100.0
        )
        wheel_rear_right_pos = (
            np.array(
                [
                    wheel_rear_right.position.x,
                    wheel_rear_right.position.y,
                    wheel_rear_right.position.z,
                ]
            )
            / 100.0
        )

        # Calculate the points determining the longitudinal axis
        front = (wheel_front_left_pos + wheel_front_right_pos) / 2.0
        rear = (wheel_rear_left_pos + wheel_rear_right_pos) / 2.0

        # Calculate offsets from the rear
        rear_vehicle_offset = vehicle_pos - rear
        rear_front_offset = front - rear

        # Project the vehicle world location onto the longitudinal axis
        # (the vehicle might be rotated somewhat during spawning)
        vehicle_pos_projected_onto_axis = (
            rear
            + np.dot(rear_vehicle_offset, rear_front_offset)
            / np.dot(rear_front_offset, rear_front_offset)
            * rear_front_offset
        )

        wheel_base = np.linalg.norm(rear_front_offset)
        wheel_track = np.linalg.norm(wheel_front_left_pos - wheel_front_right_pos)

        rear_axle_longitudinal_offset = np.linalg.norm(
            vehicle_pos_projected_onto_axis - rear
        )

        max_inner_wheel_angle = np.radians(wheel_front_left.max_steer_angle)

        return physics.VehicleGeometry(
            wheel_base=float(wheel_base),
            wheel_track=float(wheel_track),
            max_inner_wheel_angle=float(max_inner_wheel_angle),
            rear_axle_longitudinal_offset=rear_axle_longitudinal_offset,
        )


class Episode:
    def __init__(
        self,
        carla_world: carla.World,
        start_location: carla.Location,
        ego_vehicle: EgoVehicle,
        rgb_camera: carla.Sensor,
        local_planner: LocalPlannerNew,
        renderer: Renderer,
        high_resolution_rgb_camera: Optional[carla.Sensor] = None,
    ):
        self._carla_world: carla.World = carla_world
        self._location: carla.Location = start_location
        self._ego_vehicle: EgoVehicle = ego_vehicle
        self._rgb_camera: carla.Sensor = rgb_camera
        self._high_res_rgb_camera: Optional[carla.Sensor] = high_resolution_rgb_camera
        self._local_planner: LocalPlannerNew = local_planner
        self._renderer: Renderer = renderer
        self._route_completed: bool = False
        self._unmoved_ticks: int = 0
        self._distance_travelled: float = 0.0
        self.route_length: float = local_planner.distance_to_goal

    def get_vehicle_geometry(self) -> physics.VehicleGeometry:
        return self._ego_vehicle.get_vehicle_geometry()

    def apply_control(self, control: carla.VehicleControl):
        """Apply control on the ego vehicle."""
        self._ego_vehicle.apply_control(control)

    def restore(self):
        """Restore to N seconds ago."""
        # TODO
        raise NotImplementedError

    def tick(self) -> TickState:
        self._carla_world.tick()

        self._local_planner.run_step()
        if self._local_planner.is_done():
            self._route_completed = True

        assert self._local_planner.checkpoint[0] is not None
        checkpoint_location = self._local_planner.checkpoint[0].transform.location

        assert self._local_planner.checkpoint[1] is not None
        command = self._local_planner.checkpoint[1]

        # node = self._local_planner.checkpoint[0].transform.location
        # next = self._local_planner.target[0].transform.location

        # logger.trace("start {}", self._start_pose.location)
        # logger.trace("end {}", self._end_pose.location)
        # logger.trace("node {}", node)
        # logger.trace("next {}", next)

        (speed, velocity) = self._ego_vehicle.current_speed_and_velocity()
        location = self._ego_vehicle.current_location()
        rotation = self._ego_vehicle.current_rotation()
        rgb = self._ego_vehicle.latest_rgb(self._rgb_camera)
        high_resolution_rgb = (
            None
            if self._high_res_rgb_camera is None
            else self._ego_vehicle.latest_rgb(self._high_res_rgb_camera)
        )
        lane_invasion = self._ego_vehicle.latest_lane_invasion()
        collision = self._ego_vehicle.latest_collision()

        distance_to_next_checkpoint = location.distance(checkpoint_location)

        self._distance_travelled += self._location.distance(location)
        self._location = location

        if speed > 0.0001:
            self._unmoved_ticks = 0
        else:
            self._unmoved_ticks += 1

        return TickState(
            location=location,
            rotation=rotation,
            distance_travelled=self._distance_travelled,
            distance_to_next_checkpoint=distance_to_next_checkpoint,
            distance_to_goal=self._local_planner.distance_to_goal,
            distance_travelled_along_route=(
                self.route_length - self._local_planner.distance_to_goal
            ),
            rgb=rgb,
            high_resolution_rgb=high_resolution_rgb,
            lane_invasion=lane_invasion,
            collision=collision,
            speed=speed,
            velocity=velocity,
            command=command,
            birdview=self.get_birdview(),
            route_completed=self._route_completed,
            probably_stuck=self._unmoved_ticks > STUCK_TICKS,
            probably_off_course=distance_to_next_checkpoint
            > OFF_COURSE_CHECKPOINT_DISTANCE,
        )

    def render_birdview(self):
        return self._renderer.get_render()

    def get_birdview(self):
        self._renderer.update()
        self._renderer.render()
        result = self._renderer.get_result()

        birdview = [
            result["road"],
            result["lane"],
            result["traffic"],
            result["vehicle"],
            result["pedestrian"],
        ]
        birdview = [x if x.ndim == 3 else x[..., None] for x in birdview]
        birdview = np.concatenate(birdview, 2)
        return birdview


@dataclass
class ManagedEpisode:
    """
    This class is an Episode context manager. It is used to configure and set up CARLA
    episodes.

    Configure the desired episode by changing the properties of this class. For example,
    to change the default town and attach a high resolution camera, run:

    ```
    managed_episode = ManagedEpisode(carla_client)
    managed_episode.town = "Town02"
    managed_episode.attach_high_resolution_rgb_camera = True

    with managed_episode as episode:
        while True:
            state = episode.tick()
            # ...
            episode.apply_control(...)
    ```
    """

    town: CarlaTown = "Town01"
    weather: CarlaWeather = "Default"
    vehicle_name: str = "vehicle.mustang.mustang"
    target_route_length: float = 500.0
    attach_high_resolution_rgb_camera: bool = False

    def __init__(self, carla_client: carla.Client):
        self._client = carla_client
        self._carla_world: Optional[carla.World] = None
        self._traffic_manager: Optional[carla.TrafficManager] = None
        self._traffic_manager_port: Optional[int] = None
        self._sensors: List[carla.Sensor] = []
        self._pedestrian_controllers: List[carla.WalkerAIController] = []
        self._actor_dict: Dict[str, List[carla.Actor]] = collections.defaultdict(list)

    def _set_up_world_settings(self, world: carla.World):
        logger.trace("Set simulation to synchronous mode.")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

    def _set_up_traffic_manager(self) -> int:
        if "CARLA_TRAFFIC_MANAGER_PORT" in os.environ:
            port = int(os.environ["CARLA_TRAFFIC_MANAGER_PORT"])
        else:
            port = 8000
        logger.trace(f"Setting up/connecting to traffic manager on port {port}.")
        self._traffic_manager = self._client.get_trafficmanager(port)
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_hybrid_physics_mode(True)
        self._traffic_manager_port = port
        return port

    def _generate_route(
        self, carla_map: carla.Map
    ) -> Tuple[LocalPlannerNew, carla.Transform, carla.Transform]:
        spawn_points = carla_map.get_spawn_points()

        while True:
            # We generate a hopeful start and end pose, but the end pose might be
            # somewhere else (we try to generate a route of length
            # `self.target_route_length`).
            start_pose = process.rng.choice(spawn_points)
            end_pose = process.rng.choice(spawn_points)

            local_planner = LocalPlannerNew(carla_map, 2.5, 9.0, 1.5)
            local_planner.set_route(
                start_pose.location,
                end_pose.location,
                max_length=self.target_route_length,
            )

            # ... and we generate a different route if the distance is too short.
            if local_planner.distance_to_goal >= self.target_route_length - 2.0:
                logger.debug(
                    f"Generated route length: {local_planner.distance_to_goal:.2f} m."
                )
                return local_planner, start_pose, end_pose
            else:
                logger.trace(
                    "Generated route length too short: "
                    f"{local_planner.distance_to_goal:.2f} m. Regenerating."
                )

    def _set_up(self) -> Episode:
        logger.trace("Loading world.")
        self._carla_world = self._client.load_world(self.town)

        traffic_manager_port = self._set_up_traffic_manager()

        weather = getattr(carla.WeatherParameters, self.weather)
        logger.trace(f"Setting world weather to {weather}.")
        self._carla_world.set_weather(weather)

        self._set_up_world_settings(self._carla_world)

        carla_map = self._carla_world.get_map()

        logger.debug("Generating route.")
        (local_planner, start_pose, _) = self._generate_route(carla_map)

        logger.debug("Spawning ego vehicle.")
        ego_vehicle, rgb_camera, high_resolution_rgb_camera = self._spawn_ego_vehicle(
            self._carla_world, traffic_manager_port, start_pose
        )
        local_planner.set_vehicle(ego_vehicle.vehicle)

        logger.debug("Spawning vehicles.")
        self._spawn_vehicles(
            self._carla_world,
            carla_map,
            100,
            [start_pose.location],
        )

        logger.debug("Spawning pedestrians.")
        self._spawn_pedestrians(self._carla_world, 250)

        for controller in self._pedestrian_controllers:
            controller.start()
            controller.go_to_location(
                self._carla_world.get_random_location_from_navigation()
            )
            # Set walking speed between 2 and 5 km/h
            controller.set_max_speed(process.rng.uniform(2.0, 5.0) * 1000 / 60 / 60)

        renderer = Renderer(
            "placeholder",
            self._client,
            self._carla_world,
            carla_map,
            ego_vehicle.vehicle,
        )
        renderer.start()

        return Episode(
            self._carla_world,
            start_pose.location,
            ego_vehicle,
            rgb_camera,
            local_planner,
            renderer,
            high_resolution_rgb_camera=high_resolution_rgb_camera,
        )

    def _destroy_actors(self) -> None:
        logger.debug("Destroying actors.")

        # In addition to being destroyed, sensors and pedestrians must be told to stop.
        for sensor in self._sensors:
            sensor.stop()

        for controller in self._pedestrian_controllers:
            controller.stop()

        actors = [actor for actors in self._actor_dict.values() for actor in actors]
        self._client.apply_batch_sync(
            [carla.command.DestroyActor(actor.id) for actor in actors]
        )
        self._actor_dict = {}
        self._sensors = []
        self._pedestrian_controllers = []

        logger.debug("Destroyed actors.")

    def _clean_up(self) -> None:
        assert self._carla_world is not None
        assert self._traffic_manager is not None

        self._destroy_actors()

        self._traffic_manager.set_synchronous_mode(False)

        settings = self._carla_world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        self._carla_world.apply_settings(settings)
        self._carla_world.wait_for_tick()

        logger.info("Cleanup done.")

    def __enter__(self) -> Episode:
        logger.debug("Entering managed episode context.")
        logger.trace("Building episode.")
        # TODO start recording
        return self._set_up()

    def __exit__(self, *args):
        logger.debug("Exiting managed episode context.")
        self._clean_up()

        # Most likely unnecessary deletes, but there have been issues with cleanup
        del self._carla_world
        del self._traffic_manager
        del self._client

    def _spawn_vehicles(
        self,
        carla_world: carla.World,
        carla_map: carla.Map,
        n_vehicles: int,
        disallowed_spawn_points: List[carla.Location] = [],
    ) -> None:
        """
        Spawns `n_vehicles` vehicles.
        """
        blueprints = []
        for blueprint in carla_world.get_blueprint_library().filter("vehicle.*"):
            wheels = blueprint.get_attribute("number_of_wheels")
            if wheels is not None and wheels.as_int() == 4:
                blueprints.append(blueprint)

        spawn_points = carla_map.get_spawn_points()
        spawn_points = [
            spawn_point
            for spawn_point in spawn_points
            if spawn_point not in disallowed_spawn_points
        ]

        process.rng.shuffle(spawn_points)

        if n_vehicles > len(spawn_points):
            logger.warning(
                "Requested spawning {} vehicles, but only {} spawn points are available.",
                n_vehicles,
                len(spawn_points),
            )
            n_vehicles = len(spawn_points)

        spawn_points = spawn_points[:n_vehicles]
        batch = []
        for spawn_point in spawn_points[:n_vehicles]:
            blueprint = process.rng.choice(blueprints)
            blueprint.set_attribute("role_name", "autopilot")

            if blueprint.has_attribute("color"):
                color_attribute = blueprint.get_attribute("color")
                assert color_attribute is not None
                color = process.rng.choice(color_attribute.recommended_values)
                blueprint.set_attribute("color", color)

            if blueprint.has_attribute("driver_id"):
                driver_id_attribute = blueprint.get_attribute("driver_id")
                assert driver_id_attribute is not None
                driver_id = process.rng.choice(driver_id_attribute.recommended_values)
                blueprint.set_attribute("driver_id", driver_id)

            batch.append(carla.command.SpawnActor(blueprint, spawn_point))

        spawned = []
        for result in self._client.apply_batch_sync(batch, True):
            if result.error:
                logger.warning(result.error)
            else:
                spawned.append(result.actor_id)

        vehicles = carla_world.get_actors(spawned)
        for vehicle in vehicles:
            assert isinstance(vehicle, carla.Vehicle)
            assert self._traffic_manager_port is not None
            vehicle.set_autopilot(True, self._traffic_manager_port)

        self._actor_dict["vehicle"] = list(vehicles)
        logger.debug("Spawned {} vehicles.", len(spawned))

    def _spawn_pedestrians(self, carla_world: carla.World, n_pedestrians: int) -> None:
        """
        Spawns `n_pedestrians` pedestrians. Sometimes spawning a pedestrian fails due to
        a collision at the spawn location. This method keeps retrying in a loop until
        exactly `n_pedestrians` have been spawned.
        """
        walker_blueprints = carla_world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        controller_blueprint = carla_world.get_blueprint_library().find(
            "controller.ai.walker"
        )

        walkers: List[int] = []
        controllers: List[int] = []
        spawn_collisions = 0

        while len(walkers) < n_pedestrians:
            walker_batch = []
            for _ in range(n_pedestrians - len(walkers)):
                location = carla_world.get_random_location_from_navigation()
                spawn_point = carla.Transform(location=location)
                walker_bp = process.rng.choice(walker_blueprints)
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                walker_batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(walker_batch, True):
                if result.error:
                    if "collision at spawn position" in result.error:
                        spawn_collisions += 1
                    else:
                        logger.error(result.error)
                else:
                    walkers.append(result.actor_id)

        controller_batch = [
            carla.command.SpawnActor(controller_blueprint, carla.Transform(), walker)
            for walker in walkers
        ]
        for result in self._client.apply_batch_sync(controller_batch, True):
            if result.error:
                logger.error(result.error)
            else:
                controllers.append(result.actor_id)

        logger.debug(
            f"Spawned {len(controllers)} pedestrians, after accounting for "
            f"{spawn_collisions} spawn collisions."
        )
        self._actor_dict["pedestrian"] = list(carla_world.get_actors(walkers))
        self._actor_dict["pedestrian_controller"] = list(
            carla_world.get_actors(controllers)
        )
        self._pedestrian_controllers = cast(
            List[carla.WalkerAIController], self._actor_dict["pedestrian_controller"]
        )

    def _spawn_ego_vehicle(
        self,
        carla_world: carla.World,
        traffic_manager_port: int,
        start_pose: carla.Transform,
    ) -> Tuple[EgoVehicle, carla.Sensor, Optional[carla.Sensor]]:
        """
        Returns a 3-tuple of the created vehicle, an attached RGB camera, and
        (optionally) an attached high-resolution RGB camera.
        """
        blueprints = carla_world.get_blueprint_library()
        blueprint = process.rng.choice(blueprints.filter(self.vehicle_name))
        blueprint.set_attribute("role_name", "hero")

        if blueprint.has_attribute("color"):
            color = process.rng.choice(
                blueprint.get_attribute("color").recommended_values
            )
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = process.rng.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")

        player = carla_world.spawn_actor(blueprint, start_pose)
        assert isinstance(player, carla.Vehicle)
        player.set_autopilot(False, traffic_manager_port)
        self._actor_dict["player"].append(player)

        ego_vehicle = EgoVehicle(player)

        rgb_camera = ego_vehicle.add_rgb_camera(carla_world)
        self._actor_dict["sensor"].append(rgb_camera)
        self._sensors.append(rgb_camera)

        high_resolution_rgb_camera: Optional[carla.Sensor] = None
        if self.attach_high_resolution_rgb_camera:
            high_resolution_rgb_camera = ego_vehicle.add_rgb_camera(
                carla_world, image_size_x=1920, image_size_y=800, effects=False
            )
            self._actor_dict["sensor"].append(high_resolution_rgb_camera)
            self._sensors.append(high_resolution_rgb_camera)

        lane_invasion_detector = ego_vehicle.add_lane_invasion_detector(carla_world)
        self._actor_dict["sensor"].append(lane_invasion_detector)
        self._sensors.append(lane_invasion_detector)

        collision_detector = ego_vehicle.add_collision_detector(carla_world)
        self._actor_dict["sensor"].append(collision_detector)
        self._sensors.append(collision_detector)

        return ego_vehicle, rgb_camera, high_resolution_rgb_camera


def connect(
    carla_host: str = "localhost", carla_world_port: int = 2000
) -> ManagedEpisode:
    logger.trace(f"Connecting to Carla simulator at {carla_host}:{carla_world_port}.")
    client = carla.Client(carla_host, carla_world_port)
    client.set_timeout(30.0)
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    logger.info(f"Carla client version: {client_version}.")
    logger.info(f"Carla server version: {server_version}.")
    return ManagedEpisode(client)
