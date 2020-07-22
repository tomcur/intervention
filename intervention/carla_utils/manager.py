from typing import Optional, Tuple, Dict, List, cast

import collections
from dataclasses import dataclass
import random
import queue
import numpy as np
from loguru import logger
import carla

from .agents.navigation.local_planner import LocalPlannerNew
from .images import carla_image_to_np
from .map_utils import Renderer


class EgoVehicle:
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
        self._rgb_queue: queue.Queue[np.ndarray] = queue.Queue()

    def apply_control(self, control: carla.VehicleControl) -> None:
        self.vehicle.apply_control(control)

    def current_speed_and_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        velocity = self.vehicle.get_velocity()
        return (
            np.linalg.norm([velocity.x, velocity.y, velocity.z]),  # type: ignore
            np.float32([velocity.x, velocity.y, velocity.z]),  # type: ignore
        )

    def latest_rgb(self) -> np.ndarray:
        """Blocks until an image is available."""
        rgb = None
        while rgb is None or not self._rgb_queue.empty():
            rgb = self._rgb_queue.get()
        return carla_image_to_np(rgb)

    def add_rgb_camera(self, carla_world: carla.World) -> carla.Sensor:
        def _enqueue_image(image):
            logger.trace("Received image: {}", image)
            self._rgb_queue.put(image)

        blueprints = carla_world.get_blueprint_library()
        rgb_camera_bp = blueprints.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "384")
        rgb_camera_bp.set_attribute("image_size_y", "160")
        rgb_camera_bp.set_attribute("fov", "90")
        rgb_camera = carla_world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self.vehicle,
        )
        assert isinstance(rgb_camera, carla.Sensor)
        rgb_camera.listen(_enqueue_image)

        return rgb_camera


class Episode:
    def __init__(
        self,
        carla_world: carla.World,
        ego_vehicle: EgoVehicle,
        local_planner: LocalPlannerNew,
        renderer: Renderer,
    ):
        self._carla_world = carla_world
        self._ego_vehicle = ego_vehicle
        self._local_planner = local_planner
        self._renderer = renderer

    def apply_control(self, control: carla.VehicleControl):
        """Apply control on the ego vehicle."""
        self._ego_vehicle.apply_control(control)

    def restore(self):
        """Restore to N seconds ago."""
        # TODO
        raise NotImplementedError

    def tick(self):
        self._carla_world.tick()

        self._local_planner.run_step()
        command = self._local_planner.checkpoint[1]
        # node = self._local_planner.checkpoint[0].transform.location
        # next = self._local_planner.target[0].transform.location

        # logger.trace("start {}", self._start_pose.location)
        # logger.trace("end {}", self._end_pose.location)
        # logger.trace("node {}", node)
        # logger.trace("next {}", next)

        (speed, velocity) = self._ego_vehicle.current_speed_and_velocity()
        rgb = self._ego_vehicle.latest_rgb()

        return {
            "rgb": rgb,
            "speed": speed,
            "velocity": velocity,
            "command": int(command),
            "birdview": self.get_birdview(),
        }

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
    town: str = "Town01"
    weather: carla.WeatherParameters = carla.WeatherParameters.ClearNoon
    vehicle_name: str = "vehicle.mustang.mustang"
    minimal_route_distance: int = 250

    def __init__(self, carla_client: carla.Client):
        self._client = carla_client
        self._traffic_manager: Optional[carla.TrafficManager] = None
        self._pedestrian_controllers: List[carla.WalkerAIController] = []
        self._actor_dict: Dict[str, List[carla.Actor]] = collections.defaultdict(list)

    def _set_up_world_settings(self, world: carla.World):
        logger.trace("Set simulation to synchronous mode.")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

    def _set_up_traffic_manager(self) -> int:
        logger.trace("Setting up/connecting to traffic manager.")
        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(True)
        return self._traffic_manager.get_port()

    def _generate_route(
        self, carla_map: carla.Map
    ) -> Tuple[LocalPlannerNew, carla.Transform, carla.Transform]:
        spawn_points = carla_map.get_spawn_points()

        while True:
            start_pose = np.random.choice(spawn_points)
            end_pose = np.random.choice(spawn_points)

            # FIXME
            local_planner = LocalPlannerNew(carla_map, 2.5, 9.0, 1.5)
            local_planner.set_route(start_pose.location, end_pose.location)

            if local_planner.distance_to_goal >= self.minimal_route_distance:
                return local_planner, start_pose, end_pose

    def _set_up(self) -> Episode:
        logger.trace("Loading world.")
        carla_world = self._client.load_world(self.town)
        traffic_manager_port = self._set_up_traffic_manager()

        logger.trace(f"Setting world weather to {self.weather}.")
        carla_world.set_weather(self.weather)

        self._set_up_world_settings(carla_world)

        carla_map = carla_world.get_map()

        logger.debug("Generating route.")
        (local_planner, start_pose, _) = self._generate_route(carla_map)

        logger.debug("Spawning ego vehicle.")
        ego_vehicle = self._spawn_ego_vehicle(
            carla_world, traffic_manager_port, start_pose
        )
        local_planner.set_vehicle(ego_vehicle.vehicle)

        logger.debug("Spawning vehicles.")
        self._spawn_vehicles(carla_world, carla_map, traffic_manager_port, 50)

        logger.debug("Spawning pedestrians.")
        self._spawn_pedestrians(carla_world, 125)

        for controller in self._pedestrian_controllers:
            controller.start()
            controller.go_to_location(carla_world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        renderer = Renderer(
            "placeholder", self._client, carla_world, carla_map, ego_vehicle.vehicle
        )
        renderer.start()

        return Episode(carla_world, ego_vehicle, local_planner, renderer)

    def _clean_up(self) -> None:
        # For some reason sensors cannot be destroyed in batch
        for sensor in self._actor_dict["sensor"]:
            sensor.destroy()

        for actors in self._actor_dict.values():
            self._client.apply_batch([carla.command.DestroyActor(x) for x in actors])

    def __enter__(self) -> Episode:
        logger.debug("Entering managed episode context.")
        logger.trace("Building episode.")
        # TODO start recording
        return self._set_up()

    def __exit__(self, *args):
        logger.debug("Exiting managed episode context.")
        self._clean_up()

    def _spawn_vehicles(
        self,
        carla_world: carla.World,
        carla_map: carla.Map,
        traffic_manager_port: int,
        n_vehicles: int,
    ) -> None:
        blueprints = []
        for blueprint in carla_world.get_blueprint_library().filter("vehicle.*"):
            wheels = blueprint.get_attribute("number_of_wheels")
            if wheels is not None and wheels.as_int() == 4:
                blueprints.append(blueprint)
        spawn_points = carla_map.get_spawn_points()

        # TODO: don't use same spawnpoint twice (loop through shuffled spawnpoints)
        for _ in range(n_vehicles):
            blueprint = np.random.choice(blueprints)
            blueprint.set_attribute("role_name", "autopilot")

            if blueprint.has_attribute("color"):
                color = np.random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)

            if blueprint.has_attribute("driver_id"):
                driver_id = np.random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)

            vehicle = None
            while vehicle is None:
                vehicle = carla_world.try_spawn_actor(
                    blueprint, np.random.choice(spawn_points)
                )

            vehicle.set_autopilot(True, traffic_manager_port)

            self._actor_dict["vehicle"].append(vehicle)

        logger.debug("spawned %d vehicles" % len(self._actor_dict["vehicle"]))

    def _spawn_pedestrians(self, carla_world: carla.World, n_pedestrians: int) -> None:
        walker_blueprints = carla_world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        controller_blueprint = carla_world.get_blueprint_library().find(
            "controller.ai.walker"
        )

        peds_spawned = 0

        walkers = []
        controllers = []

        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []

            for _ in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = carla_world.get_random_location_from_navigation()

                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(walker_blueprints)

                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")

                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    logger.trace(result.error)
                else:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)

            batch = [
                carla.command.SpawnActor(
                    controller_blueprint, carla.Transform(), walker
                )
                for walker in _walkers
            ]

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    logger.trace(result.error)
                else:
                    _controllers.append(result.actor_id)

            controllers.extend(_controllers)
            walkers.extend(_walkers)

        logger.debug(f"Spawned {len(controllers)} pedestrians.")
        self._actor_dict["pedestrians"] = list(carla_world.get_actors(walkers))
        self._actor_dict["pedestrian_controllers"] = list(
            carla_world.get_actors(controllers)
        )
        self._pedestrian_controllers = cast(
            List[carla.WalkerAIController], list(carla_world.get_actors(controllers))
        )

    def _spawn_ego_vehicle(
        self,
        carla_world: carla.World,
        traffic_manager_port: int,
        start_pose: carla.Transform,
    ) -> EgoVehicle:
        blueprints = carla_world.get_blueprint_library()
        blueprint = np.random.choice(blueprints.filter(self.vehicle_name))
        blueprint.set_attribute("role_name", "hero")

        if blueprint.has_attribute("color"):
            color = np.random.choice(
                blueprint.get_attribute("color").recommended_values
            )
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = np.random.choice(
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
        return ego_vehicle


def connect(carla_host: str = "localhost", carla_port: int = 2000) -> ManagedEpisode:
    logger.trace("Connecting to Carla simulator.")
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)
    return ManagedEpisode(client)
