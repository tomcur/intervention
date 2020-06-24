import collections
import queue
import numpy as np
from loguru import logger

import carla

from .agents.navigation.local_planner import RoadOption, LocalPlannerNew
from .images import carla_image_to_np
from .map_utils import Renderer

VEHICLE_NAME = "vehicle.mustang.mustang"


class Manager:
    def __init__(
        self, town="Town01", vehicle_name=VEHICLE_NAME, port=2000, start=0, end=40
    ):
        logger.debug("Creating manager.")

        self._client = carla.Client("localhost", port)

        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(True)
        self._tm_port = self._traffic_manager.get_port()

        self._player = None
        self._rgb_queue = queue.Queue()

        self._client.set_timeout(30.0)

        self._world = self._client.load_world(town)
        self._world.set_weather(carla.WeatherParameters.ClearNoon)
        # self._world.set_weather(carla.WeatherParameters.Default)
        self._map = self._world.get_map()

        self._start_pose = self._map.get_spawn_points()[start]
        self._end_pose = self._map.get_spawn_points()[end]

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1

        self._world.apply_settings(settings)

        self._blueprints = self._world.get_blueprint_library()
        self._vehicle_bp = np.random.choice(self._blueprints.filter(vehicle_name))
        self._vehicle_bp.set_attribute("role_name", "hero")

        if self._vehicle_bp.has_attribute('color'):
            color = np.random.choice(self._vehicle_bp.get_attribute('color').recommended_values)
            self._vehicle_bp.set_attribute('color', color)
        if self._vehicle_bp.has_attribute('driver_id'):
            driver_id = np.random.choice(self._vehicle_bp.get_attribute('driver_id').recommended_values)
            self._vehicle_bp.set_attribute('driver_id', driver_id)
        if self._vehicle_bp.has_attribute('is_invincible'):
            self._vehicle_bp.set_attribute('is_invincible', 'true')

        self._actor_dict = collections.defaultdict(list)

        self._renderer = None

    def _spawn_player(self):
        self._player = self._world.spawn_actor(self._vehicle_bp, self._start_pose)
        self._player.set_autopilot(False, self._tm_port)
        self._actor_dict["player"].append(self._player)

        def _enqueue_image(image):
            logger.trace("Received image: {}", image)
            self._rgb_queue.put(image)

        rgb_camera_bp = self._blueprints.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "384")
        rgb_camera_bp.set_attribute("image_size_y", "160")
        rgb_camera_bp.set_attribute("fov", "90")
        rgb_camera = self._world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player,
        )
        rgb_camera.listen(_enqueue_image)
        self._actor_dict["sensor"].append(rgb_camera)

        def rss_response(event):
            print(event)

        rss_sensor_bp = self._blueprints.find("sensor.other.rss")
        rss_sensor = self._world.spawn_actor(
            rss_sensor_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self._player,
        )
        rss_sensor.listen(rss_response)
        rss_sensor.road_boundaries_mode = carla.RssRoadBoundariesMode.On
        rss_sensor.visualization_mode = carla.RssVisualizationMode.All
        rss_sensor.visualize_results = True

        # default, from ad_rss documentation
        rss_sensor.ego_vehicle_dynamics.alphaLon.accelMax = 3.5
        rss_sensor.ego_vehicle_dynamics.alphaLon.brakeMin = -4
        rss_sensor.ego_vehicle_dynamics.alphaLon.brakeMax = -8
        rss_sensor.ego_vehicle_dynamics.alphaLon.brakeMinCorrect = -3
        rss_sensor.ego_vehicle_dynamics.alphaLat.brakeMin = -0.8
        rss_sensor.ego_vehicle_dynamics.alphaLat.accelMax = 0.2
        rss_sensor.ego_vehicle_dynamics.lateralFluctuationMargin = 0.1
        rss_sensor.ego_vehicle_dynamics.responseTime = 1.0
        rss_sensor.ego_vehicle_dynamics.maxSpeed = 100

        rss_sensor.reset_routing_targets()
        self._actor_dict["sensor"].append(rss_sensor)

        # FIXME
        self._local_planner = LocalPlannerNew(self._player, 2.5, 9.0, 1.5)
        self._local_planner.set_route(
            self._start_pose.location, self._end_pose.location
        )

    def _clean_up(self):
        for sensor in self._actor_dict["sensor"]:
            sensor.destroy()

        for actors in self._actor_dict.values():
            self._client.apply_batch([carla.command.DestroyActor(x) for x in actors])

    def __enter__(self):
        logger.debug("Entering manager.")
        return self

        # TODO start recording

    def __exit__(self, *args):
        logger.debug("Exiting manager.")
        self._clean_up()
        pass

        # TODO stop recording

    def setup(self):
        logger.debug("Running manager setup.")
        logger.debug("Spawning player.")
        self._spawn_player()
        logger.debug("Spawning vehicles.")
        self._spawn_vehicles(50)

        logger.debug("Spawning pedestrians.")
        pedestrians, pedestrian_controllers = self._spawn_pedestrians(125)
        self._actor_dict["pedestrian"].extend(pedestrians)
        self._actor_dict["ped_controller"].extend(pedestrian_controllers)

        # FIXME run this periodically
        import random

        for controller in self._actor_dict["ped_controller"]:
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        self._renderer = Renderer(
            "placeholder", self._client, self._world, self._map, self._player
        )
        self._renderer.start()

    def apply_control(self, player_control):
        """Apply control on the player vehicle."""
        self._player.apply_control(player_control)

    def restore(self):
        """Restore to N seconds ago."""
        # FIXME
        pass

    def tick(self):
        self._world.tick()

        self._local_planner.run_step()
        command = self._local_planner.checkpoint[1]
        node = self._local_planner.checkpoint[0].transform.location
        next = self._local_planner.target[0].transform.location

        # logger.trace("start {}", self._start_pose.location)
        # logger.trace("end {}", self._end_pose.location)
        # logger.trace("node {}", node)
        # logger.trace("next {}", next)

        velocity = self._player.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        velocity = np.float32([velocity.x, velocity.y, velocity.z])

        # Blocks until an image is available.
        rgb = None
        while rgb is None or not self._rgb_queue.empty():
            rgb = self._rgb_queue.get()
            rgb = carla_image_to_np(rgb)

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

    def _spawn_vehicles(self, n_vehicles):
        blueprints = self._blueprints.filter("vehicle.*")
        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        spawn_points = self._map.get_spawn_points()

        # TODO: don't use same spawnpoint twice (loop through shuffled spawnpoints)
        for i in range(n_vehicles):
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
                vehicle = self._world.try_spawn_actor(
                    blueprint, np.random.choice(spawn_points)
                )

            vehicle.set_autopilot(True, self._tm_port)

            self._actor_dict["vehicle"].append(vehicle)

        logger.debug("spawned %d vehicles" % len(self._actor_dict["vehicle"]))

    def _spawn_pedestrians(self, n_pedestrians):
        import random

        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0

        walkers = []
        controllers = []

        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []

            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()

                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            blueprints = self._blueprints.filter("walker.pedestrian.*")
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)

                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")

                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    logger.trace(result.error)
                else:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)

            walker_controller_bp = self._blueprints.find("controller.ai.walker")
            batch = [
                SpawnActor(walker_controller_bp, carla.Transform(), walker)
                for walker in _walkers
            ]

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    logger.trace(result.error)
                else:
                    _controllers.append(result.actor_id)

            controllers.extend(_controllers)
            walkers.extend(_walkers)

        logger.debug("spawned %d pedestrians" % len(controllers))

        return self._world.get_actors(walkers), self._world.get_actors(controllers)
