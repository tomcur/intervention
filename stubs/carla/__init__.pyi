from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union, overload

from .command import Command, Response


@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Location:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance(self, location: "Location") -> float:
        ...


@dataclass
class Rotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def get_forward_vector(self) -> Vector3D:
        ...


class Transform:
    location: Location
    rotation: Rotation

    def __init__(
        self, location: Location = Location(), rotation: Rotation = Rotation()
    ):
        ...

    def transform(self, in_point: Location):
        ...


@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int


class ActorAttributeType:
    Bool: Any
    Int: Any
    Float: Any
    String: Any
    RGBColor: Any


class ActorAttribute:
    id: str
    is_modifiable: bool
    recommended_values: List[str]
    type: ActorAttributeType

    @property
    def recommended_values(self) -> List[str]:
        ...

    def as_bool(self) -> bool:
        ...

    def as_color(self) -> Color:
        ...

    def as_float(self) -> float:
        ...

    def as_int(self) -> int:
        ...

    def as_str(self) -> str:
        ...

    def __eq__(self, other: Union[bool, int, float, str, "ActorAttribute"]):
        ...


class ActorBlueprint:
    id: str
    tags: List[str]

    def has_attribute(self, id: str) -> bool:
        ...

    def has_tag(self, tag: str) -> bool:
        ...

    def match_tags(self, wildcard_pattern: str) -> bool:
        ...

    def get_attribute(self, id: str) -> Optional[ActorAttribute]:
        ...

    def set_attribute(self, id: str, value: str) -> None:
        ...


class BlueprintLibrary(Sequence[ActorBlueprint]):
    def filter(self, wildcard_pattern: str) -> "BlueprintLibrary":
        ...

    def find(self, id: str) -> ActorBlueprint:
        ...

    @overload
    def __getitem__(self, key: int) -> ActorBlueprint:
        ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[ActorBlueprint]:
        ...

    def __len__(self) -> int:
        ...


class Actor:
    def destroy(self) -> bool:
        ...

    def get_velocity(self) -> Vector3D:
        ...

    def get_transform(self) -> Transform:
        ...

    def set_transform(self, transform: Transform) -> None:
        ...

    def set_velocity(self, velocity: Vector3D) -> None:
        ...


class ActorList(Sequence[Actor]):
    def filter(self, wildcard_pattern: str) -> List[Actor]:
        ...

    @overload
    def __getitem__(self, key: int) -> Actor:
        ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[Actor]:
        ...

    def __len__(self) -> int:
        ...


class SensorData:
    ...


class ColorConverter:
    CityScapesPalette: Any
    Depth: Any
    LogarithmicDepth: Any
    Raw: Any


class Image(SensorData):
    fov: float
    height: int
    width: int
    raw_data: bytes

    def convert(self, color_converter: ColorConverter):
        ...

    def save_to_disk(
        self, path: str, color_converter: ColorConverter = ColorConverter.Raw
    ):
        ...


class LaneMarking:
    ...


class LaneInvasionEvent(SensorData):
    actor: Actor
    crossed_lane_markings: List[LaneMarking]


class CollisionEvent(SensorData):
    actor: Actor
    other_actor: Actor
    normal_impulse: Vector3D


class Sensor(Actor):
    def listen(self, callback: Callable[[SensorData], None]):
        ...

    def stop(self):
        ...

    @property
    def is_listening(self) -> bool:
        ...


class Walker(Actor):
    ...


class WalkerAIController(Actor):
    def go_to_location(self, destination: Location) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def set_max_speed(self, speed: float = 1.4) -> None:
        ...


@dataclass
class VehicleControl:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    manual_gear_shift: bool = False
    gear: int = 0


class Vehicle(Actor):
    def apply_control(self, control: VehicleControl):
        ...

    def set_autopilot(self, enabled: bool = True, port: int = 8000) -> None:
        ...


class TrafficLightState:
    Red: Any
    Yellow: Any
    Green: Any
    Off: Any
    Unknown: Any


class TrafficManager:
    def auto_lange_change(self, actor: Actor, enable: bool):
        ...

    def collision_detection(
        self, reference_actor: Actor, other_actor: Actor, detect_collision: bool
    ):
        ...

    def distance_to_leading_vehicle(self, actor: Actor, distance: float):
        ...

    def force_lange_change(self, actor: Actor, direction: bool):
        ...

    def global_distance_to_leading_vehicle(self, distance: float):
        ...

    def global_percentage_speed_difference(self, percentage: float):
        ...

    def ignore_lights_percentage(self, actor: Actor, perc: float):
        ...

    def ignore_vehicles_percentage(self, actor: Actor, perc: float):
        ...

    def ignore_walkers_percentage(self, actor: Actor, perc: float):
        ...

    def reset_traffic_lights(self):
        ...

    def vehicle_percentage_speed_difference(self, actor: Actor, percentage: float):
        ...

    def get_port(self) -> int:
        ...

    def set_hybrid_physics_mode(self, enable: bool = False):
        ...

    def set_hybrid_mode_radius(self, r: float = 70.0):
        ...

    def set_synchronous_mode(self, mode: bool) -> None:
        ...


class AttachmentType:
    Rigid: Any
    SpringArm: Any


class DebugHelper:
    ...


@dataclass
class WeatherParameters:
    ClearNoon: "WeatherParameters"
    CloudyNoon: "WeatherParameters"
    WetNoon: "WeatherParameters"
    WetCloudyNoon: "WeatherParameters"
    SoftRainNoon: "WeatherParameters"
    MidRainyNoon: "WeatherParameters"
    HardRainNoon: "WeatherParameters"
    ClearSunset: "WeatherParameters"
    CloudySunset: "WeatherParameters"
    WetSunset: "WeatherParameters"
    WetCloudySunset: "WeatherParameters"
    SoftRainSunset: "WeatherParameters"
    MidRainSunset: "WeatherParameters"
    HardRainSunset: "WeatherParameters"

    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    fog_distance: float
    wetness: float
    fog_falloff: float


class Map:
    def get_spawn_points(self) -> List[Transform]:
        ...

    @property
    def name(self) -> str:
        ...


@dataclass
class Timestamp:
    frame: int
    elapsed_seconds: float
    delta_seconds: float
    platform_timestamp: float


class ActorSnapshot:
    id: int

    def get_acceleration(self) -> Vector3D:
        ...

    def get_angular_velocity(self) -> Vector3D:
        ...

    def get_transform(self) -> Transform:
        ...

    def get_velocity(self) -> Vector3D:
        ...


class WorldSnapshot:
    id: int
    frame: int
    timestamp: Timestamp

    def find(self, actor_id: int) -> ActorSnapshot:
        ...

    def has_actor(self, actor_id: int) -> bool:
        ...


@dataclass
class WorldSettings:
    synchronous_mode: bool = False
    no_rendering_mode: bool = False
    fixed_delta_seconds: float = 0.0


class World:
    id: int
    debug: DebugHelper

    def apply_settings(self, world_settings: WorldSettings):
        ...

    def spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = None,
        attachment: AttachmentType = AttachmentType.Rigid,
    ) -> Actor:
        ...

    def try_spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = None,
        attachment: AttachmentType = AttachmentType.Rigid,
    ) -> Optional[Actor]:
        ...

    def tick(self) -> int:
        ...

    def wait_for_tick(self, seconds: float = 10.0) -> WorldSnapshot:
        ...

    def get_actor(self, actor_id: int) -> Optional[Actor]:
        ...

    def get_actors(self, actor_ids: List[int]) -> ActorList:
        ...

    def get_blueprint_library(self) -> BlueprintLibrary:
        ...

    def get_map(self) -> Map:
        ...

    def get_random_location_from_navigation(self) -> Location:
        ...

    def get_settings(self) -> WorldSettings:
        ...

    def get_weather(self) -> WeatherParameters:
        ...

    def set_weather(self, weather: WeatherParameters) -> None:
        ...


class Client:
    def __init__(
        self, host: str = "127.0.0.1", port: int = 2000, worker_threads: int = 0
    ):
        ...

    def apply_batch(self, commands: Sequence[Command]):
        ...

    def apply_batch_sync(
        self, commands: Sequence[Command], due_tick_cue: bool = False
    ) -> List[Response]:
        ...

    def load_world(self, map_Name: str) -> World:
        ...

    def reload_world(self) -> World:
        ...

    def get_available_maps(self) -> List[str]:
        ...

    def get_client_version(self) -> str:
        ...

    def get_server_version(self) -> str:
        ...

    def get_trafficmanager(self, client_connection: int = 8000) -> TrafficManager:
        ...

    def get_world(self) -> World:
        ...

    def set_replayer_time_factor(self, time_factor: float = 1.0):
        ...

    def set_timeout(self, seconds: float):
        ...
