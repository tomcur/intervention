from typing import List, Any
from dataclasses import dataclass

from .command import Command, Response


@dataclass
class Location:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance(self, location: "Location") -> float:
        ...


class Rotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class Transform:
    location: Location
    rotation: Rotation

    def __init__(self, location: Location, rotation: Rotation):
        ...

    def transform(self, in_point: Location):
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


class BlueprintLibrary:
    def filter(self, wildcard_pattern: str) -> "BlueprintLibrary":
        ...

    def find(self, id: str) -> ActorBlueprint:
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


class Actor:
    ...


class Walker:
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


class DebugHelper:
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


class Client:
    def __init__(
        self, host: str = "127.0.0.1", port: int = 2000, worker_threads: int = 0
    ):
        ...

    def apply_batch(self, commands: List[Command]):
        ...

    def apply_batch_sync(
        self, commands: List[carla.command], due_tick_cue: bool = False
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
