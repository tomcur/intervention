# Source: https://github.com/carla-simulator/carla


import os
import datetime
import weakref
import math

import numpy as np

import carla

from carla import TrafficLightState as tls

import pygame

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_TRAFFIC_RED = pygame.Color(255, 0, 0)
COLOR_TRAFFIC_YELLOW = pygame.Color(0, 255, 0)
COLOR_TRAFFIC_GREEN = pygame.Color(0, 0, 255)

# Module Defines
MODULE_WORLD = "WORLD"
MODULE_HUD = "HUD"
MODULE_INPUT = "INPUT"

PIXELS_PER_METER = 5

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 100


# ==============================================================================
# -- TrafficLightSurfaces ------------------------------------------------------
# ==============================================================================


class TrafficLightSurfaces:
    """Holds the surfaces (scaled and rotated) for painting traffic lights"""

    def __init__(self):
        def make_surface(tl):
            w = 40
            surface = pygame.Surface((w, 3 * w)).convert_alpha()
            surface.fill(COLOR_ALUMINIUM_5 if tl != "h" else COLOR_ORANGE_2)
            if tl != "h":
                hw = int(w / 2)
                off = COLOR_ALUMINIUM_4
                red = COLOR_SCARLET_RED_0
                yellow = COLOR_BUTTER_0
                green = COLOR_CHAMELEON_0
                pygame.draw.circle(
                    surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w)
                )
                pygame.draw.circle(
                    surface,
                    yellow if tl == tls.Yellow else off,
                    (hw, w + hw),
                    int(0.4 * w),
                )
                pygame.draw.circle(
                    surface,
                    green if tl == tls.Green else off,
                    (hw, 2 * w + hw),
                    int(0.4 * w),
                )
            return pygame.transform.smoothscale(
                surface, (15, 45) if tl != "h" else (19, 49)
            )

        self._original_surfaces = {
            "h": make_surface("h"),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown),
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)


class MapImage:
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = (
            max(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            + margin
        )
        max_y = (
            max(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            + margin
        )
        min_x = (
            min(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            - margin
        )
        min_y = (
            min(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            - margin
        )

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface(
            (width_in_pixels, width_in_pixels)
        ).convert()
        self.big_lane_surface = pygame.Surface(
            (width_in_pixels, width_in_pixels)
        ).convert()
        self.draw_road_map(
            self.big_map_surface,
            self.big_lane_surface,
            carla_world,
            carla_map,
            self.world_to_pixel,
            self.world_to_pixel_width,
        )
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(
        self,
        map_surface,
        lane_surface,
        carla_world,
        carla_map,
        world_to_pixel,
        world_to_pixel_width,
    ):
        # map_surface.fill(COLOR_ALUMINIUM_4)
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def draw_lane_marking(surface, points, solid=True):
            if solid:
                # pygame.draw.lines(surface, COLOR_ORANGE_0, False, points, 2)
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
            else:
                broken_lines = [
                    x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0
                ]
                for line in broken_lines:
                    # pygame.draw.lines(surface, COLOR_ORANGE_0, False, line, 2)
                    pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            start = transform.location
            end = start + 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [world_to_pixel(x) for x in [start, end]], 4
            )
            pygame.draw.lines(
                surface,
                color,
                False,
                [world_to_pixel(x) for x in [left, start, right]],
                4,
            )

        def draw_stop(surface, font_surface, transform, color=COLOR_ALUMINIUM_2):
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = (
                carla.Location(-forward_vector.y, forward_vector.x, forward_vector.z)
                * waypoint.lane_width
                / 2
                * 0.7
            )

            line = [
                (waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                (waypoint.transform.location + (forward_vector * 1.5) - (left_vector)),
            ]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(
                lateral_shift(waypoint.transform, shift), project_to_road=False
            )
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (
                    w.lane_id * waypoint.lane_id < 0
                ) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [
                lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints
            ]
            right_marking = [
                lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints
            ]

            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1),
                )
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1),
                )

                # Dian: Do not draw them arrows
                # for n, wp in enumerate(waypoints):
                #     if (n % 400) == 0:
                #         draw_arrow(map_surface, wp.transform)

        actors = carla_world.get_actors()
        stops_transform = [
            actor.get_transform() for actor in actors if "stop" in actor.type_id
        ]
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont("Arial", font_size, True)
        font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        font_surface = pygame.transform.scale(
            font_surface, (font_surface.get_width(), font_surface.get_height() * 2)
        )

        # Dian: do not draw stop sign

        # for stop in stops_transform:
        #     draw_stop(map_surface,font_surface, stop)

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(
                self.big_map_surface, (width, width)
            )


class Renderer:
    def __init__(self, name, client, world, town_map, hero_actor):
        self.name = name
        self.server_fps = 0.0
        self.simulation_time = 0

        self.server_clock = pygame.time.Clock()

        # World data
        self.client = client
        self.world = world
        self.town_map = town_map
        self.actors_with_transforms = []

        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0

        # Hero actor
        self.hero_actor = hero_actor
        self.hero_transform = hero_actor.get_transform()

        self.scale_offset = [0, 0]

        self.vehicle_id_surface = None
        self.result_surface = None

        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.affected_traffic_light = None

        # Map info
        self.map_image = None
        self.original_surface_size = None
        # self.actors_surface = None

        self.self_surface = None
        self.vehicle_surface = None
        self.walker_surface = None

        self.hero_map_surface = None
        self.hero_lane_surface = None
        self.hero_self_surface = None
        self.hero_vehicle_surface = None
        self.hero_walker_surface = None
        self.hero_traffic_light_surface = None

        self.window_map_surface = None
        self.window_lane_surface = None
        self.window_self_surface = None
        self.window_vehicle_surface = None
        self.window_walker_surface = None
        self.window_traffic_light_surface = None

        self.hero_map_image = None
        self.hero_lane_image = None
        self.hero_self_image = None
        self.hero_vehicle_image = None
        self.hero_walker_image = None
        self.hero_traffic_image = None

    def start(self):
        # Create Surfaces
        self.map_image = MapImage(self.world, self.town_map, PIXELS_PER_METER)

        self.original_surface_size = (
            320  # min(self.module_hud.dim[0], self.module_hud.dim[1])
        )
        self.window_width = 320
        self.window_height = 320
        self.surface_size = self.map_image.big_map_surface.get_width()

        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)

        # Render Actors
        # self.actors_surface = pygame.Surface((self.map_image.map_surface.get_width(), self.map_image.map_surface.get_height()))
        # self.actors_surface.set_colorkey(COLOR_BLACK)
        self.vehicle_surface = pygame.Surface(
            (
                self.map_image.map_surface.get_width(),
                self.map_image.map_surface.get_height(),
            )
        )
        self.vehicle_surface.set_colorkey(COLOR_BLACK)
        self.self_surface = pygame.Surface(
            (
                self.map_image.map_surface.get_width(),
                self.map_image.map_surface.get_height(),
            )
        )
        self.self_surface.set_colorkey(COLOR_BLACK)
        self.walker_surface = pygame.Surface(
            (
                self.map_image.map_surface.get_width(),
                self.map_image.map_surface.get_height(),
            )
        )
        self.walker_surface.set_colorkey(COLOR_BLACK)
        self.traffic_light_surface = pygame.Surface(
            (
                self.map_image.map_surface.get_width(),
                self.map_image.map_surface.get_height(),
            )
        )
        self.traffic_light_surface.set_colorkey(COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface(
            (self.surface_size, self.surface_size)
        ).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()

        self.hero_map_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.hero_lane_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.hero_self_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.hero_vehicle_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.hero_walker_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.hero_traffic_light_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()

        self.window_map_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_map_surface.set_colorkey(COLOR_BLACK)
        self.window_lane_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_lane_surface.set_colorkey(COLOR_BLACK)
        self.window_self_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_self_surface.set_colorkey(COLOR_BLACK)
        self.window_vehicle_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_vehicle_surface.set_colorkey(COLOR_BLACK)
        self.window_walker_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_walker_surface.set_colorkey(COLOR_BLACK)
        self.window_traffic_light_surface = pygame.Surface(
            (self.window_width, self.window_height)
        ).convert()
        self.window_traffic_light_surface.set_colorkey(COLOR_BLACK)

        self.result_surface = pygame.Surface(
            (self.surface_size, self.surface_size)
        ).convert()
        # print("self.surface_size", self.surface_size)
        self.result_surface.set_colorkey(COLOR_BLACK)

        self.rendered_surface = pygame.Surface((320, 320)).convert()
        self.rendered_surface.set_colorkey(COLOR_BLACK)

    def update(self):
        actors = self.world.get_actors()
        self.actors_with_transforms = [
            (actor, actor.get_transform()) for actor in actors
        ]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()

    def _split_actors(self):
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if "vehicle" in actor.type_id:
                vehicles.append(actor_with_transform)
            elif "traffic_light" in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif "speed_limit" in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif "walker" in actor.type_id:
                walkers.append(actor_with_transform)

        info_text = []
        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.hero_actor.id]

            # def distance(v): return location.distance(v.get_location())
            # for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
            #     if n > 15:
            #         break
            #     vehicle_type = get_actor_display_name(vehicle, truncate=22)
            #     info_text.append('% 5d %s' % (vehicle.id, vehicle_type))

        return (vehicles, traffic_lights, speed_limits, walkers)

    def get_bounding_box(self, actor):
        bb = actor.trigger_volume.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=-bb.y),
        ]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners

    def _render_traffic_lights(
        self,
        surface,
        list_tl,
        world_to_pixel,
        world_to_pixel_width,
        from_snapshot=False,
    ):
        self.affected_traffic_light = None

        for tl in list_tl:
            if from_snapshot:
                world_pos = carla.Location(
                    x=tl["location"]["x"],
                    y=tl["location"]["y"],
                )
            else:
                world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)
            # if self.hero_actor is not None and hasattr(tl, 'trigger_volume'):
            #     corners = self.get_bounding_box(tl)
            #     corners = [world_to_pixel(p) for p in corners]
            #     tl_t = tl.get_transform()

            #     transformed_tv = tl_t.transform(tl.trigger_volume.location)
            #     hero_location = self.hero_actor.get_location()
            #     d = hero_location.distance(transformed_tv)
            #     s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
            # if ( d <= s ):
            # Highlight traffic light
            # print (tl.state)
            # self.affected_traffic_light = tl
            # srf = self.traffic_light_surfaces.surfaces['h']
            # surface.blit(srf, srf.get_rect(center=pos))

            if from_snapshot:
                if tl["state"] == int(tls.Red):
                    # color = COLOR_SCARLET_RED_0
                    color = COLOR_TRAFFIC_RED
                elif tl["state"] == int(tls.Yellow):
                    color = COLOR_TRAFFIC_YELLOW
                    # color = COLOR_BUTTER_0
                elif tl["state"] == int(tls.Green):
                    color = COLOR_TRAFFIC_GREEN
                    # color = COLOR_CHAMELEON_0
                else:
                    continue  # Unknown or off traffic light
            else:
                if tl.state == tls.Red:
                    # color = COLOR_SCARLET_RED_0
                    color = COLOR_TRAFFIC_RED
                elif tl.state == tls.Yellow:
                    color = COLOR_TRAFFIC_YELLOW
                    # color = COLOR_BUTTER_0
                elif tl.state == tls.Green:
                    color = COLOR_TRAFFIC_GREEN
                    # color = COLOR_CHAMELEON_0
                else:
                    continue  # Unknown or off traffic light

            # Draw circle instead of rectangle
            # bb = tl.bounding_box.extent
            radius = world_to_pixel_width(1.5)
            pygame.draw.circle(surface, color, pos, radius)

            # corners = [
            #     (pos[0]-radius, pos[1]-radius),
            #     (pos[0]+radius, pos[1]-radius),
            #     (pos[0]+radius, pos[1]+radius),
            #     (pos[0]-radius, pos[1]+radius)
            # ]
            # # corners = [world_to_pixel(p) for p in corners]
            # pygame.draw.polygon(surface, color, corners)

            # srf = self.traffic_light_surfaces.surfaces[tl.state]
            # surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(
        self, surface, list_sl, world_to_pixel, world_to_pixel_width
    ):

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont("Arial", font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Render speed limit
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split(".")[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)

            # Blit
            if self.hero_actor is not None:
                # Rotate font surface with respect to hero vehicle front
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel, from_snapshot=False):
        # print ("Walkers")

        for w in list_w:
            # color = COLOR_PLUM_0
            color = COLOR_WHITE
            # Compute bounding box points
            if from_snapshot:
                bb = w["bbox"]
                corners = [
                    carla.Location(x=-bb["x"], y=-bb["y"]),
                    carla.Location(x=bb["x"], y=-bb["y"]),
                    carla.Location(x=bb["x"], y=bb["y"]),
                    carla.Location(x=-bb["x"], y=bb["y"]),
                ]
                w_location = carla.Location(x=w["location"]["x"], y=w["location"]["y"])
                corners = [corner + w_location for corner in corners]
            else:
                if not hasattr(w[0], "bounding_box"):
                    continue

                bb = w[0].bounding_box.extent
                corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=bb.x, y=-bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                ]
                w[1].transform(corners)

            corners = [world_to_pixel(p) for p in corners]
            # print (corners)
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(
        self, vehicle_surface, self_surface, list_v, world_to_pixel, from_snapshot=False
    ):
        # print ("rendered a car?!")
        for v in list_v:
            # color = COLOR_SKY_BLUE_0
            color = COLOR_WHITE

            if not from_snapshot and v[0].attributes["role_name"] == "hero":
                # Do not render othre vehicles
                # print (v[1])
                surface = self_surface
            else:
                surface = vehicle_surface

                # continue # Do not render itself
            # Compute bounding box points
            if from_snapshot:
                bb = v["bbox"]
                corners = [
                    carla.Location(x=-bb["x"], y=-bb["y"]),
                    carla.Location(x=bb["x"], y=-bb["y"]),
                    carla.Location(x=bb["x"], y=bb["y"]),
                    carla.Location(x=-bb["x"], y=bb["y"]),
                ]
                v_location = carla.Location(x=v["location"]["x"], y=v["location"]["y"])
                corners = [corner + v_location for corner in corners]
            else:
                bb = v[0].bounding_box.extent
                corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=-bb.y),
                ]
                v[1].transform(corners)
            # print ("Vehicle")
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)
            # pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))

    def render_actors(
        self,
        vehicle_surface,
        self_surface,
        walker_surface,
        traffic_light_surface,
        vehicles,
        traffic_lights,
        speed_limits,
        walkers,
        from_snapshot=False,
    ):
        # Static actors

        # TODO: render traffic lights and speed limits on respective channels
        if from_snapshot:
            self._render_traffic_lights(
                traffic_light_surface,
                traffic_lights,
                self.map_image.world_to_pixel,
                self.map_image.world_to_pixel_width,
                from_snapshot=True,
            )
        else:
            self._render_traffic_lights(
                traffic_light_surface,
                [tl[0] for tl in traffic_lights],
                self.map_image.world_to_pixel,
                self.map_image.world_to_pixel_width,
                from_snapshot=False,
            )
        # self._render_speed_limits(
        # surface, [sl[0] for sl in speed_limits],
        # self.map_image.world_to_pixel,
        # self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(
            vehicle_surface,
            self_surface,
            vehicles,
            self.map_image.world_to_pixel,
            from_snapshot=from_snapshot,
        )
        self._render_walkers(
            walker_surface,
            walkers,
            self.map_image.world_to_pixel,
            from_snapshot=from_snapshot,
        )

    def clip_surfaces(self, clipping_rect):
        # self.actors_surface.set_clip(clipping_rect)
        self.self_surface.set_clip(clipping_rect)
        self.vehicle_surface.set_clip(clipping_rect)
        self.walker_surface.set_clip(clipping_rect)
        self.traffic_light_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor):
        # FIXME
        m = (0, 0)

        # Percentage of surface where mouse position is actually
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)

        # Offset will be the previously accumulated offset added with the
        # difference of mouse positions in the old and new scales
        diff_between_scales = (
            (float(self.prev_scaled_size) * px) - (float(self.scaled_size) * px),
            (float(self.prev_scaled_size) * py) - (float(self.scaled_size) * py),
        )

        self.scale_offset = (
            self.scale_offset[0] + diff_between_scales[0],
            self.scale_offset[1] + diff_between_scales[1],
        )

        # Update previous scale
        self.prev_scaled_size = self.scaled_size

        # Scale performed
        self.map_image.scale_map(scale_factor)

    def render(self, snapshot=None):
        if snapshot is None and self.actors_with_transforms is None:
            return

        self.result_surface.fill(COLOR_BLACK)

        if snapshot is None:
            vehicles, traffic_lights, speed_limits, walkers = self._split_actors()
        else:
            vehicles = snapshot["vehicles"]
            traffic_lights = snapshot["traffic_lights"]
            speed_limits = []
            walkers = snapshot["walkers"]

        scale_factor = 1.0
        # print("scale_factor", scale_factor)
        # print("self.map_image.width", self.map_image.width)
        self.scaled_size = int(self.map_image.width * scale_factor)

        if self.scaled_size != self.prev_scaled_size:
            self._compute_scale(scale_factor)

        # Render Actors
        self.self_surface.fill(COLOR_BLACK)
        self.vehicle_surface.fill(COLOR_BLACK)
        self.walker_surface.fill(COLOR_BLACK)
        self.traffic_light_surface.fill(COLOR_BLACK)

        self.render_actors(
            # self.actors_surface,
            self.vehicle_surface,
            self.self_surface,
            self.walker_surface,
            self.traffic_light_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers,
            from_snapshot=snapshot is not None,
        )

        # Blit surfaces
        surfaces = [(self.map_image.map_surface, (0, 0))]
        # surfaces = ((self.map_image.map_surface, (0, 0)),
        #             # (self.actors_surface, (0, 0)),
        #             # (self.vehicle_id_surface, (0, 0)),
        #             )

        center_offset = (0, 0)
        angle = (
            0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90
        )
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)

        if self.hero_actor is not None:
            if snapshot is None:
                hero_front = self.hero_transform.get_forward_vector()
                hero_location_screen = self.map_image.world_to_pixel(
                    self.hero_transform.location
                )
            else:
                hero_location = snapshot["player"]["transform"]["location"]
                hero_location = carla.Location(
                    x=hero_location["x"],
                    y=hero_location["y"],
                    z=hero_location["z"],
                )
                hero_location_screen = self.map_image.world_to_pixel(hero_location)

                hero_orientation = snapshot["player"]["transform"]["orientation"]
                hero_front = carla.Location(
                    x=hero_orientation["x"], y=hero_orientation["y"]
                )

            # print(hero_location_screen)
            offset = [0, 0]
            offset[0] += hero_location_screen[0] - self.hero_map_surface.get_width() / 2
            offset[0] += hero_front.x * PIXELS_AHEAD_VEHICLE
            offset[1] += (
                hero_location_screen[1] - self.hero_map_surface.get_height() / 2
            )
            offset[1] += hero_front.y * PIXELS_AHEAD_VEHICLE

            # Apply clipping rect
            clipping_rect = pygame.Rect(
                offset[0],
                offset[1],
                self.hero_map_surface.get_width(),
                self.hero_map_surface.get_height(),
            )
            # print("clipping_rect", clipping_rect)

            self.clip_surfaces(clipping_rect)
            # self.border_round_surface.set_clip(clipping_rect)

            # self.hero_surface.fill(COLOR_ALUMINIUM_4)
            # self.hero_surface.blit(self.result_surface, (-translation_offset[0],
            #                                              -translation_offset[1]))

            # self.hero_map_surface.blit

            self.hero_self_surface.fill(COLOR_BLACK)
            self.hero_map_surface.fill(COLOR_BLACK)
            self.hero_vehicle_surface.fill(COLOR_BLACK)
            self.hero_walker_surface.fill(COLOR_BLACK)
            self.hero_traffic_light_surface.fill(COLOR_BLACK)

            self.hero_self_surface.blit(self.self_surface, (-offset[0], -offset[1]))
            self.hero_map_surface.blit(
                self.map_image.map_surface, (-offset[0], -offset[1])
            )
            self.hero_lane_surface.blit(
                self.map_image.lane_surface, (-offset[0], -offset[1])
            )
            self.hero_vehicle_surface.blit(
                self.vehicle_surface, (-offset[0], -offset[1])
            )
            self.hero_walker_surface.blit(self.walker_surface, (-offset[0], -offset[1]))
            self.hero_traffic_light_surface.blit(
                self.traffic_light_surface, (-offset[0], -offset[1])
            )

            # rotated_result_surface = pygame.transform.rotozoom(
            # self.hero_surface, angle, 0.9).convert()

            # Rotate: map/vehicle/walker surface
            # TODO: traffic lights and speed limits surface
            rz = pygame.transform.rotozoom

            rotated_map_surface = rz(self.hero_map_surface, angle, 0.9).convert()
            rotated_lane_surface = rz(self.hero_lane_surface, angle, 0.9).convert()
            rotated_vehicle_surface = rz(
                self.hero_vehicle_surface, angle, 0.9
            ).convert()
            rotated_walker_surface = rz(self.hero_walker_surface, angle, 0.9).convert()
            rotated_traffic_surface = rz(
                self.hero_traffic_light_surface, angle, 0.9
            ).convert()
            # rotated_self_surface = rz(self.hero_self_surface, angle, 0.9).convert()

            # center = (display.get_width() / 2, display.get_height() / 2)
            center = (320.0 / 2, 320.0 / 2)
            # print("center", center)
            rotation_map_pivot = rotated_map_surface.get_rect(center=center)
            rotation_lane_pivot = rotated_lane_surface.get_rect(center=center)
            rotation_vehicle_pivot = rotated_vehicle_surface.get_rect(center=center)
            rotation_walker_pivot = rotated_walker_surface.get_rect(center=center)
            rotation_traffic_pivot = rotated_traffic_surface.get_rect(center=center)
            # rotation_self_pivot = rotated_self_surface.get_rect(center=center)

            self.window_map_surface.blit(rotated_map_surface, rotation_map_pivot)
            self.window_lane_surface.blit(rotated_lane_surface, rotation_lane_pivot)
            self.window_vehicle_surface.blit(
                rotated_vehicle_surface, rotation_vehicle_pivot
            )
            self.window_walker_surface.blit(
                rotated_walker_surface, rotation_walker_pivot
            )
            self.window_traffic_light_surface.blit(
                rotated_traffic_surface, rotation_traffic_pivot
            )
            # self.window_self_surface.blit(rotated_self_surface, rotation_self_pivot)

            make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(
                axis=-1
            )

            # Save surface as rgb array
            self.hero_map_image = make_image(self.window_map_surface)
            self.hero_lane_image = make_image(self.window_lane_surface)
            self.hero_vehicle_image = make_image(self.window_vehicle_surface)
            self.hero_walker_image = make_image(self.window_walker_surface)
            self.hero_traffic_image = np.swapaxes(
                pygame.surfarray.array3d(self.window_traffic_light_surface), 0, 1
            )
            # print("self.hero_map_image.shape", self.hero_map_image.shape)
            # self.hero_self_image = np.swapaxes(
            # pygame.surfarray.array3d(self.window_self_surface),0,1).mean(axis=-1)
        else:
            # Translation offset
            translation_offset = (
                self.module_input.mouse_offset[0] * scale_factor + self.scale_offset[0],
                self.module_input.mouse_offset[1] * scale_factor + self.scale_offset[1],
            )
            center_offset = (
                abs(display.get_width() - self.surface_size) / 2 * scale_factor,
                0,
            )

            # Apply clipping rect
            clipping_rect = pygame.Rect(
                -translation_offset[0] - center_offset[0],
                -translation_offset[1],
                self.module_hud.dim[0],
                self.module_hud.dim[1],
            )
            self.clip_surfaces(clipping_rect)
            # Util.blits(self.result_surface, surfaces)

            # display.blit(
            #         self.result_surface,
            #         (translation_offset[0] + center_offset[0], translation_offset[1]))

    def _get_rendered_surfaces(self):
        return (
            self.hero_map_image,
            self.hero_lane_image,
            # self.hero_self_image,
            self.hero_vehicle_image,
            self.hero_walker_image,
            self.hero_traffic_image,
        )

    def get_result(self):
        road, lane, vehicle, pedestrian, traffic = self._get_rendered_surfaces()
        result = {
            "road": np.uint8(road),
            "lane": np.uint8(lane),
            "vehicle": np.uint8(vehicle),
            "pedestrian": np.uint8(pedestrian),
            "traffic": np.uint8(traffic),
        }
        return result

    def get_render(self):
        size = self.hero_map_surface.get_size()
        surface = pygame.Surface(size)
        surface.fill((255, 255, 255))

        map_surface = pygame.Surface(size)
        map_surface.set_colorkey(COLOR_BLACK)
        map_surface.fill((255, 208, 77))
        map_surface.blit(
            self.hero_map_surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT
        )
        surface.blit(map_surface, (0, 0))

        lane_surface = pygame.Surface(size)
        lane_surface.set_colorkey(COLOR_BLACK)
        lane_surface.fill((227, 158, 20))
        lane_surface.blit(
            self.hero_lane_surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT
        )
        surface.blit(lane_surface, (0, 0))

        vehicle_surface = pygame.Surface(size)
        vehicle_surface.set_colorkey(COLOR_BLACK)
        vehicle_surface.fill((0, 49, 128))
        vehicle_surface.blit(
            self.hero_vehicle_surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT
        )
        surface.blit(vehicle_surface, (0, 0))

        self_surface = pygame.Surface(size)
        self_surface.set_colorkey(COLOR_BLACK)
        self_surface.fill((0, 170, 179))
        self_surface.blit(
            self.hero_self_surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT
        )
        surface.blit(self_surface, (0, 0))

        walker_surface = pygame.Surface(size)
        walker_surface.set_colorkey(COLOR_BLACK)
        walker_surface.fill((145, 37, 132))
        walker_surface.blit(
            self.hero_walker_surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT
        )
        surface.blit(walker_surface, (0, 0))

        traffic_light_surface = pygame.Surface(size)
        traffic_light_surface.set_colorkey(COLOR_BLACK)
        traffic_light_surface.blit(self.hero_traffic_light_surface, (0, 0))
        traffic_light_pixel_array = pygame.PixelArray(traffic_light_surface)
        traffic_light_pixel_array.replace(COLOR_TRAFFIC_RED, (204, 0, 0))
        traffic_light_pixel_array.replace(COLOR_TRAFFIC_YELLOW, (204, 184, 0))
        traffic_light_pixel_array.replace(COLOR_TRAFFIC_GREEN, (80, 179, 0))
        traffic_light_pixel_array.close()

        surface.blit(traffic_light_surface, (0, 0))

        angle = (
            0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90
        )
        rotated = pygame.transform.rotate(surface, angle)
        pivot = rotated.get_rect(center=(size[0] / 2, size[1] / 2))
        surface.blit(rotated, pivot)
        return surface
