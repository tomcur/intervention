import math
from collections import deque
from enum import Enum
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

import carla
import numpy as np
import pygame
import pygame.locals as pglocals
from pygame import gfxdraw
from typing_extensions import Literal

from .carla_utils.agents.navigation.local_planner import RoadOption
from .coordinates import ego_coordinate_to_image_coordinate


class Action(Enum):
    SWITCH_CONTROL = 0
    THROTTLE = 10
    BRAKE = 11
    LEFT = 12
    RIGHT = 13
    PREVIOUS = 20
    NEXT = 21
    PLAY = 22
    GO_LEFT = 30
    GO_STRAIGHT = 31
    GO_RIGHT = 32


def _render_control(control: carla.VehicleControl, font: pygame.font.Font):
    BAR_WIDTH = 100
    BAR_HEIGHT = 20

    surf = pygame.Surface((220, 100))
    text_surf = pygame.Surface((120, 100))
    t = font.render("throttle:", True, (220, 220, 220))
    text_surf.blit(t, (0, 0))
    t = font.render("brake:", True, (220, 220, 220))
    text_surf.blit(t, (0, 30))
    t = font.render("steering:", True, (220, 220, 220))
    text_surf.blit(t, (0, 60))

    bar_surf = pygame.Surface((100, 100))

    # Throttle
    r = pygame.Rect((0, 0), (BAR_WIDTH, BAR_HEIGHT))
    pygame.draw.rect(bar_surf, (220, 220, 220), r, 2)

    r = pygame.Rect((0, 0), (round(control.throttle * BAR_WIDTH), BAR_HEIGHT))
    pygame.draw.rect(bar_surf, (220, 220, 220), r)

    # Brake
    r = pygame.Rect((0, 30), (BAR_WIDTH, BAR_HEIGHT))
    pygame.draw.rect(bar_surf, (220, 220, 220), r, 2)

    r = pygame.Rect((0, 30), (round(control.brake * BAR_WIDTH), BAR_HEIGHT))
    pygame.draw.rect(bar_surf, (220, 220, 220), r)

    # Steering
    r = pygame.Rect((0, 60), (BAR_WIDTH, BAR_HEIGHT))
    pygame.draw.rect(bar_surf, (220, 220, 220), r, 2)

    scaled = round(abs(control.steer) * BAR_WIDTH)
    if control.steer < 0:
        r = pygame.Rect((BAR_WIDTH / 2 - scaled, 60), (scaled, BAR_HEIGHT))
        pygame.draw.rect(bar_surf, (220, 220, 220), r)
    else:
        r = pygame.Rect((BAR_WIDTH / 2, 60), (scaled, BAR_HEIGHT))
        pygame.draw.rect(bar_surf, (220, 220, 220), r)

    surf.blit(text_surf, (0, 0))
    surf.blit(bar_surf, (120, 0))

    return surf


class FramePainter:
    """
    A very bare-bones, but stateful, painter of PyGame frames.
    """

    PADDING = 5
    IMAGE_PANEL_X = 0
    IMAGE_PANEL_WIDTH = 450
    COMMAND_LABEL_X = 25 + 384 // 2
    COMMAND_LABEL_Y = 25
    IMAGE_X = 25
    IMAGE_Y = COMMAND_LABEL_Y + PADDING + 25
    CONTROL_X = IMAGE_PANEL_X + IMAGE_PANEL_WIDTH + PADDING
    CONTROL_WIDTH = 220
    CONTROL_GROUP_HEIGHT = 170
    CONTROL_FIGURE_HEIGHT = 100
    CONTROL_FIGURE_GRAPH_X = 16 * 4
    CONTROL_FIGURE_GRAPH_Y = 16 / 2
    CONTROL_FIGURE_GRAPH_HEIGHT = 100 - 16
    CONTROL_FIGURE_GRAPH_WIDTH = CONTROL_WIDTH - CONTROL_FIGURE_GRAPH_X
    BIRDVIEW_X = CONTROL_X + CONTROL_WIDTH + PADDING

    def __init__(
        self,
        size: Tuple[int, int],
        font: pygame.font.Font,
        control_difference: Deque[float],
    ):
        self._surface = pygame.Surface(size)
        self._font = font
        self._control_difference = control_difference

        self._next_control_y = 0
        self._rgb_height = 0

    def add_command(self, command: RoadOption) -> None:
        label: Optional[str] = None
        if command is RoadOption.LEFT:
            label = "<-"
        elif command is RoadOption.STRAIGHT:
            label = "^"
        elif command is RoadOption.RIGHT:
            label = "->"

        if label:
            (width, height) = self._font.size(label)
            self._surface.blit(
                self._font.render(label, True, (240, 240, 240)),
                (
                    FramePainter.COMMAND_LABEL_X - width // 2,
                    FramePainter.COMMAND_LABEL_Y,
                ),
            )

    def add_rgb(self, rgb: np.ndarray) -> None:
        """
        Add an RGB image. You should only add it once per frame.
        """
        self._rgb_height = rgb.shape[0]
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self._surface.blit(surface, (FramePainter.IMAGE_X, FramePainter.IMAGE_Y))

    def add_annotation(self, annotation: Iterable[str]) -> None:
        """
        Add an annotation. Each string in the list is printed on a new line.

        If this method is used, it _MUST_ be called after the call to `add_rgb`.
        """
        cumulative_height = 0
        for i, text in enumerate(annotation):
            label = self._font.render(text, True, (220, 220, 220))
            self._surface.blit(
                label,
                (
                    FramePainter.IMAGE_X,
                    FramePainter.IMAGE_Y
                    + FramePainter.PADDING
                    + self._rgb_height
                    + cumulative_height,
                ),
            )
            cumulative_height += label.get_height()

    def add_waypoints(
        self,
        waypoints: Iterable[Tuple[float, float]],
        color: Tuple[int, int, int] = (240, 240, 240),
        grayout: bool = False,
    ) -> None:
        if grayout:
            color = (180, 180, 180)

        for [location_x, location_y] in waypoints:
            im_location_x, im_location_y = ego_coordinate_to_image_coordinate(
                location_x, location_y, forward_offset=0.0
            )
            draw_x = int(im_location_x) + FramePainter.IMAGE_X
            draw_y = int(im_location_y) + FramePainter.IMAGE_Y
            if not (0 <= draw_x < self._surface.get_width()) or not (
                0 <= draw_y < self._surface.get_height()
            ):
                continue

            gfxdraw.aacircle(self._surface, draw_x, draw_y, 5, (255, 255, 255))
            gfxdraw.filled_circle(self._surface, draw_x, draw_y, 5, (255, 255, 255))
            gfxdraw.aacircle(self._surface, draw_x, draw_y, 4, (0, 0, 0))
            gfxdraw.filled_circle(self._surface, draw_x, draw_y, 4, (0, 0, 0))
            gfxdraw.aacircle(self._surface, draw_x, draw_y, 3, color)
            gfxdraw.filled_circle(self._surface, draw_x, draw_y, 3, color)

    def add_turn_radius(
        self,
        radius: float,
        direction: Union[Literal["LEFT"], Literal["RIGHT"]],
        color: Tuple[int, int, int] = (240, 240, 240),
        grayout: bool = False,
    ) -> None:
        if grayout:
            color = (110, 110, 110)

        max_y = min(radius, 40.0)
        step_size = 0.25

        ys = np.arange(step_size, max_y, step_size)
        if math.isfinite(radius):
            xs = radius - np.sqrt(-(ys ** 2) + radius ** 2)
            if direction == "LEFT":
                xs *= -1
        else:
            xs = np.repeat(0.0, len(ys))

        prev_im_location_x, prev_im_location_y = ego_coordinate_to_image_coordinate(
            0, 0, forward_offset=0.0
        )
        for (location_x, location_y) in zip(xs, ys):
            im_location_x, im_location_y = ego_coordinate_to_image_coordinate(
                float(location_x), float(location_y), forward_offset=0.0
            )
            start_draw_x = int(prev_im_location_x) + FramePainter.IMAGE_X
            start_draw_y = int(prev_im_location_y) + FramePainter.IMAGE_Y
            end_draw_x = int(im_location_x) + FramePainter.IMAGE_X
            end_draw_y = int(im_location_y) + FramePainter.IMAGE_Y
            if (
                (0 <= start_draw_x < self._surface.get_width())
                and (0 <= end_draw_x < self._surface.get_width())
                and (0 <= start_draw_y < self._surface.get_height())
                and (0 <= end_draw_y < self._surface.get_height())
            ):
                gfxdraw.line(
                    self._surface,
                    start_draw_x,
                    start_draw_y,
                    end_draw_x,
                    end_draw_y,
                    color,
                )
            prev_im_location_x = im_location_x
            prev_im_location_y = im_location_y

    def add_control(
        self, name: str, control: carla.VehicleControl, grayout=False
    ) -> None:
        self._surface.blit(
            self._font.render(name, True, (240, 240, 240)),
            (
                FramePainter.CONTROL_X,
                self._next_control_y,
            ),
        )
        control_surf = _render_control(control, self._font)

        if grayout:
            dark = pygame.Surface(
                (control_surf.get_width(), control_surf.get_height()),
                flags=pygame.SRCALPHA,
            )
            dark.fill((75, 75, 75, 0))
            control_surf.blit(dark, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        self._surface.blit(
            control_surf,
            (
                FramePainter.CONTROL_X,
                self._next_control_y + 25,
            ),
        )

        self._next_control_y += FramePainter.CONTROL_GROUP_HEIGHT + FramePainter.PADDING

    def add_control_difference(
        self,
        control_difference: float,
        max_difference: float = 10.0,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Add the current control difference integral. The history of this integral is
        kept track of over multiple frames to draw a graph.
        """
        self._control_difference.append(control_difference)

        surf = pygame.Surface(
            (FramePainter.CONTROL_WIDTH, FramePainter.CONTROL_FIGURE_HEIGHT)
        )

        graph_surf = pygame.Surface(
            (
                FramePainter.CONTROL_FIGURE_GRAPH_WIDTH,
                FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT,
            )
        )

        # Draw graph labels
        for idx in range(0, 6):
            y_label_y = idx / 5.0
            label = self._font.render(
                f"{y_label_y*max_difference:4.1f}", True, (220, 220, 220)
            )
            surf.blit(
                label, (0, (1 - y_label_y) * (FramePainter.CONTROL_FIGURE_HEIGHT - 16))
            )

        # Draw graph outline
        pygame.draw.lines(
            graph_surf,
            (240, 240, 240),
            False,
            [
                (0, 0),
                (0, FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT - 1),
                (
                    FramePainter.CONTROL_FIGURE_GRAPH_WIDTH - 1,
                    FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT - 1,
                ),
            ],
        )

        # Draw graph raster
        for idx in range(1, 5):
            horizontal_raster_y = idx / 5 * FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT
            pygame.draw.line(
                graph_surf,
                (160, 160, 160),
                (0, horizontal_raster_y),
                (FramePainter.CONTROL_FIGURE_GRAPH_WIDTH - 1, horizontal_raster_y),
            )

        if threshold is not None:
            threshold_y = (
                1.0 - threshold / max_difference
            ) * FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT
            pygame.draw.line(
                graph_surf,
                (200, 200, 80),
                (0, threshold_y),
                (FramePainter.CONTROL_FIGURE_GRAPH_WIDTH - 1, threshold_y),
            )

        # Calculate graph points
        num = len(self._control_difference)
        points = []
        for (idx, diff) in enumerate(self._control_difference):
            points.append(
                (
                    (idx / num * FramePainter.CONTROL_FIGURE_GRAPH_WIDTH),
                    (1 - diff / max_difference)
                    * FramePainter.CONTROL_FIGURE_GRAPH_HEIGHT
                    - 1,
                )
            )

        if len(points) >= 2:
            pygame.draw.lines(graph_surf, (240, 240, 240), False, points)

        surf.blit(
            graph_surf,
            (FramePainter.CONTROL_FIGURE_GRAPH_X, FramePainter.CONTROL_FIGURE_GRAPH_Y),
        )
        self._surface.blit(surf, (FramePainter.CONTROL_X, self._next_control_y))

        self._next_control_y += (
            FramePainter.CONTROL_FIGURE_HEIGHT + FramePainter.PADDING
        )

    def add_birdview(self, birdview) -> None:
        self._surface.blit(birdview, (FramePainter.BIRDVIEW_X, 0))

    def add_heatmap(self) -> None:
        pass
        # for idx in range(5):
        #     heatmap = heatmaps[0][idx].cpu().numpy()
        #     scaled = ((heatmap - heatmap.min()) * (255 / heatmap.max())).astype("uint8")
        #     rgb = np.stack((scaled,) * 3, axis=-1)
        #     rgb = np.swapaxes(rgb, 0, 1)
        #     rgb_surf = pygame.pixelcopy.make_surface(rgb)
        #     self._screen.blit(rgb_surf, (0, 50 * idx))

        # print(heatmap.size())
        # # import pdb
        # # pdb.set_trace()
        # heatmap = heatmap[0][1].cpu().numpy()
        # # rgb = np.swapaxes(rgb[:][1], 0, 1)
        # rgb = heatmap
        # print(rgb.shape)
        # rgb = ((rgb - rgb.min()) * (255 / rgb.max())).astype('uint8')
        # rgb = np.stack((rgb,)*3, axis=-1)
        # rgb = np.swapaxes(rgb, 0, 1)
        # print(rgb)
        # rgb_surf = pygame.pixelcopy.make_surface(rgb)
        # self._screen.blit(rgb_surf, (0, 0))


def drive_control_event_processor(
    keydown: List[int], pressed: Dict[int, bool], modifier: int
) -> List[Action]:
    """
    A Pygame event processor used for driving control output.
    """
    actions = []
    if pressed[pygame.K_w]:
        actions.append(Action.THROTTLE)
    if pressed[pygame.K_s]:
        actions.append(Action.BRAKE)
    if pressed[pygame.K_a]:
        actions.append(Action.LEFT)
    if pressed[pygame.K_d]:
        actions.append(Action.RIGHT)
    return actions


def drive_command_event_processor(
    keydown: List[int], pressed: Dict[int, bool], modifier: int
) -> List[Action]:
    """
    A Pygame event processor used for generating driving command input.
    """
    actions = []
    if pressed[pygame.K_LEFT]:
        actions.append(Action.GO_LEFT)
    if pressed[pygame.K_UP]:
        actions.append(Action.GO_STRAIGHT)
    if pressed[pygame.K_RIGHT]:
        actions.append(Action.GO_RIGHT)
    return actions


def dataset_explorer_event_processor(
    keydown: List[int], pressed: Dict[int, bool], modifier: int
) -> List[Action]:
    """
    A Pygame event processor used for dataset exploration.
    """
    actions = []

    if pygame.K_LEFT in keydown:
        actions.append(Action.PREVIOUS)
    elif pressed[pygame.K_LEFT] and not modifier & pygame.KMOD_SHIFT:
        actions.append(Action.PREVIOUS)

    if pygame.K_RIGHT in keydown:
        actions.append(Action.NEXT)
    elif pressed[pygame.K_RIGHT] and not modifier & pygame.KMOD_SHIFT:
        actions.append(Action.NEXT)

    if pygame.K_SPACE in keydown:
        actions.append(Action.PLAY)

    return actions


class Visualizer:
    SIZE = (round(640 * 16 / 9), 640)

    def __init__(
        self,
        event_processor: Callable[
            [List[int], Dict[int, bool], int], List[Action]
        ] = drive_control_event_processor,
    ):
        pygame.init()
        self._screen = pygame.display.set_mode(
            Visualizer.SIZE,
            pglocals.HWSURFACE | pglocals.DOUBLEBUF | pglocals.RESIZABLE,
            32,
        )

        self._painter: Optional[FramePainter] = None
        self._actions: Deque[Action] = deque(maxlen=50)

        self._control_difference: Deque[float] = deque(maxlen=100)

        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 20)

        self._event_processor = event_processor

    def __enter__(self) -> FramePainter:
        self._painter = FramePainter(
            Visualizer.SIZE, self._font, self._control_difference
        )
        return self._painter

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if exc_type:
            return
        assert self._painter is not None

        self._process_events()

        surf = self._painter._surface
        self._screen.fill((0, 0, 0))

        (surface_width, surface_height) = surf.get_size()
        (screen_width, screen_height) = self._screen.get_size()
        scale_factor = min(screen_width / surface_width, screen_height / surface_height)

        self._screen.blit(
            pygame.transform.scale(
                surf,
                (int(scale_factor * surface_width), int(scale_factor * surface_height)),
            ),
            (0, 0),
        )
        pygame.display.flip()

        self._painter = None

    def _process_events(self) -> None:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(
                    event.dict["size"],
                    pglocals.HWSURFACE | pglocals.DOUBLEBUF | pglocals.RESIZABLE,
                )

        keydown_events = [event.key for event in events if event.type == pygame.KEYDOWN]
        pressed = pygame.key.get_pressed()
        modifier = pygame.key.get_mods()

        self._actions.extend(self._event_processor(keydown_events, pressed, modifier))

    def get_actions(self) -> List[Action]:
        self._process_events()

        actions = list(self._actions)
        self._actions.clear()
        return actions
