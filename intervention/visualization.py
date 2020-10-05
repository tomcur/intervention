from enum import Enum
import pygame
import numpy as np

from .coordinates import ego_coordinate_to_image_coordinate


class Action(Enum):
    SWITCH_CONTROL = 1
    THROTTLE = 2
    BRAKE = 3
    LEFT = 4
    RIGHT = 5


class Visualizer:
    def __init__(self):
        pygame.init()
        size = (round(640 * 16 / 9), 640)
        # self._screen = pygame.display.set_mode(size, 0, 32)
        self._screen = pygame.display.set_mode(size, 0, 32)
        self._surface = pygame.Surface(size)

        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 16)

    def _render_control(self, control):
        from pygame import Rect, draw

        BAR_WIDTH = 100
        BAR_HEIGHT = 20

        surf = pygame.Surface((200, 80))
        text_surf = pygame.Surface((100, 80))
        t = self._font.render("throttle:", True, (220, 220, 220))
        text_surf.blit(t, (0, 2))
        t = self._font.render("brake:", True, (220, 220, 220))
        text_surf.blit(t, (0, 32))
        t = self._font.render("steering:", True, (220, 220, 220))
        text_surf.blit(t, (0, 62))

        bar_surf = pygame.Surface((100, 80))

        # Throttle
        r = Rect((0, 0), (BAR_WIDTH, BAR_HEIGHT))
        draw.rect(bar_surf, (220, 220, 220), r, 2)

        r = Rect((0, 0), (round(control.throttle * BAR_WIDTH), BAR_HEIGHT))
        draw.rect(bar_surf, (220, 220, 220), r)

        # Brake
        r = Rect((0, 30), (BAR_WIDTH, BAR_HEIGHT))
        draw.rect(bar_surf, (220, 220, 220), r, 2)

        r = Rect((0, 30), (round(control.brake * BAR_WIDTH), BAR_HEIGHT))
        draw.rect(bar_surf, (220, 220, 220), r)

        # Steering
        r = Rect((0, 60), (BAR_WIDTH, BAR_HEIGHT))
        draw.rect(bar_surf, (220, 220, 220), r, 2)

        scaled = round(abs(control.steer) * BAR_WIDTH)
        if control.steer < 0:
            r = Rect((BAR_WIDTH / 2 - scaled, 60), (scaled, BAR_HEIGHT))
            draw.rect(bar_surf, (220, 220, 220), r)
        else:
            r = Rect((BAR_WIDTH / 2, 60), (scaled, BAR_HEIGHT))
            draw.rect(bar_surf, (220, 220, 220), r)

        surf.blit(text_surf, (0, 0))
        surf.blit(bar_surf, (100, 0))

        return surf

    def render(
        self,
        rgb,
        controller,
        control_difference,
        student_control,
        teacher_control,
        s,
        target_waypoints,
    ):
        """Render state of a tick.

        Parameters:
        rgb: The camera image.
        controller: The current controller (student or teacher).
        control_difference: The current (discounted) integral of control difference.
        student_control: The current student control input.
        teacher_control: The current teacher control input.

        Returns:
        [Action]: a list of user-input actions
        """

        self._screen.fill((0, 0, 0))

        rgb = np.swapaxes(rgb, 0, 1)
        # print(rgb.shape)
        rgb_surf = pygame.pixelcopy.make_surface(rgb)

        for [location_x, location_y] in target_waypoints:
            im_location_x, im_location_y = ego_coordinate_to_image_coordinate(
                location_x, location_y, forward_offset=0.0
            )
            pygame.draw.circle(
                rgb_surf, (150, 0, 0), (int(im_location_x), int(im_location_y)), 5
            )
            pygame.draw.circle(
                rgb_surf, (255, 255, 255), (int(im_location_x), int(im_location_y)), 3
            )

        self._screen.blit(rgb_surf, (0, 0))

        controller_surf = self._font.render(
            "controller:               %s" % controller, True, (240, 240, 240)
        )
        self._screen.blit(controller_surf, (0, 200))

        control_difference_surf = self._font.render(
            "discounted control error: %.2f" % control_difference, True, (240, 240, 240)
        )
        self._screen.blit(control_difference_surf, (0, 220))

        self._screen.blit(
            self._font.render("student control", True, (240, 240, 240)), (400, 0)
        )
        student_control_surf = self._render_control(student_control)
        self._screen.blit(student_control_surf, (400, 20))

        self._screen.blit(
            self._font.render("teacher control", True, (240, 240, 240)), (400, 150)
        )
        teacher_control_surf = self._render_control(teacher_control)
        self._screen.blit(teacher_control_surf, (400, 170))

        # print(s)
        # print(s.shape)
        self._screen.blit(s, (300, 300))

        pygame.display.flip()

        actions = []
        events = pygame.event.get()
        events = [event.key for event in events if event.type == pygame.KEYDOWN]
        if pygame.K_TAB in events:
            actions.append(Action.SWITCH_CONTROL)

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_w]:
            actions.append(Action.THROTTLE)
        if pressed[pygame.K_s]:
            actions.append(Action.BRAKE)
        if pressed[pygame.K_a]:
            actions.append(Action.LEFT)
        if pressed[pygame.K_d]:
            actions.append(Action.RIGHT)

        return actions

    def render_heatmap(self, heatmaps):
        for idx in range(5):
            heatmap = heatmaps[0][idx].cpu().numpy()
            scaled = ((heatmap - heatmap.min()) * (255 / heatmap.max())).astype("uint8")
            rgb = np.stack((scaled,) * 3, axis=-1)
            rgb = np.swapaxes(rgb, 0, 1)
            rgb_surf = pygame.pixelcopy.make_surface(rgb)
            self._screen.blit(rgb_surf, (0, 50 * idx))

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
        pygame.display.flip()

        actions = []
        events = pygame.event.get()
        events = [event.key for event in events if event.type == pygame.KEYDOWN]
        if pygame.K_TAB in events:
            actions.append(Action.SWITCH_CONTROL)

        return actions
