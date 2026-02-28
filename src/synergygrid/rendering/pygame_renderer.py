import pygame
import json
from os import path
import sys
import numpy as np
from synergygrid.core import ResourceMeta, ResourceCategory, DirectType, SynergyType


class PygameRenderer:
    RENDER_FPS = 60

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5, fps: int = 2) -> None:
        """
        Initializes the Pygame renderer.

        Sets:
        - Size of grid world
        - One logic step fps, not game.
        - Game colors
        - Game font
        - Graphic elements
        """

        pygame.init()  # Initialize pygame
        pygame.display.init()  # initialize the display module
        self.clock = pygame.time.Clock()  # Game clock

        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._step_fps = fps

        # Default font
        self.font = pygame.font.SysFont("Calibre", 30)
        self.world_info_height = self.font.get_height()

        # Sizes of each cell in the grid
        self.cell_width = 64
        self.cell_height = 64

        # Define game window size (width, height)
        self.window_size = (
            self.cell_width * self._grid_cols,
            self.cell_height * self._grid_rows + self.world_info_height + 5,
        )

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)

        self._init_colors()
        self._init_vars()
        self._load_graphics()
        self._set_frames_per_step()

    # ================= #
    #        API        #
    # ================= #

    def render(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_meta: list[ResourceMeta],
        agent_score: int,
    ) -> None | str:
        """
        Draws the game window and all its content, updates and limits the fps.

        Also catches user events (clicking the X top right or hitting ESC) for closing the game.
        """

        # TODO: decide if we go all in on animation and either keep this method or fix the
        # animation render method so it looks slicker, Keep both as of now

        self.window_surface.fill(self.background_clr)

        col = agent_pos[1] * self.cell_width
        row = agent_pos[0] * self.cell_height

        self.window_surface.fill(self.background_clr)

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        self._draw_floor_and_resources(
            resource_positions, resource_meta, is_active_statuses
        )
        self._draw_agent((col, row))
        self._draw_hud(agent_score)

        self._update(self._step_fps)  # TODO: also remove self._step_fps when approach is chosen
        return self._process_quit_events()


    def render_with_animation(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_meta: list[ResourceMeta],
        agent_score: int,
    ) -> None:
        """
        Blocking render: animate agent over a fixed number of frames while everything
        else stays visually static.
        """

        # Track last cell and score
        if not hasattr(self, "_last_agent_cell"):
            self._last_agent_cell = list(agent_pos)
            self._last_agent_score = agent_score
        last = self._last_agent_cell

        # Animate agent if moving
        if agent_pos != last:
            for frame in range(self.frames_per_step + 1):
                # Sets the x and y coordinates for where the agent should be drawn, updating only
                # the axis the it is currently moving along
                progress = frame / self.frames_per_step
                distance_to_target = 1 - progress
                col = (distance_to_target * last[1]) + (progress * agent_pos[1])
                row = (distance_to_target * last[0]) + (progress * agent_pos[0])
                coordinates = (int(col * self.cell_width), int(row * self.cell_height))

                self.window_surface.fill(self.background_clr)

                # Draw the graphics with pygame. blit() draws things in order, so we need to stack
                # elements in the order we want them to be drawn
                self._draw_floor_and_resources(
                    resource_positions,
                    resource_meta,
                    is_active_statuses,
                )
                self._draw_agent(coordinates)

                if not frame == self.frames_per_step:
                    self._draw_hud(self._last_agent_score)
                else:
                    self._draw_hud(agent_score)

                self._update(self.RENDER_FPS)

            # snap logical cell
            self._last_agent_score = agent_score
            self._last_agent_cell = [agent_pos[0], agent_pos[1]]
        else:
            # no movement, single frame
            self.render(
                agent_pos,
                is_active_statuses,
                resource_positions,
                resource_meta,
                agent_score,
            )

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init ===#

    def _init_colors(self) -> None:
        """Sets all the colors the game uses"""

        self.background_clr = (26, 26, 26)
        self.text_clr = (0, 210, 210)

    def _init_vars(self):
        self._padding = 6

    def _load_graphics(self) -> None:
        """Load graphics via JSON file"""

        ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "..", "..", ".."))
        json_file = path.join(ROOT_DIR, "assets/paths.json")

        with open(json_file, "r") as f:
            graphics_paths = json.load(f)

        self.graphics = {}
        for attr, rel_path in graphics_paths.items():
            full_path = path.join(ROOT_DIR, rel_path)
            self.graphics[attr] = pygame.image.load(full_path)

    def _set_frames_per_step(self):
        for f in range(self._step_fps, 0, -1):
            if self.RENDER_FPS % f == 0:
                self.frames_per_step = self.RENDER_FPS // f
                break

    # === API ===#

    def _draw_floor_and_resources(
        self, resource_positions, resource_meta, is_active_statuses
    ):
        """Draw floor tiles and resources"""

        for r in range(self._grid_rows):
            for c in range(self._grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.graphics["floor_img"], pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        res_r, res_c = resource_positions[i]
                        if (r, c) == (res_r, res_c):
                            self._draw_resource(resource_meta[i], pos)

    def _draw_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        if resource_meta.category == ResourceCategory.DIRECT:
            self._draw_direct_resource(resource_meta, pos)
        else:
            self._draw_synergy_resource(resource_meta, pos)

    def _draw_direct_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        if resource_meta.type == DirectType.POSITIVE:
            self.window_surface.blit(self.graphics["positive_resource"], pos)
        elif resource_meta.type == DirectType.NEGATIVE:
            self.window_surface.blit(self.graphics["negative_resource"], pos)

    def _draw_synergy_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        if resource_meta.type == SynergyType.TIER:
            self._draw_tier_resource(resource_meta, pos)

    def _draw_tier_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        tier = resource_meta.tier
        self.window_surface.blit(self.graphics[f"tier_{tier}_resource"], pos)

    def _draw_agent(self, pos: tuple[int, int]):
        """Draw agent at a specific pixel position"""

        self.window_surface.blit(self.graphics["agent_img"], pos)

    def _draw_hud(self, agent_score):
        """Draw HUD / score"""

        text = self.font.render(
            f"Score: {agent_score}", True, self.text_clr, self.background_clr
        )
        text_pos = (
            self._padding,
            self.window_size[1] - self.world_info_height - self._padding / 2,
        )
        self.window_surface.blit(text, text_pos)

    def _update(self, render_fps):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self.clock.tick(render_fps)

    def _process_quit_events(self) -> None | str:
        """Handle quitting / ESC"""

        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    pass
                if event.key == pygame.K_LEFT:
                    action = 'left'
                if event.key == pygame.K_DOWN:
                    action = 'down'
                if event.key == pygame.K_RIGHT:
                    action = 'right'
                if event.key == pygame.K_UP:
                    action = 'up'

        return action
