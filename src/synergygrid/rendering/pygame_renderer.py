import pygame
from os import path
import sys
import numpy as np
from synergygrid.core import AgentAction, ResourceMeta, ResourceCategory, DirectType


class PygameRenderer:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows=5, grid_cols=5, fps=4) -> None:
        """
        Initializes the Pygame renderer.

        Sets:
        - Size of grid world
        - Fps
        - Game colors
        - Game font
        - Graphic elements
        """

        pygame.init()  # Initialize pygame
        pygame.display.init()  # initialize the display module
        self.clock = pygame.time.Clock()  # Game clock

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.step_fps = fps
        self._init_colors()

        # Default font
        self.font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.font.get_height()

        # Sizes of each cell in the grid
        self.cell_width = 64
        self.cell_height = 64

        # Define game window size (width, height)
        self.window_size = (
            self.cell_width * self.grid_cols,
            self.cell_height * self.grid_rows + self.action_info_height,
        )

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)

        self._load_graphics()

    # ================= #
    #        API        #
    # ================= #

    def render(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_types: list[ResourceMeta],
        agent_score: int,
    ) -> None:
        """
        Draws the game window and all its content, updates and limits the fps.

        Also catches user events (clicking the X top right or hitting ESC) for closing the game.
        """

        self.window_surface.fill(self.background_clr)

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        if [r, c] == resource_positions[i]:
                            self._draw_resource(resource_types[i], pos)

                if [r, c] == agent_pos:
                    # Draw agent
                    self.window_surface.blit(self.agent_img, pos)

        # Draw a box on the bottom signifying the agents action
        # Just for demonstration of how to work with Pygame, will be removed later
        text = self.font.render(
            f"Score: {agent_score}", True, self.text_clr, self.background_clr
        )
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text, text_pos)

        self._update()
        self._process_events()

    # ------------------------- #
    #      New Render Test      #
    # ------------------------- #

    def render2(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_types: list[ResourceMeta],
        agent_score: int,
    ) -> None:
        """
        Blocking render: animate agent over a fixed number of frames while everything
        else stays visually static. Simple linear interpolation.
        """

        step_duration = 0.5
        frames_per_step = 10
        render_fps = int(frames_per_step / step_duration)

        # --- Snapshot resources ---
        snapshot_resource_positions = [
            (int(p[0]), int(p[1])) for p in resource_positions
        ]
        snapshot_resource_types = list(resource_types)

        # --- Track last cell ---
        target = [int(agent_pos[0]), int(agent_pos[1])]
        if not hasattr(self, "_last_agent_cell"):
            self._last_agent_cell = [target[0], target[1]]
        last = self._last_agent_cell

        # --- Clamp diagonal moves ---
        dr = target[0] - last[0]
        dc = target[1] - last[1]
        if dr != 0 and dc != 0:
            if abs(dr) >= abs(dc):
                target = [last[0] + (1 if dr > 0 else -1), last[1]]
            else:
                target = [last[0], last[1] + (1 if dc > 0 else -1)]

        # --- Animate agent if moving ---
        if target != last:
            for frame in range(frames_per_step + 1):
                t = frame / float(frames_per_step)
                x_render = ((1.0 - t) * last[1] + t * target[1]) * self.cell_width
                y_render = ((1.0 - t) * last[0] + t * target[0]) * self.cell_height

                self.window_surface.fill(self.background_clr)
                self._draw_floor_and_resources(
                    snapshot_resource_positions,
                    snapshot_resource_types,
                    is_active_statuses,
                )
                self._draw_agent(x_render, y_render)
                self._draw_hud(agent_score)

                pygame.display.update()
                self._process_quit_events()
                self.clock.tick(render_fps)

            # snap logical cell
            self._last_agent_cell = [target[0], target[1]]

        else:
            # no movement, single frame
            x_render = target[1] * self.cell_width
            y_render = target[0] * self.cell_height

            self.window_surface.fill(self.background_clr)
            self._draw_floor_and_resources(
                snapshot_resource_positions, snapshot_resource_types, is_active_statuses
            )
            self._draw_agent(x_render, y_render)
            self._draw_hud(agent_score)

            pygame.display.update()
            self._process_quit_events()
            self.clock.tick(render_fps)

    # ------------------------- #
    #      Helper Methods       #
    # ------------------------- #

    def _draw_floor_and_resources(
        self, snapshot_resource_positions, snapshot_resource_types, is_active_statuses
    ):
        """Draw floor tiles and resources (snapshot)"""
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        res_r, res_c = snapshot_resource_positions[i]
                        if (r, c) == (res_r, res_c):
                            self._draw_resource(snapshot_resource_types[i], pos)

    def _draw_agent(self, x_pixel, y_pixel):
        """Draw agent at a specific pixel position"""
        self.window_surface.blit(self.agent_img, (int(x_pixel), int(y_pixel)))

    def _draw_hud(self, agent_score):
        """Draw HUD / score"""
        text = self.font.render(
            f"Score: {agent_score}", True, self.text_clr, self.background_clr
        )
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text, text_pos)

    def _process_quit_events(self):
        """Handle quitting / ESC"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init ===#

    def _init_colors(self) -> None:
        """Sets all the colors the game uses"""

        self.background_clr = (26, 26, 26)
        self.text_clr = (0, 210, 210)

    def _load_graphics(self):
        # Load graphics
        ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "..", "..", ".."))

        file_name = path.join(ROOT_DIR, "assets/sprites/agent.png")
        img = pygame.image.load(file_name)
        self.agent_img = img

        file_name = path.join(ROOT_DIR, "assets/sprites/positive_resource.png")
        img = pygame.image.load(file_name)
        self.positive_resource = img

        file_name = path.join(ROOT_DIR, "assets/sprites/negative_resource.png")
        img = pygame.image.load(file_name)
        self.negative_resource = img

        file_name = path.join(ROOT_DIR, "assets/tiles/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = img

    # === API ===#

    def _draw_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        if resource_meta.category == ResourceCategory.DIRECT:
            if resource_meta.subtype == DirectType.POSITIVE:
                self.window_surface.blit(self.positive_resource, pos)
            elif resource_meta.subtype == DirectType.NEGATIVE:
                self.window_surface.blit(self.negative_resource, pos)

    def _process_events(self) -> None:
        """Process user events and key presses"""

        waiting = False  # HACK: Flip to True to activate this part
        while waiting:
            for event in pygame.event.get():
                # User clicked on X at the top right corner of window
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # User hit escape
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    # TODO: Pause the game until the user hits space. Right now it's a step
                    # function, so need to rework this a little bit, but good for debugging as of
                    # now
                    elif event.key == pygame.K_SPACE:
                        waiting = False

            self.clock.tick(self.step_fps)

    def _update(self):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self.clock.tick(self.step_fps)
