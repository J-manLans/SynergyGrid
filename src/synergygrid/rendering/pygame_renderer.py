import pygame
from os import path
import sys
import numpy as np
from synergygrid import AgentAction


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
        self.fps = fps
        self.__init_colors()

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

        # Load graphics
        ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "..", "..", ".."))

        file_name = path.join(ROOT_DIR, "assets/sprites/agent.png")
        img = pygame.image.load(file_name)
        self.agent_img = img

        file_name = path.join(ROOT_DIR, "assets/sprites/green-resource.png")
        img = pygame.image.load(file_name)
        self.green_resource_img = img

        file_name = path.join(ROOT_DIR, "assets/tiles/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = img

    # ================= #
    #        API        #
    # ================= #

    def render(
        self,
        agent_pos: list[int],
        resource_pos: list[np.int64],
        last_action: str | AgentAction,
    ) -> None:
        """
        Draws the game window and all its content, updates and limits the fps.

        Also catches user events (clicking the X top right or hitting ESC) for closing the game.
        """

        self.__process_events()
        self.window_surface.fill(self.background_clr)

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                # Draw resource
                if [r, c] == resource_pos:
                    self.window_surface.blit(self.green_resource_img, pos)

                # Draw agent
                if [r, c] == agent_pos:
                    self.window_surface.blit(self.agent_img, pos)

        # Draw a box on the bottom signifying the agents action
        # Just for demonstration of how to work with Pygame, will be removed later
        text = self.font.render(
            f"Action: {last_action}", True, self.text_clr, self.background_clr
        )
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text, text_pos)

        self.__update()

    # ================= #
    #      Helpers      #
    # ================= #

    def __init_colors(self) -> None:
        """Sets all the colors the game uses"""

        self.background_clr = (45, 29, 29)
        self.text_clr = (255, 246, 213)

    def __process_events(self) -> None:
        """Process user events and key presses"""

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

    def __update(self):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self.clock.tick(self.fps)
