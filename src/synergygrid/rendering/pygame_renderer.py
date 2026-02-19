import pygame
from os import path
import sys


class PygameRenderer:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows=5, grid_cols=5, fps=4):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fps = fps
        self.last_action = ''

        self.__init_colors()  # Initialize colors
        pygame.init()  # Initialize pygame
        pygame.display.init()  # initialize the display module

        self.clock = pygame.time.Clock()  # Game clock

        # Default font
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_width = 64
        self.cell_height = 64
        self.cell_size = (self.cell_width, self.cell_height)

        # Define game window size (width, height)
        self.window_size = (
            self.cell_width * self.grid_cols,
            self.cell_height * self.grid_rows + self.action_info_height,
        )

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)

        # Load & resize sprites
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

    def __init_colors(self):
        self.background_clr = (45, 29, 29)
        self.text_clr = (255, 246, 213)

    # ================= #
    #       Logic       #
    # ================= #

    def render(self):
        self._process_events()
        self.window_surface.fill(self.background_clr)

        # Draw the graphics with pygame
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if [r, c] == self.resource_pos:
                    self.window_surface.blit(self.green_resource_img, pos)

                if [r, c] == self.agent_pos:
                    self.window_surface.blit(self.agent_img, pos)

        text = self.action_font.render(
            f"Action: {self.last_action}", True, self.text_clr, self.background_clr
        )
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text, text_pos)

        pygame.display.update()
        self.clock.tick(self.fps)  # Limits fps

    def _process_events(self):
        """Process user events, key presses"""

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

    # ================= #
    #      Setters      #
    # ================= #

    def set_agent_pos(self, agent_pos):
        """Set the agent's position."""
        self.agent_pos = agent_pos

    def set_resource_pos(self, resource_pos):
        """Set the resource's position."""
        self.resource_pos = resource_pos

    def set_last_action(self, action):
        """Set the agent's last action."""
        self.last_action = action
