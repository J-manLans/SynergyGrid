from enum import Enum
from numpy.random import Generator, default_rng
import pygame
import sys
from os import path


# Actions the Agent is capable of performing i.e. go in a certain direction
class AgentAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


# The SynergyGrid is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    _FLOOR = 0
    AGENT = 1
    RESOURCE = 2

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        return self.name[:1]


class GridWorld:
    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10, fps=1):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.starting_score = starting_score
        self.reset()

        self.fps = fps
        self.last_action = ""
        self.__init_pygame()

    def __init_pygame(self):
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

    def reset(self, rng: Generator | None = None):
        if rng is None:
            rng = default_rng()
        # Initialize Agents starting position
        self.agent_pos = [2, 2]

        self.resource_pos = [
            rng.integers(1, self.grid_rows),
            rng.integers(1, self.grid_cols),
        ]

    def perform_action(self, agent_action: AgentAction) -> bool:
        self.last_action = agent_action

        # Move Robot to the next cell
        if agent_action == AgentAction.LEFT:
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        elif agent_action == AgentAction.RIGHT:
            if self.agent_pos[1] < self.grid_cols - 1:
                self.agent_pos[1] += 1
        elif agent_action == AgentAction.UP:
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
        elif agent_action == AgentAction.DOWN:
            if self.agent_pos[0] < self.grid_rows - 1:
                self.agent_pos[0] += 1

        # Return true if Agent reaches resource
        return self.agent_pos == self.resource_pos

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
        # Process user events, key presses
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


# For testing the graphics, remove when working
if __name__ == "__main__":
    grid_world = GridWorld()
    grid_world.render()
    rng = default_rng()

    while True:
        actions = list(AgentAction)
        rand_index = rng.integers(0, len(actions))
        rand_action = actions[rand_index]

        grid_world.perform_action(rand_action)
        grid_world.render()
