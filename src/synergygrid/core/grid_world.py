from enum import Enum
from numpy.random import Generator, default_rng
from synergygrid import PygameRenderer


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


class GridWorld:
    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10, fps=1):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.starting_score = starting_score
        self.renderer = PygameRenderer(grid_rows, grid_cols, fps)

        self.reset()

        self.last_action = ""


    def reset(self, rng: Generator | None = None):
        if rng is None:
            rng = default_rng()

        # Initialize Agents starting position
        self.agent_pos = [2, 2]
        self.renderer.set_agent_pos(self.agent_pos)

        # Initialize the resource's position
        self.resource_pos = [
            rng.integers(1, self.grid_rows),
            rng.integers(1, self.grid_cols),
        ]
        self.renderer.set_resource_pos(self.resource_pos)

    def perform_action(self, agent_action: AgentAction) -> bool:
        self.last_action = agent_action
        self.renderer.set_last_action(agent_action)

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

        self.renderer.set_agent_pos(self.agent_pos)

        # Return true if Agent reaches resource
        return self.agent_pos == self.resource_pos


# For testing the graphics, remove when working
if __name__ == "__main__":
    grid_world = GridWorld()
    grid_world.renderer.render()
    rng = default_rng()

    while True:
        actions = list(AgentAction)
        rand_index = rng.integers(0, len(actions))
        rand_action = actions[rand_index]

        grid_world.perform_action(rand_action)
        grid_world.renderer.render()
