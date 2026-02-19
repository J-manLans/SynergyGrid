from enum import Enum
from numpy.random import Generator, default_rng


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
    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.starting_score = starting_score
        self.reset()

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
        # Will see at what part this becomes necessary
        # self.last_action = agent_action

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
        # Print current state to console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if [r, c] == self.agent_pos:
                    print(GridTile.AGENT, end=" ")
                elif [r, c] == self.resource_pos:
                    print(GridTile.RESOURCE, end=" ")
                else:
                    print(GridTile._FLOOR, end=" ")

            print()  # New line
        print()  # New line
