from synergygrid.core import (
    AgentAction,
    SynergyAgent,
    BaseResourceTest,
    PositiveResource,
)
import numpy as np
from numpy.random import Generator, default_rng


class GridWorld:
    RESOURCES: list[BaseResourceTest]

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10):
        """
        Initializes the grid world. Defines the game world's size and initializes the agent and resource.
        """

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent = SynergyAgent(grid_rows, grid_cols, starting_score=starting_score)

        # New shit
        self.RESOURCES = []
        self.RESOURCES.append(PositiveResource((grid_rows, grid_cols)))

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset()  # Initialize Agents starting position

        if rng == None:
            rng = default_rng()

        # Initialize the resource's position
        self.resource = self.RESOURCES[rng.integers(0, len(self.RESOURCES))]
        self.resource.spawn(rng)

    # ================= #
    #       API         #
    # ================= #

    # === Logic === #

    def perform_agent_action(
        self, agent_action: AgentAction
    ) -> tuple[bool, int]:
        """
        Perform an action through the agent and compare if its position is the same as the resource, if it is it returns True, otherwise False

        :param agent_action: the action the agent will perform
        """

        self.agent.perform_action(agent_action)

        # TODO: here somewhere lies the problem, it only flickers to true when agent is right on it,
        # it doesn't consume it, only hides it at that certain state. So some despawn mechanic
        # needs to be applied
        if self.agent.pos == self.resource.pos:
            return True, self.resource.consume()

        return False, 0

    # === Getters === #

    def get_agent_pos(self) -> list[int]:
        return self.agent.pos

    def get_resource_pos(self) -> list[np.int64]:
        return self.resource.pos

    def get_last_action(self) -> str | AgentAction:
        return self.agent.last_action
