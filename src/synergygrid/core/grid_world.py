from synergygrid.core import (
    AgentAction,
    SynergyAgent,
    BaseResourceTest,
    PositiveResource,
)
import numpy as np
from numpy.random import Generator, default_rng

#TODO: fix re-spawning mechanic
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

        self.RESOURCES = []
        self.RESOURCES.append(PositiveResource((grid_rows, grid_cols)))

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset()  # Initialize Agents starting position

        if rng == None:
            rng = default_rng()

        self.rng = rng

        # Initialize the resource's position
        self.resource = self.RESOURCES[rng.integers(0, len(self.RESOURCES))]
        self.resource.spawn(rng)

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_agent_action(
        self, agent_action: AgentAction
    ) -> tuple[bool, bool, int]:
        """
        Perform an action through the agent and if the resource isn't consumed the agents position is compared to the resource's. If it is, consume the resource and store the reward, then return the resources consumed status, the agents score (since one move cost one score) and optional reward.

        :param agent_action: the action the agent will perform
        """

        reward = 0
        self.agent.perform_action(agent_action)

        if not self.resource.consumed:
            if self.agent.pos == self.resource.pos:
                reward = self.agent.consume_resource(self.resource)
        else:
            self._ensure_spawn_timer()
            self._update_spawn_timer(self.rng)

        return self.resource.consumed, self.agent.score == 0, reward

    # === Getters === #

    def get_agent_pos(self) -> list[int]:
        return self.agent.pos

    def get_resource_pos(self) -> list[np.int64]:
        return self.resource.pos

    def get_last_action(self) -> str | AgentAction:
        return self.agent.last_action

    # ================= #
    #      Helpers      #
    # ================= #

    def _ensure_spawn_timer(self) -> None:
        if not self.resource.timer.is_set():
            self.resource.timer.set(int(self.rng.integers(5, 10)))

    def _update_spawn_timer(self, rng: Generator) -> None:
        if self.resource.timer.tick():
            self.resource.spawn(rng)