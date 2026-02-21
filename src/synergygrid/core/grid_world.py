from synergygrid.core import (
    AgentAction,
    SynergyAgent,
    BaseResource,
    PositiveResource,
    NegativeResource
)
import numpy as np
from numpy.random import Generator, default_rng


# TODO: fix re-spawning mechanic
class GridWorld:
    RESOURCES: list[BaseResource]

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

        self.RESOURCES = [
            PositiveResource((grid_rows, grid_cols)),
            NegativeResource((grid_rows, grid_cols))
        ]

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

    def perform_agent_action(self, agent_action: AgentAction) -> tuple[bool, int, int]:
        """
        Perform an action through the agent and if the resource isn't consumed the agents position is compared to the resource's. If it is, consume the resource and store the reward, then return the resources consumed status, the agents score (since one move cost one score) and optional reward.

        :param agent_action: the action the agent will perform
        """

        reward = 0
        self.agent.perform_action(agent_action)

        if not self.resource.present:
            if self.resource.timer.tick():
                self.resource.deplete_resource()
            if self.agent.position == self.resource.position:
                reward = self.agent.consume_resource(self.resource)
        else:
            self._ensure_spawn_timer()
            self._update_spawn_timer(self.rng)

        return self.resource.present, self.agent.score, reward

    # === Getters === #

    def get_agent_pos(self) -> list[int]:
        return self.agent.position

    def get_resource_pos(self) -> list[np.int64]:
        return self.resource.position

    # ================= #
    #      Helpers      #
    # ================= #

    def _ensure_spawn_timer(self) -> None:
        if not self.resource.timer.is_set():
            self.resource.timer.set(int(self.rng.integers(2, 7)))

    def _update_spawn_timer(self, rng: Generator) -> None:
        if self.resource.timer.tick():
            self.resource.spawn(rng)
