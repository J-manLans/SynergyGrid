from synergygrid.core import (
    AgentAction,
    SynergyAgent,
    BaseResource,
    PositiveResource,
    NegativeResource,
)
import numpy as np
from numpy.random import Generator, default_rng


class GridWorld:
    RESOURCES: list[BaseResource]

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int, grid_cols: int, starting_score: int):
        """
        Initializes the grid world. Defines the game world's size and initializes the agent and resource.
        """

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent = SynergyAgent(grid_rows, grid_cols, starting_score=starting_score)

        self.RESOURCES = [
            PositiveResource((grid_rows, grid_cols)),
            NegativeResource((grid_rows, grid_cols)),
        ]

    def reset(self, starting_score: int, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset(starting_score)  # Initialize Agents starting position

        if rng == None:
            rng = default_rng()

        self.rng = rng

        # Initialize the resource's position
        self._spawn_random_resource()

    # ================= #
    #        API        #
    # ================= #

    def perform_agent_action(self, agent_action: AgentAction) -> int:
        """
        Perform an action through the agent and if the resource isn't consumed the agents position is compared to the resource's. If it is, consume the resource and store the reward, then return the resources consumed status, the agents score (since one move cost one score) and optional reward.

        :param agent_action: the action the agent will perform
        """

        reward = 0
        self.agent.perform_action(agent_action)

        if not self.resource.consumed:
            if self.resource.timer.tick():
                self.resource.deplete_resource()
                self._set_respawn_timer()
            elif self.agent.position == self.resource.position:
                reward = self.agent.consume_resource(self.resource)
        else:
            self._set_respawn_timer()

        return reward

    # ================= #
    #      Helpers      #
    # ================= #

    def _spawn_random_resource(self):
        resource_index = self.rng.integers(0, len(self.RESOURCES))
        self.resource = self.RESOURCES[resource_index]
        self.resource.spawn(self.rng)

    def _set_respawn_timer(self):
        self._ensure_spawn_timer()
        self._update_spawn_timer()

    def _ensure_spawn_timer(self) -> None:
        if not self.resource.timer.is_set():
            self.resource.timer.set(int(self.rng.integers(2, 7)))

    def _update_spawn_timer(self) -> None:
        if self.resource.timer.tick():
            self._spawn_random_resource()
