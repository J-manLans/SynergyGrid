from synergygrid.core import (
    AgentAction,
    SynergyAgent,
    ResourceMeta,
    BaseResource,
    PositiveResource,
    NegativeResource,
)
import numpy as np
from numpy.random import Generator, default_rng
from typing import Final


# TODO: make multiple active resources possible (max 3)
class GridWorld:
    _INACTIVE_RESOURCES: list[BaseResource]
    _ALL_RESOURCES: list[BaseResource]
    _ACTIVE_RESOURCES: list[BaseResource] = []

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        max_active_resources: int,
        grid_rows: int,
        grid_cols: int,
        starting_score: int,
    ):
        """
        Initializes the grid world. Defines the game world's size and initializes the agent and resource.
        """
        self.max_active_resources = max_active_resources
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent = SynergyAgent(grid_rows, grid_cols, starting_score=starting_score)

    def reset(self, starting_score: int, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset(starting_score)  # Initialize Agents starting position

        if rng == None:
            rng = default_rng()

        self.rng = rng

        self._ALL_RESOURCES = self._create_resources(
            self.max_active_resources, (self.grid_rows, self.grid_cols)
        )
        self._INACTIVE_RESOURCES = list(self._ALL_RESOURCES)

        # Initialize the resource's position
        self._spawn_random_resource()

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_agent_action(self, agent_action: AgentAction) -> int:
        reward = 0
        self.agent.perform_action(agent_action)

        for resource in self._ALL_RESOURCES:
            if resource.is_active:
                if self._update_timer_and_check_if_completed(resource):
                    resource.deplete_resource(self.rng)
                    self._remove_resource(resource)
                elif self.agent.position == resource.position:
                    reward = self.agent.consume_resource(resource)
                    self._remove_resource(resource)
            else:
                if self._update_timer_and_check_if_completed(resource):
                    self._spawn_random_resource()

        return reward

    # === Getters === #

    def get_resource_positions(self) -> list[list[np.int64]]:
        return [r.position for r in self._ALL_RESOURCES]

    def get_resource_is_active_status(self) -> list[bool]:
        return [r.is_active for r in self._ALL_RESOURCES]

    def get_resource_types(self) -> list[ResourceMeta]:
        return [r.type for r in self._ALL_RESOURCES]

    def get_resource_timers(self) -> list[BaseResource.Timer]:
        return [r.timer for r in self._ALL_RESOURCES]

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _create_resources(
        self, max_active_resources, grid_shape: tuple[int, int], ratio=(0.75, 0.25)
    ) -> list[BaseResource]:
        # TODO: the ratio works for two resources, need to look over this when more is added
        n_pos = max(1, int(max_active_resources * ratio[0]))
        n_neg = max_active_resources - n_pos

        resources = []
        for _ in range(n_pos):
            resources.append(PositiveResource(grid_shape, self.rng))
        for _ in range(n_neg):
            resources.append(NegativeResource(grid_shape, self.rng))

        return resources

    # === API === #

    def _update_timer_and_check_if_completed(self, resource: BaseResource) -> bool:
        resource.timer.tick()
        return resource.timer.is_completed()

    # === Global === #

    def _remove_resource(self, resource: BaseResource):
        idx = self._ACTIVE_RESOURCES.index(resource)
        depleted = self._ACTIVE_RESOURCES.pop(idx)
        self._INACTIVE_RESOURCES.append(depleted)

    def _spawn_random_resource(self):
        available = len(self._INACTIVE_RESOURCES)

        if available > 0 and len(self._ACTIVE_RESOURCES) < self.max_active_resources:
            resource = self._INACTIVE_RESOURCES.pop(self.rng.integers(0, available))
            self._ACTIVE_RESOURCES.append(resource)

            resource.spawn()