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
    _INACTIVE_RESOURCES: list[BaseResource] = []
    _ALL_RESOURCES: Final[list[BaseResource]]
    _ACTIVE_RESOURCES: list[BaseResource] = []

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, max_active_resources: int, grid_rows: int, grid_cols: int):
        """
        Initializes the grid world. Defines the game world's size and initializes the agent and resource.
        """
        self.max_active_resources = max_active_resources
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent = SynergyAgent(grid_rows, grid_cols)

        self._ALL_RESOURCES = self._create_resources(self.max_active_resources)

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset()  # Initialize Agents starting position

        self._ACTIVE_RESOURCES.clear()
        self._INACTIVE_RESOURCES.clear()
        self._INACTIVE_RESOURCES = list(self._ALL_RESOURCES)
        for resource in self._ALL_RESOURCES:
            resource.reset()

        if rng == None:
            rng = default_rng()

        self.rng = rng

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
                    resource.deplete_resource()
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
        self, max_active_resources: int, ratio=(0.75, 0.25)
    ) -> list[BaseResource]:
        # TODO: the ratio works for two resource types, need to look over this when more is added
        n_pos = max(1, int(max_active_resources * ratio[0]))
        n_neg = max_active_resources - n_pos

        resources = []
        for _ in range(n_pos):
            resources.append(PositiveResource((self.grid_rows, self.grid_cols)))
        for _ in range(n_neg):
            resources.append(NegativeResource((self.grid_rows, self.grid_cols)))

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
        available_resources = [
            r for r in self._INACTIVE_RESOURCES if r.timer.is_completed()
        ]

        if (
            len(available_resources) > 0
            and len(self._ACTIVE_RESOURCES) < self.max_active_resources
        ):
            resource_idx = self.rng.integers(0, len(available_resources))
            resource = self._INACTIVE_RESOURCES.pop(resource_idx)

            while True:
                position = [
                    self.rng.integers(0, self.grid_rows),
                    self.rng.integers(0, self.grid_cols),
                ]

                if self._empty_spawn_cell(position):
                    resource.spawn(position)
                    self._ACTIVE_RESOURCES.append(resource)
                    break

    def _empty_spawn_cell(self, position: list[np.int64]) -> bool:
        # Check against agent
        if position == self.agent.position:
            return False

        # If there are no active resources we can spawn right away
        if len(self._ACTIVE_RESOURCES) == 0:
            return True

        # Check against all active resources
        for r in self._ACTIVE_RESOURCES:
            if position == r.position:
                return False

        return True
