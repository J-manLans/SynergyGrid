from syn_grid.config.models import GridWorldConf, DroidConf, NegativeConf, TierConf
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.resources.resource_meta import ResourceMeta
from syn_grid.core.resources.base_resource import BaseResource
from syn_grid.core.resources.direct.negative_resource import NegativeResource
from syn_grid.core.resources.synergy.tier_resource import TierResource

import numpy as np
from numpy.random import Generator, default_rng
from typing import Final


class GridWorld:
    ALL_RESOURCES: Final[list[BaseResource]]
    _inactive_resources: list[BaseResource] = []
    _active_resources: list[BaseResource] = []

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        conf: GridWorldConf,
        droid_conf: DroidConf,
        negative_resource_conf: NegativeConf,
        tier_resource_conf: TierConf,
    ):
        """
        Initializes the grid world. Defines the game world's size and initializes the agent and resources.
        """

        if conf.grid_rows < 1 or conf.grid_cols < 1:
            raise ValueError("grid_cols and grid_rows should be larger than 0")

        self._max_active_resources = conf.max_active_resources
        self.grid_rows = conf.grid_rows
        self.grid_cols = conf.grid_cols
        self.max_tier = conf.max_tier

        self.agent = SynergyDroid(droid_conf)

        self.ALL_RESOURCES = self._create_resources(
            negative_resource_conf, tier_resource_conf
        )

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the agent to its starting position and re-spawns the resource at a random location
        """

        self.agent.reset()  # Initialize Agents starting position

        self._active_resources.clear()
        self._inactive_resources.clear()
        self._inactive_resources = list(self.ALL_RESOURCES)
        for resource in self.ALL_RESOURCES:
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

    def perform_agent_action(self, agent_action: DroidAction) -> float:
        reward = self.agent.perform_action(agent_action)

        for resource in self.ALL_RESOURCES:
            if resource.is_active:
                if self._update_timer_and_return_is_completed(resource):
                    resource.deplete_resource()
                    self._remove_resource(resource)
                elif self.agent.position == resource.position:
                    reward = self.agent.consume_resource(resource)
                    self._remove_resource(resource)
            else:
                if self._update_timer_and_return_is_completed(resource):
                    self._spawn_random_resource()

        return reward

    # === Getters === #

    def get_resource_positions(self, only_active: bool) -> list[list[np.int64]]:
        if only_active:
            return [r.position for r in self._active_resources]

        return [r.position for r in self.ALL_RESOURCES]

    def get_resource_is_active_status(self, only_active: bool) -> list[bool]:
        if only_active:
            return [r.is_active for r in self._active_resources]

        return [r.is_active for r in self.ALL_RESOURCES]

    def get_resource_meta(self, only_active: bool) -> list[ResourceMeta]:
        if only_active:
            return [r.meta for r in self._active_resources]

        return [r.meta for r in self.ALL_RESOURCES]

    def get_resource_categories(self) -> list[int]:
        return [r.meta.category.value for r in self.ALL_RESOURCES]

    def get_resource_types(self) -> list[int]:
        return [r.meta.type.value for r in self.ALL_RESOURCES]

    def get_resource_life(self) -> list[int]:
        return [r.timer.remaining for r in self.ALL_RESOURCES]

    def get_resource_tiers(self) -> list[int]:
        return [r.meta.tier for r in self.ALL_RESOURCES]

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _create_resources(
        self, negative_resource_conf: NegativeConf, tier_resource_conf: TierConf
    ) -> list[BaseResource]:
        resources = []

        ratio = (0.75, 0.25)
        n_tier = self._compute_spawn_count(ratio[0])
        n_neg = self._compute_spawn_count(ratio[1])

        TierResource.MAX_TIER = self.max_tier
        BaseResource.set_life_span(self.grid_rows, self.grid_cols)
        for _ in range(n_neg):
            resources.append(NegativeResource(negative_resource_conf))
        for tier in range(0, self.max_tier + 1):
            for _ in range(n_tier):
                resources.append(TierResource(tier, tier_resource_conf))

        return resources

    def _compute_spawn_count(self, ratio: float) -> int:
        return max(1, int(self._max_active_resources * ratio + 0.5))

    # === API === #

    def _update_timer_and_return_is_completed(self, resource: BaseResource) -> bool:
        resource.timer.tick()
        return resource.timer.is_completed()

    def _remove_resource(self, resource: BaseResource):
        idx = self._active_resources.index(resource)
        depleted = self._active_resources.pop(idx)
        self._inactive_resources.append(depleted)

    # === Global === #

    def _spawn_random_resource(self):
        available_resources = [
            r for r in self._inactive_resources if r.timer.is_completed()
        ]

        if (
            len(available_resources) > 0
            and len(self._active_resources) < self._max_active_resources
        ):
            resource_idx = self.rng.integers(0, len(available_resources))
            resource = self._inactive_resources.pop(resource_idx)

            while True:
                position = [
                    self.rng.integers(0, self.grid_rows),
                    self.rng.integers(0, self.grid_cols),
                ]

                if self._empty_spawn_cell(position):
                    resource.spawn(position)
                    self._active_resources.append(resource)
                    break

    def _empty_spawn_cell(self, position: list[np.int64]) -> bool:
        # Check against agent
        if position == self.agent.position:
            return False

        # If there are no active resources we can spawn right away
        if len(self._active_resources) == 0:
            return True

        # Else check against all active resources
        for r in self._active_resources:
            if position == r.position:
                return False

        return True
