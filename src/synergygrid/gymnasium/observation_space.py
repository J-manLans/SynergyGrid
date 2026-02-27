from numpy.typing import NDArray
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
from synergygrid.core import DirectType, ResourceCategory, BaseResource



class ObservationHandler:
    def __init__(self, world, grid_rows, grid_cols, _max_steps, _step_count_down):
        self._world = world
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._max_steps = _max_steps
        self._step_count_down = _step_count_down
        pass

    def _setup_obs_space(self) -> Box:
        # TODO: fix obs space
        """
        Gymnasium requires an observation space definition. Here we represent the state as a flat
        vector. The space is used by Gymnasium to validate observations returned by reset() and step().

        Set up:
        - self._raw_low / self._raw_high: the original raw ranges (used for normalization)
        - self._sentinel_mask: boolean mask that indicates "absent" resources
        - self.observation_space: the normalized observation space that agents will see
        (0..1 for active features, -1 for absent resource fields)
        """
        # original raw bounds — match _get_observation()
        raw_low, raw_high = self._build_observation_bounds(False)

        # store raw bounds for use in normalize_obs()
        self._raw_high = raw_high
        # Absent resource mask: True where low == -1.0 (these fields mean "absent" when -1)
        # This mask will be used to keep -1 as a special value instead of normalizing it.
        self._resource_mask = raw_low == -1.0

        # normalized bounds — match _normalize_obs()
        # inactive resources keep -1 as a valid "low" value; active features map to 0..1
        low_norm, high_norm = self._build_observation_bounds(True)

        return spaces.Box(
            low=low_norm, high=high_norm, dtype=np.float32
        )

    def _build_observation_bounds(
            self, normalized: bool
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # TODO: fix obs space
        # TODO: group logical units together like all agents data and resource data together
        if normalized:
            agent_and_steps_low = 0.0
            no_resource_yx = -1.0
            r_tier_low = -1.0
            r_cat_low = 0.0
            r_type_low = 0.0
            r_timer_low = 0.0

            max_steps = 1.0
            max_row = 1.0
            max_col = 1.0
            max_r_cat = 1.0
            max_r_type = 1.0
            max_r_timer = 1.0
            max_r_tier = 1.0
        else:
            agent_and_steps_low = 0
            no_resource_yx = -1
            r_tier_low = -1
            r_cat_low = 0
            r_type_low = 0
            r_timer_low = 0

            max_steps = self._max_steps
            max_row = self.grid_rows - 1
            max_col = self.grid_cols - 1
            max_r_cat = len(ResourceCategory) - 1
            max_r_type = len(DirectType) - 1
            max_r_timer = (self.grid_rows - 1) + (self.grid_cols - 1)
            max_r_tier = self._world.max_tier

        # episode length and agent position
        low = [agent_and_steps_low] * 3
        # resource data
        low.extend(
            [
                no_resource_yx,  # row
                no_resource_yx,  # col
                r_cat_low,
                r_type_low,
                r_timer_low,
                r_tier_low
            ]
            * len(self._world._ALL_RESOURCES)
        )
        # episode length and agent position
        high = [max_steps, max_row, max_col]
        # resource data
        high.extend(
            [
                max_row,
                max_col,
                max_r_cat,
                max_r_type,
                max_r_timer,
                max_r_tier
            ]
            * len(self._world._ALL_RESOURCES)
        )

        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)

        if low_arr.shape != high_arr.shape:
            raise ValueError(
                f"low/high shape mismatch: {low_arr.shape} != {high_arr.shape}"
            )
        if np.any(high_arr <= low_arr):
            raise ValueError(
                "All high bounds must be greater than low bounds for raw ranges"
            )

        return low_arr, high_arr

    def _get_observation(self) -> NDArray[np.float32]:
        # TODO: fix obs space
        """
        Build a flat observation vector dynamically based on max_active_resources.
        - Uses sentinel -1 for absent resource position/type and 0 for absent timers.
        """
        # Get step + agent info
        agent_row, agent_col = self._world._agent.position
        obs: list[float] = [self._step_count_down, agent_row, agent_col]

        # Cache resource info
        positions = self._world.get_resource_positions(False)
        types = self._world.get_resource_types(False)
        timers = self._world.get_resource_timers(False)
        active = self._world.get_resource_is_active_status(False)
        tiers = self._world.get_resource_tiers(False)

        # For each resource slot, append (row, col, type, timer) or absent values
        for i in range(len(positions)):
            if i < len(active) and active[i]:
                # Resource is active: extract real values
                pos = positions[i]
                r_category = types[i].category.value
                r_type = types[i].type.value
                r_timer = timers[i].remaining
                r_tier = tiers[i]
                obs.extend(
                    [
                        float(pos[0]),
                        float(pos[1]),
                        float(r_category),
                        float(r_type),
                        float(r_timer),
                        float(r_tier),
                    ]
                )
            else:
                # Resource inactive or slot unused: sentinel values
                obs.extend([-1, -1, 0, 0, 0, -1])

        return np.array(obs, dtype=np.float32)

    def _normalize_obs(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        # TODO: fix obs space
        """
        Normalize observation to 0..1 while preserving sentinel values (-1) for absent resources.

        :param obs: raw observation array (shape: 15,) from _get_observation()
        Returns:
            normalized_obs: float32 np.array (15,) where:
            - regular features scaled 0..1
            - sentinel fields are -1.0 when absent, otherwise scaled 0..1
        """

        # Prepare output array
        normalized_obs = np.empty_like(obs, dtype=np.float32)

        # Create resource and non-resource indices
        resource_idx = np.where(self._resource_mask)[0]
        non_resource_idx = np.where(~self._resource_mask)[0]

        # Prep the non-resource indices
        normalized_obs[non_resource_idx] = (
            obs[non_resource_idx] / self._raw_high[non_resource_idx]
        )

        # If resource absent, keep -1, otherwise — normalize
        resource_values = obs[resource_idx]

        absent_mask = resource_values == -1.0
        present_mask = ~absent_mask

        # Keep absent resources
        normalized_obs[resource_idx[absent_mask]] = -1.0

        # Normalize only present ones
        normalized_obs[resource_idx[present_mask]] = (
            resource_values[present_mask] / self._raw_high[resource_idx[present_mask]]
        )

        return normalized_obs