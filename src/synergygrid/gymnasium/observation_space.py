from numpy.typing import NDArray
import numpy as np
from typing import Any
from gymnasium import spaces
from gymnasium.spaces import Dict
from synergygrid.core import (
    GridWorld,
    ResourceCategory,
    BaseResource,
    DirectType,
    SynergyType,
)


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        world: GridWorld,
        grid_rows: int,
        grid_cols: int,
        _max_steps: int,
        _step_count_down: int,
    ):
        self._world = world
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._max_steps = _max_steps
        self._step_count_down = _step_count_down

    # ================== #
    #       API      #
    # ================== #

    def setup_obs_space(self) -> Dict:
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
        agent_raw_low, self._agent_raw_high = self._build_agent_box_bounds(False)
        resource_raw_low, self._resource_raw_high = self._build_resource_box_bounds(
            False
        )

        # store raw bounds for use in normalize_obs()
        self._raw_high = np.concatenate(
            [self._agent_raw_high, self._resource_raw_high.flatten()]
        )
        # Absent resource mask: True where low == -1.0 (these fields mean "absent" when -1)
        # This mask will be used to keep -1 as a special value instead of normalizing it.
        # TODO: can be removed when the new normalize method is completed
        raw_low = np.concatenate([agent_raw_low, resource_raw_low.flatten()])
        self._resource_mask = raw_low == -1.0
        self._agent_resource_mask = agent_raw_low == -1.0

        # normalized bounds — match _normalize_obs()
        # inactive resources keep -1 as a valid "low" value; active features map to 0..1
        self._agent_low_norm, self._agent_high_norm = self._build_agent_box_bounds(True)
        self._resource_low_norm, self._resource_high_norm = (
            self._build_resource_box_bounds(True)
        )

        self.observation_space = spaces.Dict(
            {
                "agent data": spaces.Box(
                    # steps, row, col, current tier chain
                    low=self._agent_low_norm,
                    high=self._agent_high_norm,
                    dtype=np.float32,
                ),
                "resources data": spaces.Box(
                    # row, col, timer, tier
                    low=self._resource_low_norm,
                    high=self._resource_high_norm,
                    dtype=np.float32,
                ),
                "resources type": spaces.MultiDiscrete(
                    np.array(
                        [
                            [
                                len(ResourceCategory),
                                max(len(SynergyType), len(DirectType)),
                            ]
                        ]
                        * len(self._world._ALL_RESOURCES)
                    ),
                    dtype=np.int64
                )
            }
        )

        return self.observation_space

    def get_observation(self) -> dict[str, Any]:
        agent_row, agent_col = self._world._agent.position

        # ---- Agent ----
        agent_data = np.array(
            [
                # TODO: don't forget to fix all values that used to be in environment.py like
                # _step_count_down — it has to be updated each step
                float(self._step_count_down),
                float(agent_row),
                float(agent_col),
                float(len(BaseResource._chained_tiers) - 1),
            ],
            dtype=np.float32,
        )

        # ---- Resources ----
        N = len(self._world._ALL_RESOURCES)

        resource_data = np.tile(np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32), (N, 1))
        resource_type = np.zeros((N, 2), dtype=np.int64)

        active = self._world.get_resource_is_active_status(False)
        positions = self._world.get_resource_positions(False)
        timers = self._world.get_resource_timers(False)
        tiers = self._world.get_resource_tiers(False)
        types = self._world.get_resource_types(False)

        for i in range(N):
            r_category = types[i].category.value
            r_type = types[i].type.value
            resource_type[i] = [
                int(r_category),
                int(r_type),
            ]
            if i < len(active) and active[i]:
                pos = positions[i]
                r_timer = timers[i].remaining
                r_tier = tiers[i]

                resource_data[i] = [
                    float(pos[0]),
                    float(pos[1]),
                    float(r_timer),
                    float(r_tier),
                ]

        return {
            "agent data": agent_data,
            "resources data": resource_data,
            "resources type": resource_type,
        }

    def normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        # --- Agent --- #
        agent = obs["agent data"].astype(np.float32)
        norm_agent = self._normalize_obs_fields(
            agent, self._agent_resource_mask, self._agent_raw_high
        )

        # --- Resources --- #
        norm_res = obs["resources data"].astype(np.float32)
        for i in range(norm_res.shape[0]):
            row = norm_res[i]
            absent_mask = row == -1.0

            norm_row = self._normalize_obs_fields(
                row, absent_mask, self._resource_raw_high[i]
            )
            norm_res[i] = norm_row

        # --- Types unchanged --- #
        res_type = obs["resources type"]

        return {
            "agent data": norm_agent,
            "resources data": norm_res,
            "resources type": res_type,
        }

    # ================== #
    #       Helpers      #
    # ================== #

    def _build_agent_box_bounds(
        self, normalized: bool
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if normalized:
            min_steps = min_row = min_col = 0.0
            min_tier_chain = -1.0

            max_steps = max_row = max_col = max_tier_chain = 1.0
        else:
            min_steps = min_row = min_col = 0
            min_tier_chain = -1

            max_steps = self._max_steps
            max_row = self.grid_rows - 1
            max_col = self.grid_cols - 1
            max_tier_chain = self._world.max_tier

        low = [min_steps, min_row, min_col, min_tier_chain]
        high = [max_steps, max_row, max_col, max_tier_chain]

        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)

        self._control_arrays(low_arr, high_arr)

        return low_arr, high_arr

    def _build_resource_box_bounds(
        self, normalized: bool
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if normalized:
            no_resource_yx = -1.0
            r_timer_low = 0.0
            r_tier_low = -1.0

            max_row = 1.0
            max_col = 1.0
            max_r_timer = 1.0
            max_r_tier = 1.0
        else:
            no_resource_yx = -1
            r_timer_low = 0
            r_tier_low = -1

            max_row = self.grid_rows - 1
            max_col = self.grid_cols - 1
            max_r_timer = (self.grid_rows - 1) + (self.grid_cols - 1)
            max_r_tier = self._world.max_tier

        N = len(self._world._ALL_RESOURCES)
        low = np.tile([no_resource_yx, no_resource_yx, r_timer_low, r_tier_low], (N, 1))
        high = np.tile([max_row, max_col, max_r_timer, max_r_tier], (N, 1))

        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)

        self._control_arrays(low_arr, high_arr)

        return low_arr, high_arr

    def _control_arrays(self, low_arr, high_arr):
        if low_arr.shape != high_arr.shape:
            raise ValueError(
                f"low/high shape mismatch: {low_arr.shape} != {high_arr.shape}"
            )
        if np.any(high_arr <= low_arr):
            raise ValueError(
                "All high bounds must be greater than low bounds for raw ranges"
            )

    def _normalize_obs_fields(
        self, array: NDArray[np.float32], mask, scale: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Normalize observation to 0..1 while preserving sentinel values (-1) for absent resources.

        :param obs: raw observation array (shape: 15,) from _get_observation()
        Returns:
            normalized_obs: float32 np.array (15,) where:
            - regular features scaled 0..1
            - sentinel fields are -1.0 when absent, otherwise scaled 0..1
        """

        # Prepare output array
        norm_array = np.empty_like(array, dtype=np.float32)

        # Indices for "regular" entries and "masked/special" entries
        masked_idx = np.where(mask)[0]
        regular_idx = np.where(~mask)[0]

        # Normalize regular entries element wise
        norm_array[regular_idx] = array[regular_idx] / scale[regular_idx]

        # Handle masked/special entries
        special_values = array[masked_idx]

        absent_mask = special_values == -1.0
        present_mask = ~absent_mask

        # Keep sentinel values
        norm_array[masked_idx[absent_mask]] = -1.0

        # Normalize only present values
        norm_array[masked_idx[present_mask]] = (
            special_values[present_mask] / scale[masked_idx[present_mask]]
        )

        return norm_array
