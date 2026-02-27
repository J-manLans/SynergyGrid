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
        agent_raw_low, agent_raw_high = self._build_agent_box_bounds(False)
        resource_raw_low, resource_raw_high = self._build_resource_box_bounds(False)

        # store raw bounds for use in normalize_obs()
        self._raw_high = np.concatenate([agent_raw_high, resource_raw_high])
        # Absent resource mask: True where low == -1.0 (these fields mean "absent" when -1)
        # This mask will be used to keep -1 as a special value instead of normalizing it.
        raw_low = np.concatenate([agent_raw_low, resource_raw_low])
        self._resource_mask = raw_low == -1.0

        # normalized bounds — match _normalize_obs()
        # inactive resources keep -1 as a valid "low" value; active features map to 0..1
        agent_low_norm, agent_high_norm = self._build_agent_box_bounds(True)
        resource_low_norm, resource_high_norm = self._build_resource_box_bounds(True)

        return spaces.Dict({
            "agent": spaces.Dict({
                "data": spaces.Box(
                    # steps, row, col, current tier chain
                    agent_low_norm,
                    agent_high_norm,
                    dtype=np.float32,
                )}
            ),
            "resources": spaces.Dict({
                "data": spaces.Box(
                    # row, col, timer, tier
                    resource_low_norm,
                    resource_high_norm,
                    dtype=np.float32,
                ),
                "type": spaces.MultiDiscrete(
                    np.array(
                        [[
                            len(ResourceCategory),
                            max(len(SynergyType), len(DirectType)),
                        ]] * len(self._world._ALL_RESOURCES)
                    )
                )}
            )}
        )

    # def get_observation(self) -> NDArray[np.float32]:
    #     """
    #     Build a flat observation vector dynamically based on max_active_resources.
    #     - Uses sentinel -1 for absent resource position/type and 0 for absent timers.
    #     """
    #     # Get step + agent info
    #     agent_row, agent_col = self._world._agent.position
    #     obs: list[float] = [self._step_count_down, agent_row, agent_col]

    #     # Cache resource info
    #     positions = self._world.get_resource_positions(False)
    #     types = self._world.get_resource_types(False)
    #     timers = self._world.get_resource_timers(False)
    #     active = self._world.get_resource_is_active_status(False)
    #     tiers = self._world.get_resource_tiers(False)

    #     # For each resource slot, append (row, col, type, timer) or absent values
    #     for i in range(len(positions)):
    #         if i < len(active) and active[i]:
    #             # Resource is active: extract real values
    #             pos = positions[i]
    #             r_category = types[i].category.value
    #             r_type = types[i].type.value
    #             r_timer = timers[i].remaining
    #             r_tier = tiers[i]
    #             obs.extend(
    #                 [
    #                     float(pos[0]),
    #                     float(pos[1]),
    #                     float(r_category),
    #                     float(r_type),
    #                     float(r_timer),
    #                     float(r_tier),
    #                 ]
    #             )
    #         else:
    #             # Resource inactive or slot unused: sentinel values
    #             obs.extend([-1, -1, 0, 0, 0, -1])

    #     return np.array(obs, dtype=np.float32)

    def get_observation(self) -> dict[str, dict[str, Any]]:
        agent_row, agent_col = self._world._agent.position

        # ---- Agent ----
        agent_data = np.array(
            [
                float(self._step_count_down),
                float(agent_row),
                float(agent_col),
                float(len(BaseResource._chained_tiers)),
            ],
            dtype=np.float32,
        )

        # ---- Resources ----
        N = len(self._world._ALL_RESOURCES)

        resource_data = np.full((N, 4), -1.0, dtype=np.float32)
        resource_type = np.full((N, 2), -1, dtype=np.int64)

        active = self._world.get_resource_is_active_status(False)
        positions = self._world.get_resource_positions(False)
        timers = self._world.get_resource_timers(False)
        tiers = self._world.get_resource_tiers(False)
        types = self._world.get_resource_types(False)

        for i in range(N):
            if i < len(active) and active[i]:
                pos = positions[i]
                r_timer = timers[i].remaining
                r_tier = tiers[i]
                r_category = types[i].category.value
                r_type = types[i].type.value

                resource_data[i] = [
                    float(pos[0]),
                    float(pos[1]),
                    float(r_timer),
                    float(r_tier),
                ]

                resource_type[i] = [
                    int(r_category),
                    int(r_type),
                ]

        return {
            "agent": {
                "data": agent_data
            },
            "resources": {
                "data": resource_data,
                "type": resource_type,
            },
        }

    # def normalize_obs(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
    #     """
    #     Normalize observation to 0..1 while preserving sentinel values (-1) for absent resources.

    #     :param obs: raw observation array (shape: 15,) from _get_observation()
    #     Returns:
    #         normalized_obs: float32 np.array (15,) where:
    #         - regular features scaled 0..1
    #         - sentinel fields are -1.0 when absent, otherwise scaled 0..1
    #     """

    #     # Prepare output array
    #     normalized_obs = np.empty_like(obs, dtype=np.float32)

    #     # Create resource and non-resource indices
    #     resource_idx = np.where(self._resource_mask)[0]
    #     non_resource_idx = np.where(~self._resource_mask)[0]

    #     # Prep the non-resource indices
    #     normalized_obs[non_resource_idx] = (
    #         obs[non_resource_idx] / self._raw_high[non_resource_idx]
    #     )

    #     # If resource absent, keep -1, otherwise — normalize
    #     resource_values = obs[resource_idx]

    #     absent_mask = resource_values == -1.0
    #     present_mask = ~absent_mask

    #     # Keep absent resources
    #     normalized_obs[resource_idx[absent_mask]] = -1.0

    #     # Normalize only present ones
    #     normalized_obs[resource_idx[present_mask]] = (
    #         resource_values[present_mask] / self._raw_high[resource_idx[present_mask]]
    #     )

    #     return normalized_obs

    def normalize_obs(self, obs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """
        Normalize nested observation dict to 0..1 for Box fields while preserving sentinels.
        Expects obs in form:
        {"agent": {"data": np.ndarray(dtype=float32, shape=(A,))},
        "resources": {"data": np.ndarray(dtype=float32, shape=(N, F)),
                        "type": np.ndarray(dtype=int64, shape=(N, 2))}}
        Returns same structure with normalized float arrays and untouched int array.
        """

        # --- prepare bounds (compute & cache if not present) ---
        if not hasattr(self, "_agent_low_norm") or not hasattr(self, "_agent_high_norm"):
            a_low, a_high = self._build_agent_box_bounds(True)
            self._agent_low_norm = np.array(a_low, dtype=np.float32)
            self._agent_high_norm = np.array(a_high, dtype=np.float32)
        agent_low = self._agent_low_norm
        agent_high = self._agent_high_norm

        if not hasattr(self, "_resource_low_norm") or not hasattr(self, "_resource_high_norm"):
            r_low, r_high = self._build_resource_box_bounds(True)
            self._resource_low_norm = np.array(r_low, dtype=np.float32)
            self._resource_high_norm = np.array(r_high, dtype=np.float32)
        resource_low = self._resource_low_norm
        resource_high = self._resource_high_norm

        # --- agent ---
        agent_arr = np.asarray(obs["agent"]["data"], dtype=np.float32)
        denom = (agent_high - agent_low).astype(np.float32)
        denom[denom == 0.0] = 1.0  # avoid div0
        agent_norm = (agent_arr - agent_low) / denom
        # clip numerics to [0,1] for stability
        agent_norm = np.clip(agent_norm, 0.0, 1.0).astype(np.float32)

        # --- resources (preserve sentinel rows == all -1.0) ---
        res_arr = np.asarray(obs["resources"]["data"], dtype=np.float32)
        N = res_arr.shape[0]

        # ensure resource bounds have matching shape
        if resource_low.shape[0] != N:
            # broadcast/repeat if needed (keep minimal, but robust)
            resource_low = np.broadcast_to(resource_low, (N, resource_low.shape[-1]))
            resource_high = np.broadcast_to(resource_high, (N, resource_high.shape[-1]))

        res_norm = np.full_like(res_arr, -1.0, dtype=np.float32)

        # define absent rows as rows where all entries == -1.0 (your sentinel)
        absent_mask = np.all(np.isclose(res_arr, -1.0), axis=1)
        present_idx = np.where(~absent_mask)[0]

        # TODO: look through this part

        if present_idx.size > 0:
            denom = (resource_high - resource_low).astype(np.float32)
            denom[denom == 0.0] = 1.0
            # normalize only present rows (vectorized)
            res_norm[present_idx] = (res_arr[present_idx] - resource_low[present_idx]) / denom[present_idx]
            res_norm[present_idx] = np.clip(res_norm[present_idx], 0.0, 1.0)

        # --- types: keep as-is, ensure integer dtype ---
        types_arr = np.asarray(obs["resources"]["type"], dtype=np.int64)

        return {
            "agent": {"data": agent_norm},
            "resources": {"data": res_norm, "type": types_arr},
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

        low = [
            no_resource_yx, no_resource_yx, r_timer_low, r_tier_low,
        ] * len(self._world._ALL_RESOURCES)
        high = [
            max_row, max_col, max_r_timer, max_r_tier
        ] * len(self._world._ALL_RESOURCES)

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
