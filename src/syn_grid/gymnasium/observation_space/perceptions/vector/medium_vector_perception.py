from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.config.models import PerceptionConf
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class MediumVectorPerception(BasePerception):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, conf: PerceptionConf, orbs: int) -> None:
        super().__init__(conf, orbs)

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None: ...

    def setup_obs_space(self) -> spaces.Space:
        self._max_vals = []
        orb_data = []

        self._max_vals.extend(self._get_max_droid_positions())
        orb_data.extend(self._get_max_droid_positions())
        orb_data.extend(self._get_max_orb_identity())
        self._num_orb_slots = len(orb_data)
        self._max_vals.extend(orb_data * self._ORBS_IN_ENV)

        self._SHAPE = len(self._max_vals)

        low = np.full(self._SHAPE, -1.0, dtype=np.float32)
        low[0:2] = 0.0

        return spaces.Box(
            low=low,
            high=1.0,
            shape=(self._SHAPE,),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        obs = np.full(self._SHAPE, -1.0, dtype=np.float32)
        obs_index = 0

        # Droid data
        droid_y, droid_x = state.DROID.position

        obs[0] = droid_y / self._max_vals[obs_index]
        obs_index += 1
        obs[1] = droid_x / self._max_vals[1]
        obs_index += 1

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                orb_y, orb_x = orb.position

                obs[obs_index] = orb_y / self._max_vals[obs_index]
                obs_index += 1
                obs[obs_index] = orb_x / self._max_vals[obs_index]
                obs_index += 1
                obs[obs_index] = orb.META.CATEGORY.value / self._max_vals[obs_index]
                obs_index += 1
                obs[obs_index] = orb.META.TYPE.value / self._max_vals[obs_index]
                obs_index += 1
                obs[obs_index] = orb.META.TIER / self._max_vals[obs_index]
                obs_index += 1
            else:
                obs_index += self._num_orb_slots

        return obs
