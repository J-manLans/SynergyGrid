from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models import MediumDifficultyConf

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, world: GridWorld, medium_conf: MediumDifficultyConf):
        self._world = world
        self._medium_conf = medium_conf

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, hard_obs_high: NDArray) -> spaces.Space:
        spatial_obs = self._setup_spatial_obs(hard_obs_high)
        episode_meta = self._setup_episode_meta()
        droid_meta = self._setup_droid_meta()

        return spaces.Dict({
            "grid": spatial_obs,
            "episode_meta": episode_meta,
            "droid_meta": droid_meta
        })

    def reset(self):
        self._steps_left = self._medium_conf.max_steps

    def get_observation(self, state: GridWorld)-> dict[str, np.ndarray]:
        self._steps_left -= 1 # So we mirror the state in the gymnasium wrapper TODO: don't work because reset shouldn't make a move, so need to find a way to find one point of truth for this value...

        spatial_obs = self._get_spatial_obs()
        episode_meta = self._get_episode_meta()
        droid_meta = self._get_droid_meta()

        return {
            "grid": spatial_obs,
            "episode_meta": episode_meta,
            "droid_meta": droid_meta
        }

    # === Getters === #

    def get_steps_left(self):
        return self._steps_left

    # ================= #
    #      Helpers      #
    # ================= #

    # === Setup obs === #

    def _setup_spatial_obs(self, hard_obs_high: NDArray) -> spaces.Box:
        max_orb_lifespan = (self._medium_conf.grid_rows - 1) + (self._medium_conf.grid_cols - 1)
        high = np.asarray([*hard_obs_high, max_orb_lifespan],dtype=np.float32)

        self._ROWS = self._medium_conf.grid_rows
        self._COLS = self._medium_conf.grid_cols
        self._CHANNELS = len(high)

        high = np.tile(high, (self._ROWS, self._COLS, 1))

        return spaces.Box(
            low=0,
            high=high,
            shape=(self._medium_conf.grid_rows, self._medium_conf.grid_cols, self._CHANNELS),
            dtype=np.float32,
        )

    def _setup_episode_meta(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=self._medium_conf.max_steps,
            shape=(1,),
            dtype=np.float32,
        )

    def _setup_droid_meta(self) -> spaces.Box:
        max_score = self._medium_conf.max_score
        max_tier_chain = self._medium_conf.max_tier

        high = np.asarray([max_score, max_tier_chain], dtype=np.float32)

        return spaces.Box(
            low=0,
            high=high,
            shape=(2,),
            dtype=np.float32,
        )

    # === Get obs === #

    def _get_spatial_obs(self) -> np.ndarray:
        grid = np.zeros((self._ROWS, self._COLS, self._CHANNELS), dtype=np.float32)

        # Droid data
        droid_y, droid_x = self._world.droid.position
        grid[droid_y, droid_x, 0] = 1

        # Orb data
        for orb in self._world.ALL_ORBS:
            if not orb.is_active:
                continue

            y, x = orb.position

            grid[y, x, 1] = orb.meta.category.value
            grid[y, x, 2] = orb.meta.type.value
            grid[y, x, 3] = orb.meta.tier
            grid[y, x, 4] = orb.timer.remaining

        return grid

    def _get_episode_meta(self) -> np.ndarray:
        return np.asarray([self._steps_left], dtype=np.float32)

    def _get_droid_meta(self) -> np.ndarray:
        score = self._world.droid.score
        current_tier_chain = self._world.droid.digestion_engine.chained_tiers

        return np.asarray([score, current_tier_chain], dtype=np.float32)