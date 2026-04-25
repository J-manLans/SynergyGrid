from syn_grid.config.models import PerceptionConf
from syn_grid.core.orbs.orb_meta import OrbCategory, DirectType, SynergyType
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.grid_world import GridWorld

import numpy as np
from abc import ABC, abstractmethod
from gymnasium import spaces
from typing import Final


class BasePerception(ABC):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, conf: PerceptionConf, orbs: int) -> None:
        # Global values
        self._ORBS_IN_ENV = orbs
        self._MAX_ACTIVE_ORBS: Final[int] = conf.max_active_orbs
        self._MAX_STEPS: Final[int] = conf.max_steps
        self._MAX_GRID_Y: Final[int] = conf.grid_rows - 1
        self._MAX_GRID_X: Final[int] = conf.grid_cols - 1

        # Droid data
        self._MAX_SCORE: Final[int] = conf.max_score
        self._MAX_TIER_CHAIN: Final[int] = conf.max_tier

        # Orb data
        self._MAX_CATEGORY: Final[int] = len(OrbCategory) - 1
        self._MAX_TYPE: Final[int] = max(len(DirectType) - 1, len(SynergyType) - 1)
        self._MAX_TIER: Final[int] = conf.max_tier
        self._MAX_ORB_LIFESPAN: Final[int] = BaseOrb._LIFE_SPAN

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_max_global_values(self) -> list[int]:
        return [self._MAX_STEPS]

    def _get_max_droid_positions(self) -> list[int]:
        return [self._MAX_GRID_Y, self._MAX_GRID_X]

    def _get_max_droid_data(self) -> list[int]:
        return [self._MAX_SCORE, self._MAX_TIER_CHAIN]

    def _get_max_orb_positions(self) -> list[int]:
        return [self._MAX_GRID_Y, self._MAX_GRID_X]

    def _get_max_orb_identity(self) -> list[int]:
        return [self._MAX_CATEGORY, self._MAX_TYPE, self._MAX_TIER]

    def _get_max_orb_data(self) -> list[int]:
        return [self._MAX_ORB_LIFESPAN]

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def setup_obs_space(self) -> spaces.Space: ...

    @abstractmethod
    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray: ...
