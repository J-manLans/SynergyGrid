from syn_grid.gymnasium.observation_space_developing.difficulty.base_difficulty import BaseDifficulty
from syn_grid.core.grid_world import GridWorld

from gymnasium import spaces
from numpy.typing import NDArray


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, world: GridWorld):
        self._world = world

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, spatial_obs: spaces.Space) -> spaces.Space:
        grid_meta = self._setup_grid_meta()
        agent_meta = self._setup_agent_meta()
        orb_meta = self._setup_orb_meta()

        return spaces.Dict ({
            "grid": spatial_obs,
            "grid_meta": grid_meta,
            "agent_meta": agent_meta,
            "orb_meta": orb_meta
        })

    def apply(self, state)-> dict[str, NDArray]:
        ...

    # ================= #
    #      Helpers      #
    # ================= #

    def _setup_grid_meta(self) -> spaces.Box:
        ...

    def _setup_agent_meta(self) -> spaces.Box:
        ...

    def _setup_orb_meta(self) -> spaces.Box:
        ...
