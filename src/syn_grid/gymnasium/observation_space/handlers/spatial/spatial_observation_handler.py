from syn_grid.gymnasium.observation_space.base_observation_handler import BaseObservationHandler

from gymnasium.spaces.space import Space
from gymnasium import spaces
import numpy as np

class SpatialObservationHandler(BaseObservationHandler):
    def setup_obs_space(self) -> Space:
        return spaces.Dict({
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(H, W, C),
                dtype=np.float32
            ),
            'grid meta': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N),
                dtype=np.float32
            ),
            'agent meta': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N),
                dtype=np.float32
            ),
            'orb meta': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N),
                dtype=np.float32
            ),
        })