from abc import ABC, abstractmethod
from gymnasium import spaces
from numpy.typing import NDArray



class BaseDifficulty(ABC):
    @abstractmethod
    def setup_obs_space(self, spatial_obs: spaces.Space) -> spaces.Space:
        ...

    @abstractmethod
    def apply(self, state) -> dict[str, NDArray]:
        ...