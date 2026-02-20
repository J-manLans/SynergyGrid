from synergygrid.core.resources import BaseResourceTest
from numpy.random import Generator
from typing import Final
import numpy as np


class PositiveResource(BaseResourceTest):
    """
    A resource that gives the agent a positive effect.
    """

    REWARD: Final[int] = 5
    COLLECTED = True

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int]):
        super().__init__(world_boundaries)

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        self.COLLECTED = True
        return self.REWARD

    def spawn(self, rng: Generator):
        """Initializes the resource in the grid"""

        self.pos = [
            rng.integers(1, self.world_boundaries[0]),
            rng.integers(1, self.world_boundaries[1]),
        ]

        self.COLLECTED = False

    def is_at_position(self, pos: list[np.int64]) -> bool:
        """
        Check if the resource is at a given position.
        """
        return self.pos == pos
