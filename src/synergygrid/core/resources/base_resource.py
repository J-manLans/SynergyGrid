from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator
from synergygrid.core.resources import ResourceMeta
from typing import Final


class BaseResource(ABC):
    consumed = False
    position = [np.int64(0), np.int64(0)]

    class Timer:
        def __init__(self):
            self.remaining = 0

        def is_completed(self) -> bool:
            return self.remaining <= 0

        def set(self, duration: int) -> None:
            self.remaining = duration

        def tick(self) -> None:
            if self.remaining > 0:
                self.remaining -= 1

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int], count_down: int, reward: int, type: ResourceMeta):
        """Defines the game world so resources know their bounds"""

        self._LIFE_SPAN = count_down
        self._world_boundaries = world_boundaries  # (row, col) of the grid
        self._reward = reward
        self.type = type
        self.timer = self.Timer()

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        """
        Defines how an agent interacts with the resource.
        """

        self.consumed = True
        # TODO: refactor this later so the resource actually removes from the observation when consumed - either here or in the obs space return, I'm not sure yet, but when that is done, depending on strategy, the two rows of code below will be redundant.
        self.timer.set(0)
        return self._reward

    def deplete_resource(self, rng: Generator) -> None:
        """
        Removes the resource without giving any reward.
        """

        self.consumed = True
        # Set cool down timer
        self.timer.set(int(rng.integers(2, 7)))

    def spawn(self, rng: Generator):
        """
        Spawns the resource.
        """

        self.position = [
            rng.integers(1, self._world_boundaries[0]),
            rng.integers(1, self._world_boundaries[1]),
        ]

        self.consumed = False
        self.timer.set(self._LIFE_SPAN)
