from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator
from synergygrid.core.resources import ResourceMeta
from typing import Final


class BaseResource(ABC):
    is_active = False
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

    def __init__(
        self,
        world_boundaries: tuple[int, int],
        count_down: int,
        reward: int,
        type: ResourceMeta,
        rng: Generator
    ):
        """Defines the game world so resources know their bounds"""

        self._life_span = count_down
        self._world_boundaries = world_boundaries  # (row, col) of the grid
        self._reward = reward
        self.type = type
        self.rng = rng
        self.timer = self.Timer()

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        """
        Defines how an agent interacts with the resource.
        """

        self.is_active = False
        self.timer.set(int(self.rng.integers(2, 7)))
        return self._reward

    def deplete_resource(self, rng: Generator) -> None:
        """
        Removes the resource without giving any reward.
        """

        self.is_active = False
        # Set cool down timer
        self.timer.set(int(rng.integers(2, 7)))

    def spawn(self):
        """
        Spawns the resource.
        """

        self.position = [
            self.rng.integers(1, self._world_boundaries[0]),
            self.rng.integers(1, self._world_boundaries[1]),
        ]

        self.is_active = True
        self.timer.set(self._life_span)
