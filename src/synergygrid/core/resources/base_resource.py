from abc import ABC, abstractmethod
import numpy as np
from synergygrid.core.resources import ResourceMeta
from typing import Final


class BaseResource(ABC):
    is_active = False
    position = [np.int64(0), np.int64(0)]
    _cool_down: int
    _LIFE_SPAN: Final[int]

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
        reward: int,
        cool_down: int,
        type: ResourceMeta,
    ):
        """Defines the game world so resources know their bounds"""
        self._world_boundaries = world_boundaries  # (row, col) of the grid
        self._LIFE_SPAN = (world_boundaries[0] - 1) + (world_boundaries[1] - 1)
        self._cool_down = cool_down
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

        self.is_active = False
        self.timer.set(self._cool_down)
        return self._reward

    def deplete_resource(self) -> None:
        """
        Removes the resource without giving any reward.
        """

        self.is_active = False
        self.timer.set(self._cool_down)

    def spawn(self, position: list[np.int64]):
        """
        Spawns the resource.
        """

        self.position = position
        self.is_active = True
        self.timer.set(self._LIFE_SPAN)
