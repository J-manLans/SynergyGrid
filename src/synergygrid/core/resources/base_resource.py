from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator
from synergygrid.core.resources import ResourceMeta


class BaseResource(ABC):
    consumed = False
    position = [np.int64(0), np.int64(0)]

    class Timer:
        def __init__(self):
            self.remaining = 0

        def is_set(self) -> bool:
            return self.remaining > 0

        def set(self, duration: int) -> None:
            self.remaining = duration

        def tick(self) -> bool:
            if self.remaining > 0:
                self.remaining -= 1
            return self.remaining == 0

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int], type: ResourceMeta):
        """Defines the game world so resources know their bounds"""
        self.world_boundaries = world_boundaries  # (row, col) of the grid
        self.type = type
        self.timer = self.Timer()

    # ================= #
    #        API        #
    # ================= #

    @abstractmethod
    def consume(self) -> int:
        """
        Defines how an agent interacts with the resource.
        """
        pass

    @abstractmethod
    def deplete_resource(self) -> None:
        '''
        Removes the resource without giving any reward.
        '''
        pass

    @abstractmethod
    def spawn(self, rng: Generator):
        """
        Spawns the resource.
        """
        pass
