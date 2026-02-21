from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator


class BaseResourceTest(ABC):
    consumed = True
    pos = [np.int64(0), np.int64(0)]

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

    def __init__(self, world_boundaries: tuple[int, int]):
        """Defines the game world so resources know their bounds"""
        self.world_boundaries = world_boundaries  # (row, col) of the grid
        self.timer = self.Timer()


    # ================= #
    #        API        #
    # ================= #

    @abstractmethod
    def consume(self) -> int:
        """
        Defines how an agent interacts with the resource.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def spawn(self, rng: Generator):
        """
        Spawns the resource.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def is_at_position(self, pos: list[np.int64]) -> bool:
        """
        Check if the resource is at a given position.
        """
        pass

    @abstractmethod
    def is_consumed(self) -> bool:
        """
        Check if the resource is consumed.
        """
        pass

    # ================= #
    #      Setters      #
    # ================= #
