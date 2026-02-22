from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
    Tier,
)
from numpy.random import Generator
from typing import Final


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    REWARD: Final[int] = -3
    COUNT_DOWN: Final[int] = 5

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int]):
        super().__init__(
            world_boundaries,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.NEGATIVE,
                tier=Tier.ZERO,
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        self.consumed = True
        return self.REWARD

    def deplete_resource(self) -> None:
        self.consumed = True

    def spawn(self, rng: Generator):
        self.position = [
            rng.integers(1, self.world_boundaries[0]),
            rng.integers(1, self.world_boundaries[1]),
        ]

        self.consumed = False
        self.timer.set(self.COUNT_DOWN)
