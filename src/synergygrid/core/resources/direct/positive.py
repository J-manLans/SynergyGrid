from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
    Tier,
)

from numpy.random import Generator
from typing import Final


class PositiveResource(BaseResource):
    """
    A resource that gives the agent a positive score.
    """

    REWARD: Final[int] = 15
    COUNT_DOWN: Final[int] = 8

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int]):
        super().__init__(
            world_boundaries,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.POSITIVE,
                tier=Tier.ZERO,
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        self.consumed = True
        # TODO: refactor this later so the resource actually removes from the observation when consumed - either here or in the obs space return, I'm not sure yet, but when that is done, depending on strategy, the two rows of code below will be redundant.
        self.timer.set(0)
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
