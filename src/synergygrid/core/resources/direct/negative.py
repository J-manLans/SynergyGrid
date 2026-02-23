from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
    Tier,
)
from typing import Final
from numpy.random import Generator


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    _REWARD: Final[int] = -3

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int], rng: Generator):
        super().__init__(
            world_boundaries,
            (world_boundaries[0] - 1) + (world_boundaries[1] - 1),
            self._REWARD,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.NEGATIVE,
                tier=Tier.ZERO,
            ),
            rng
        )
