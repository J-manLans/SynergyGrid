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

    _REWARD: Final[int] = 5

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
                subtype=DirectType.POSITIVE,
                tier=Tier.ZERO,
            ),
            rng
        )
