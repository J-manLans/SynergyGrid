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

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, world_boundaries: tuple[int, int], reward: int = 5, cool_down: int = 5
    ):
        super().__init__(
            world_boundaries,
            reward,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.POSITIVE,
                tier=Tier.ZERO,
            ),
        )
