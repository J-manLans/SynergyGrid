from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
)
from typing import Final


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    _REWARD: Final[int] = -3

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int], cool_down: int = 7):
        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT, type=DirectType.NEGATIVE, tier=-1
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()
        return super()._break_tier_chain(self._REWARD)
