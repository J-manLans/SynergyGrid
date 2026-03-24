from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.resource_meta import (
    ResourceMeta,
    ResourceCategory,
    DirectType,
)
from typing import Final


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    REWARD: Final[int] = -3

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, cool_down: int = 7):
        super().__init__(
            self.REWARD,
            cool_down,
            ResourceMeta(category=ResourceCategory.DIRECT, type=DirectType.NEGATIVE),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> "NegativeResource":
        super()._consume()
        return self
