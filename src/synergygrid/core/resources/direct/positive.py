from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType
)


class PositiveResource(BaseResource):
    """
    A resource that gives the agent a positive score.

    It also serves as a base for the tier resource chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, world_boundaries: tuple[int, int], cool_down: int = 5
    ):
        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.POSITIVE,
                tier=0
            )
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()
        return super()._chain_tier(self._POSITIVE_BASE_REWARD)
