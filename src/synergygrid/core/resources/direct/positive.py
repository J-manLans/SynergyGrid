from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
    Tier
)


class PositiveResource(BaseResource):
    """
    A resource that gives the agent a positive score.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, world_boundaries: tuple[int, int], cool_down: int = 5
    ):
        super().__init__(
            world_boundaries,
            self._POSITIVE_BASE_REWARD,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.POSITIVE,
                tier=Tier.ZERO
            )
        )
