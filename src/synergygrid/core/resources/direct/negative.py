from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
    Tier,
)


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, world_boundaries: tuple[int, int], reward: int = -3, cool_down: int = 7
    ):
        super().__init__(
            world_boundaries,
            reward,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT,
                subtype=DirectType.NEGATIVE,
                tier=Tier.ZERO
            )
        )
