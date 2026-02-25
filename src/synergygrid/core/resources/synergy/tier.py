from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    SynergyType
)
from typing import Final


class TierResource(BaseResource):
    _REWARD: Final[int]
    reward_growth: str = 'linear'


    """
    A resource that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 resource a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, tier, world_boundaries: tuple[int, int], cool_down: int = 10
    ):
        self._REWARD = self._calculate_reward(tier + 1)

        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.SYNERGY,
                subtype=SynergyType.TIER,
                tier=tier
            )
        )

    def _calculate_reward(self, multiplier: int) -> int:
        reward = 0

        if self.reward_growth == 'linear':
            reward = self._POSITIVE_BASE_REWARD * multiplier
        elif self.reward_growth == 'exponential':
            reward = int(self._POSITIVE_BASE_REWARD * (1.5 ** multiplier))

        return reward

    # ================= #
    #        API        #
    # ================= #

    # TODO: continue here

    def consume(self) -> int:
        super().preConsume()


        return self._REWARD
