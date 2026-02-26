from synergygrid.core.resources import (
    BaseResource,
    ResourceMeta,
    ResourceCategory,
    SynergyType
)
from typing import Final


class TierResource(BaseResource):
    """
    A resource that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 resource a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    _REWARD: Final[int]
    _reward_growth: str = 'linear'
    _scoring_type: str = 'step wise'


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

        if self._reward_growth == 'linear':
            reward = self._POSITIVE_BASE_REWARD * multiplier
        # TODO: just for experimenting if incentive structure change if we power up the reward
        elif self._reward_growth == 'exponential':
            reward = int(self._POSITIVE_BASE_REWARD * (1.5 ** multiplier))

        return reward

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super().preConsume()

        # If no tiers have been chained yet, just return nothing
        if len(self._chained_tiers) == 0:
            return 0

        if self._scoring_type == 'step wise':
            # If the last chained tier was the one below this one, return the reward
            if self._chained_tiers[-1] == self.type.tier - 1:
                self._chained_tiers.append(self.type.tier)
                return self._REWARD

            # Clear the chain array and return 0 else
            self._chained_tiers.clear()
            return 0
        else:
            if self._chained_tiers[-1] == self.type.tier - 1:
                self._chained_tiers.append(self.type.tier)
                return 0

            self._chained_tiers.clear()
            return self._REWARD