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

    _linear_reward_growth: bool = True
    _step_wise_scoring_type: bool = True

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

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()

        # If no tiers have been chained yet, just return nothing
        if len(self._chained_tiers) == 0:
            return 0

        # Two types of scoring:
        # Reward at every correct collected tier resource
        if self._step_wise_scoring_type:
            return self._resolve_tier_progression(self._REWARD, 0)

        # Or reward only when the current chain is broken
        return self._resolve_tier_progression(0, self._REWARD)

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, multiplier: int) -> int:
        reward = 0

        if self._linear_reward_growth:
            reward = self._POSITIVE_BASE_REWARD * multiplier
        # TODO: just for testing if incentive structure for the agent changes if we power up the
        # reward, remove if not necessary
        else:
            reward = int(self._POSITIVE_BASE_REWARD * (1.5 ** multiplier))

        return reward

    def _resolve_tier_progression(self, chain_reward: int, break_reward: int) -> int:
        if self._chained_tiers[-1] == self.meta.tier - 1:
            return super()._chain_tier(chain_reward)

        return super()._break_tier_chain(break_reward)