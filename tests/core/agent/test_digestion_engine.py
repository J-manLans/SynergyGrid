import pytest

from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.synergy.tier_resource import TierResource
from synergygrid.core.agent.digestion_engine import DigestionEngine


class TestDigestionEngine:
    _MAX_TIER = 10

    # ================= #
    #      Helpers      #
    # ================= #

    @staticmethod
    def _tier_params(max_tier=_MAX_TIER):
        BaseResource.set_life_span(5, 5)
        TierResource.MAX_TIER = max_tier
        return [(TierResource(t)) for t in range(0, max_tier + 1)]

    # ================= #
    #        Init       #
    # ================= #

    @pytest.fixture
    def digestion_engine(self) -> DigestionEngine:
        return DigestionEngine()

    # ================= #
    #       Tests       #
    # ================= #

    @pytest.mark.parametrize("resource", _tier_params())
    def test_successful_stepwise_reward(
        self, digestion_engine: DigestionEngine, resource: TierResource
    ):
        # reset and set correct scoring type
        resource.reset()
        resource.step_wise_scoring_type = True

        # prep the "chain" and digest
        digestion_engine.chained_tiers = resource.meta.tier - 1

        assert resource.REWARD == digestion_engine.digest(resource)

    @pytest.mark.parametrize("resource", _tier_params())
    def test_delayed_reward(
        self, digestion_engine: DigestionEngine, resource: TierResource
    ):
        # reset and set correct scoring type
        resource.reset()
        resource.step_wise_scoring_type = False

        # prep the "chain" and digest
        digestion_engine.chained_tiers = resource.meta.tier - 1
        digestion_engine.digest(resource)

        # raise the max tier for our fault tier
        TierResource.MAX_TIER = self._MAX_TIER + 10
        faulty_resource = TierResource(self._MAX_TIER + 2)

        # ensure the last correct resource gives the reward
        assert resource.REWARD == digestion_engine.digest(faulty_resource)
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN
