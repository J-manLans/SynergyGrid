from synergygrid.core.resources.synergy.tier_resource import TierResource
from synergygrid.core.resources.base_resource import BaseResource

import pytest
import numpy as np


class TestTier:
    _GRID_ROWS = 5
    _GRID_COLS = 5
    _MAX_TIER = 5
    _COOL_DOWN = 10
    _TIER = 2

    """
    Unit tests for the TierResource class.

    Verifies:
    - Step-wise reward behavior for correctly chained tiers
    - Step-wise reward behavior for incomplete or invalid tier chains
    - Combo reward behavior for correctly chained tiers
    - Combo reward behavior for incomplete or invalid tier chains
    - That the resource updates _chained_tiers correctly during consumption
    """

    # ================= #
    #       Init        #
    # ================= #

    @pytest.fixture
    def resource(self):
        """
        Creates a Tier 2 resource with the world boundaries set to rows x cols used to calculate its life span.
        """

        BaseResource.set_life_span(self._GRID_ROWS, self._GRID_COLS)
        TierResource.MAX_TIER = self._MAX_TIER
        t = TierResource(self._TIER, self._COOL_DOWN)
        t.reset()

        return t

    # ================= #
    #       Tests       #
    # ================= #

    def test_created_resource(self, resource: TierResource):
        assert resource._cool_down == self._COOL_DOWN
        assert resource.meta.tier == self._TIER
        assert resource.MAX_TIER == self._MAX_TIER
        assert resource._LIFE_SPAN == (self._GRID_ROWS - 1) + (self._GRID_COLS - 1)

    def test_consuming_resource_returns_the_resource(self, resource: TierResource):
        assert resource.consume() is resource

    def test_stepwise_reward_is_correct(self, resource: TierResource):
        resource.step_wise_scoring_type = True

        assert (resource.meta.tier + 1) * resource._TIER_BASE_REWARD == resource.REWARD

    def test_factor_reward_is_correct(self, resource: TierResource):
        resource.step_wise_scoring_type = False

        assert (
            resource._TIER_BASE_REWARD * (resource._GROWTH_FACTOR * (self._TIER - 1))
        ) + 0.5

    def test_active_resource_is_correct(self, resource: TierResource):
        position = [
            np.int64(max(0, self._GRID_ROWS - 2)),
            np.int64(max(0, self._GRID_COLS - 2)),
        ]
        resource.spawn(position)

        assert resource.timer.remaining == resource._LIFE_SPAN
        assert resource.is_active
        assert resource.position == position

    def test_creating_resource_with_negative_tier(self):
        with pytest.raises(ValueError):
            TierResource(-1)

    def test_creating_resource_with_high_tier(self):
        TierResource.MAX_TIER = 999
        resource = TierResource(666)

        resource.step_wise_scoring_type = True
        assert (resource.meta.tier + 1) * resource._TIER_BASE_REWARD == resource.REWARD

        resource.step_wise_scoring_type = False
        assert (
            resource._TIER_BASE_REWARD * (resource._GROWTH_FACTOR * (self._TIER - 1))
        ) + 0.5

    def test_creating_resource_with_to_high_tier(self):
        TierResource.MAX_TIER = self._MAX_TIER

        with pytest.raises(ValueError):
            TierResource(666)
