import pytest
from synergygrid.core.resources import TierResource, NegativeResource

# ================= #
#      Helpers      #
# ================= #


def _tier_params(max_tier=10):
    return [(TierResource(t, (5, 5)), list(range(t))) for t in range(1, max_tier + 1)]


class TestTier:
    @pytest.fixture
    def tier_resource(self):
        """
        Creates a Tier 1 resource with the world boundaries set to 5x5 used to calculate its life span.
        """
        tr = TierResource(1, (5, 5))
        tr.reset()

        return tr

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_successful_step_wise_consume(self, resource: TierResource, resource_chain):
        """Test that the tier reward is given correctly for a valid step-wise chain."""

        resource._step_wise_scoring_type = True
        resource._chained_tiers.extend(resource_chain)
        assert resource.consume() == resource._REWARD

    def test_unsuccessful_step_wise_consume(self, tier_resource: TierResource):
        """Test that step-wise consume returns 0 when the tier chain is invalid for the resource."""

        tier_resource._step_wise_scoring_type = True
        tier_resource._chained_tiers.extend([0, 1, 2])

        assert tier_resource.consume() == 0

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_successful_combo_consume(self, resource: TierResource, resource_chain):
        """Test that combo consume gives the correct reward when the tier chain is complete."""

        resource._step_wise_scoring_type = False
        resource._chained_tiers.extend(resource_chain)
        assert resource.consume() == 0

    def test_unsuccessful_combo_consume(self, tier_resource: TierResource):
        """
        Test that combo consume returns the expected value when the tier chain is invalid for the resource.
        """
        tier_resource._step_wise_scoring_type = False
        tier_resource._chained_tiers.extend([0, 1, 2])

        assert tier_resource.consume() == tier_resource._REWARD
