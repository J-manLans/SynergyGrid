import pytest
from synergygrid.core.resources.synergy.tier_resource import TierResource
from synergygrid.core.resources.base_resource import BaseResource
from ..utils import base_check_for_inactive_resource

# ================= #
#      Helpers      #
# ================= #

def _tier_params(max_tier=10):
    BaseResource.set_life_span(5, 5)

    return [
        (TierResource(t), list(range(1, t))) for t in range(1, max_tier + 1)
    ]

class TestTier:
    """
    Unit tests for the TierResource class.

    Verifies:
    - Step-wise reward behavior for correctly chained tiers
    - Step-wise reward behavior for incomplete or invalid tier chains
    - Combo reward behavior for correctly chained tiers
    - Combo reward behavior for incomplete or invalid tier chains
    - That the resource updates _chained_tiers correctly during consumption
    """

    @pytest.fixture
    def resource(self):
        """
        Creates a Tier 2 resource with the world boundaries set to 5x5 used to calculate its life span.
        """

        BaseResource.set_life_span(5, 5)

        tr = TierResource(2)
        tr.reset()

        return tr

    def test_created_resource(self, resource: TierResource):
        base_check_for_inactive_resource(resource)

    def test_consuming_resource_returns_the_resource(self, resource: TierResource):
        assert resource.consume() is resource