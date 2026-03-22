import pytest

from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.synergy.tier_resource import TierResource
from synergygrid.core.agent.digestion_engine import DigestionEngine

# ================= #
#      Helpers      #
# ================= #

def _tier_params(max_tier=10):
    BaseResource.set_life_span(5, 5)

    return [
        (TierResource(t), list(range(1, t))) for t in range(1, max_tier + 1)
    ]

class testDigestionEngine:
    @pytest.fixture
    def resource(self):
        """
        Creates a Tier 2 resource with the world boundaries set to 5x5 used to calculate its life span.
        """
        BaseResource.set_life_span(5, 5)

        tr = TierResource(2)
        tr.reset()

        return tr