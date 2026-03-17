from synergygrid.core.resources.base_tier_resource import BaseTierResource
from synergygrid.core.resources.resource_meta import ResourceMeta
from synergygrid.core.resources.resource_meta import ResourceCategory
from synergygrid.core.resources.resource_meta import SynergyType


class TierBase(BaseTierResource):
    """
    A resource that gives the agent a positive score.

    It also serves as a base for the tier resource chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, tier: int, world_boundaries: tuple[int, int], cool_down: int = 5
    ):
        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.SYNERGY, type=SynergyType.TierBase, tier=tier
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()
        if not super()._resolve_tier_progression():
            self._is_restart_needed()
        return self._TIER_BASE_REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    def _is_restart_needed(self):
        """If we just broke the chain, restart it since this is the base for the tier resources"""

        if len(self._chained_tiers) == 0:
            super()._chain_tier()
