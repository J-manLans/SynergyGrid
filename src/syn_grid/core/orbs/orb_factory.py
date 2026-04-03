from syn_grid.config.models import OrbConf, OrbFactoryConf, NegativeConf, TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb

from typing import Final


class OrbFactory:
    # ================= #
    #       Init        #
    # ================= #
    _MIN_POOL_SIZE: Final[int]

    def __init__(
        self,
        orb_manager_conf: OrbFactoryConf,
        negative_orb_conf: NegativeConf,
        tier_orb_conf: TierConf,
    ):
        self._MIN_POOL_SIZE = orb_manager_conf.max_active_orbs * 3
        self._orb_manager_conf = orb_manager_conf
        self._negative_orb_conf = negative_orb_conf
        self._tier_orb_conf = tier_orb_conf
        self._max_active_orbs = orb_manager_conf.max_active_orbs
        self._max_tier = orb_manager_conf.max_tier

    # ================= #
    #        API        #
    # ================= #

    def create_orbs(self) -> list[BaseOrb]:
        """Create all orbs according to the config and weights"""

        enabled_orbs = self._get_conf_enabled_orbs()
        num_enabled_orbs = len(enabled_orbs)
        total_weight = sum(enabled_orbs.values())

        # Shared setup
        BaseOrb.set_life_span(self._max_active_orbs, self._max_active_orbs)
        TierOrb.MAX_TIER = self._max_tier

        orbs: list[BaseOrb] = []
        for orb_type, orb_weight in enabled_orbs.items():
            ratio = orb_weight / total_weight
            count = self._compute_spawn_count(orb_type, num_enabled_orbs, ratio)

            if orb_type == "negative":
                orbs.extend(
                    [NegativeOrb(self._negative_orb_conf) for _ in range(count)]
                )
            elif orb_type == "tier":
                for tier in range(self._max_tier + 1):
                    orbs.extend(
                        [TierOrb(tier, self._tier_orb_conf) for _ in range(count)]
                    )

        return orbs

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_conf_enabled_orbs(self) -> dict[str, int]:
        """Return enabled orb types and their weights from orb_manager_conf"""

        enabled_orbs = {}
        for orb_type, orb_conf in self._orb_manager_conf.types.items():
            if orb_conf.enabled:
                enabled_orbs[orb_type] = orb_conf.weight
        if not enabled_orbs:
            raise ValueError("At least one orb must be enabled")
        return enabled_orbs

    def _compute_spawn_count(
        self, orb_type: str, num_enabled_orbs: int, ratio: float
    ) -> int:
        """Compute how many orbs to spawn based on weight ratio"""

        if orb_type == 'tier':
            return self._compute_tier_spawn_count(num_enabled_orbs, ratio)

        count = self._MIN_POOL_SIZE * ratio

        # Change to correct return type
        return 0

    def _compute_tier_spawn_count(self, num_enabled_orbs: int, ratio: float) -> int:
        ...