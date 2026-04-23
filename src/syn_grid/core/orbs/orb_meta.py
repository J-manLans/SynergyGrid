from enum import Enum
from typing import Final


class OrbCategory(Enum):
    NONE = 0
    DIRECT = 1
    SYNERGY = 2


class SynergyType(Enum):
    NONE = 0
    TIER = 1


class DirectType(Enum):
    NONE = 0
    NEGATIVE = 1


class OrbMeta:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        category: OrbCategory,
        type: DirectType | SynergyType,
        tier: int | None = None,
    ):
        self._assert_type_and_tier_matches_category(category, type, tier)

        # These values are for finding correct image to render in PygameRenderer and to help the
        # agent identify the orb in the observation
        self.CATEGORY: Final[OrbCategory] = category
        self.TYPE: Final[DirectType | SynergyType] = type
        self.TIER: Final[int] = tier if (tier is not None) else 0

    # ================= #
    #      Helpers      #
    # ================= #

    def _assert_type_and_tier_matches_category(
        self,
        category: OrbCategory,
        type: DirectType | SynergyType,
        tier: int | None,
    ) -> None:
        if category == OrbCategory.DIRECT:
            if not isinstance(type, DirectType):
                raise TypeError(
                    "If the category is DIRECT, the orb need to be of direct type"
                )
            if tier is not None:
                raise ValueError(
                    "If the category is DIRECT, tier should not be applied"
                )

        if category == OrbCategory.SYNERGY:
            if not isinstance(type, SynergyType):
                raise TypeError(
                    "If the category is SYNERGY, the orb need to be of synergy type"
                )

            if tier is None:
                raise ValueError("If the category is SYNERGY, tier should be applied")

            if tier < 1:
                raise ValueError("Tier orbs can't have tiers less than 1")
