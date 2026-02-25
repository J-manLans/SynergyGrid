from enum import Enum


class ResourceCategory(Enum):
    DIRECT = 0
    SYNERGY = 1


class Tier(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class SynergyType(Enum):
    TIER = 0


class DirectType(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class ResourceMeta:
    def __init__(
        self, category: ResourceCategory, subtype: DirectType | SynergyType, tier: Tier
    ):
        self.category = category # For finding correct image to render together with subtype
        self.subtype = subtype # Render + for identifying subtype to the agent
        self.tier = tier # TODO: clarify this when feature fully implemented
