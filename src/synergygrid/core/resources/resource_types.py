from enum import Enum, auto


class ResourceCategory(Enum):
    DIRECT = 0
    SYNERGY = 1


class Tier(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class SynergyType(Enum):
    REVERT_SCORE = 0
    PERCEPTION = 1
    ENV_EFFECT = 2


class DirectType(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class ResourceMeta:
    def __init__(
        self, category: ResourceCategory, subtype: DirectType | SynergyType, tier: Tier
    ):
        self.category = category  # ResourceCategory
        self.subtype = subtype  # SynergyType or None
