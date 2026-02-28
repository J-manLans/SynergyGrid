from enum import Enum


class ResourceCategory(Enum):
    DIRECT = 0
    SYNERGY = 1


class SynergyType(Enum):
    TIER = 0


class DirectType(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class ResourceMeta:
    def __init__(
        self, category: ResourceCategory, type: DirectType | SynergyType, tier: int
    ):
        self.category = (
            category  # For finding correct image to render together with subtype
        )
        self.type = type  # Render + for identifying subtype to the agent
        # Resources tier.
        # -1 if not applicable
        # 0 for base (positive resource)
        # ...n for rest of the tier resources
        self.tier = tier
