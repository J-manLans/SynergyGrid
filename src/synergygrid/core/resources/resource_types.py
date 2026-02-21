from enum import Enum, auto

class ResourceCategory(Enum):
    DIRECT = auto()
    SYNERGY = auto()

class SynergyType(Enum):
    REVERT_SCORE = auto()
    PERCEPTION = auto()
    ENV_EFFECT = auto()

class DirectType(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()

class ResourceMeta:
    def __init__(self, category, subtype=None) :
        self.category = category        # ResourceCategory
        self.subtype = subtype          # SynergyType or None