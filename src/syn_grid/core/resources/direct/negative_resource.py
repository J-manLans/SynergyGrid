from syn_grid.config.models import NegativeConf
from syn_grid.core.resources.base_resource import BaseResource
from syn_grid.core.resources.resource_meta import (
    ResourceMeta,
    ResourceCategory,
    DirectType,
)
from typing import Final


class NegativeResource(BaseResource):
    """
    A resource that gives the agent a negative score.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: NegativeConf):
        super().__init__(
            conf.reward,
            conf.cool_down,
            ResourceMeta(category=ResourceCategory.DIRECT, type=DirectType.NEGATIVE),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> "NegativeResource":
        super()._consume()
        return self
