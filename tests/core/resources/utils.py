import pytest
from synergygrid.core.resources.base_resource import BaseResource


def base_check_for_inactive_resource(resource: BaseResource):
    assert not resource.is_active
    assert resource.timer.remaining == 0
