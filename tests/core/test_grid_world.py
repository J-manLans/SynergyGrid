import pytest
from synergygrid.core.grid_world import GridWorld, AgentAction


class TestGridWorld:
    @pytest.fixture
    def gw(self):
        """Fixture for a fresh Agent"""
        return GridWorld(3, 5, 5, 1)
