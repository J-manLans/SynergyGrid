import pytest
from synergygrid.core.grid_world import GridWorld, AgentAction, GridTile


class TestGridWorld:
    @pytest.fixture
    def gw(self):
        """Fixture for a fresh Agent"""
        return GridWorld()

    def test_initial_position(self, gw: GridWorld):
        assert gw.agent_pos == [2, 2]

    def test_resetting_agent(self, gw: GridWorld):
        gw.perform_agent_action(AgentAction.DOWN)
        gw.reset()
        assert gw.agent_pos == [2, 2]

    @pytest.mark.parametrize(
        "action, expected_pos",
        [
            (AgentAction.RIGHT, [2, 3]),
            (AgentAction.DOWN, [3, 2]),
            (AgentAction.LEFT, [2, 1]),
            (AgentAction.UP, [1, 2]),
        ],
    )
    def test_cardinal_directions(self, gw: GridWorld, action, expected_pos):
        gw.perform_agent_action(action)
        assert gw.agent_pos == expected_pos

    @pytest.mark.parametrize(
        "action, row_col, value",
        [
            (AgentAction.RIGHT, 1, 4),
            (AgentAction.DOWN, 0, 4),
            (AgentAction.LEFT, 1, 0),
            (AgentAction.UP, 0, 0),
        ],
    )
    def test_grid_bounds(self, gw: GridWorld, action, row_col, value):
        for i in range(gw.grid_cols + 5):
            gw.perform_agent_action(action)

        assert gw.agent_pos[row_col] == value
