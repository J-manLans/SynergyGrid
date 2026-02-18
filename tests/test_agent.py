import pytest
from synergygrid.core.agent import Agent, AgentAction, GridTile


class TestAgent:
    @pytest.fixture
    def agent(self):
        """Fixture for a fresh Agent"""
        return Agent()

    def test_initial_position(self, agent: Agent):
        # Check that agent starts in the middle
        assert agent.agent_pos == [2, 2]

    def test_resetting_agent(self, agent: Agent):
        agent.perform_action(AgentAction.DOWN)
        agent.reset()
        assert agent.agent_pos == [2, 2]

    @pytest.mark.parametrize(
        "action, expected_pos",
        [
            (AgentAction.RIGHT, [2, 3]),
            (AgentAction.DOWN, [3, 2]),
            (AgentAction.LEFT, [2, 1]),
            (AgentAction.UP, [1, 2]),
        ],
    )
    def test_cardinal_directions(self, agent: Agent, action, expected_pos):
        agent.perform_action(action)
        assert agent.agent_pos == expected_pos

    @pytest.mark.parametrize(
        "action, row_col, value",
        [
            (AgentAction.RIGHT, 1, 4),
            (AgentAction.DOWN, 0, 4),
            (AgentAction.LEFT, 1, 0),
            (AgentAction.UP, 0, 0),
        ],
    )
    def test_grid_bounds(self, agent: Agent, action, row_col, value):
        for i in range(agent.grid_cols + 5):
            agent.perform_action(action)

        assert agent.agent_pos[row_col] == value
