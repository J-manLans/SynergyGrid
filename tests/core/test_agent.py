import pytest
from synergygrid.core.agent import AgentAction, SynergyAgent


class DummyResource:
    """
    Minimal stub resource used for testing.

    Simulates a resource that always returns a fixed reward
    when consumed. This isolates the agent logic from the
    actual resource implementation.
    """
    def consume(self):
        return 5


class TestAgent:
    """
    Unit tests for the SynergyAgent class.

    Verifies:
    - Initial positioning and score initialization
    - Movement behavior for all valid actions
    - Boundary handling at grid edges
    - Reset functionality
    - Resource consumption and score updates
    """

    @pytest.fixture
    def agent(self):
        """
        Creates a SynergyAgent instance on a 6x6 grid
        with a starting score of 10 and resets it to
        its initial state.
        """
        agent = SynergyAgent(grid_rows=6, grid_cols=6, starting_score=10)
        agent.reset()

        return agent

    def test_initial_position_center(self, agent):
        """
        Ensures that after reset:
        - The agent starts in the center of the grid.
        - The score is initialized correctly.
        """
        assert agent.position == [3, 3]
        assert agent.score == 10

    @pytest.mark.parametrize(
        "action, expected_position",
        [
            (AgentAction.RIGHT, [3, 4]),
            (AgentAction.LEFT, [3, 2]),
            (AgentAction.UP, [2, 3]),
            (AgentAction.DOWN, [4, 3]),
        ],
    )
    def test_movement(self, agent, action, expected_position):
        """
        Verifies that each movement action correctly
        updates the agent's position when not at a boundary.
        """
        agent.perform_action(action)

        assert agent.position == expected_position

    @pytest.mark.parametrize(
        "position, action, expected_position",
        [
            ([0, 0], AgentAction.LEFT, [0, 0]),
            ([0, 5], AgentAction.RIGHT, [0, 5]),
            ([0, 0], AgentAction.UP, [0, 0]),
            ([5, 0], AgentAction.DOWN, [5, 0]),
        ],
    )
    def test_boundaries(self, agent, position, action, expected_position):
        """
        Ensures that the agent does not move outside
        the grid boundaries when attempting invalid moves.
        """
        agent.position = position
        agent.perform_action(action)

        assert agent.position == expected_position

    def test_reset_restores_position(self, agent):
        """
        Ensures that calling reset() restores the agent's
        position to the center of the grid.
        """
        starting_position = agent.position.copy()
        agent.position = [0, 0]
        agent.reset()

        assert agent.position == starting_position

    def test_reset_restores_score(self, agent):
        """
        Ensures that calling reset() restores the agent's
        score to its original starting value.
        """
        starting_score = agent.score
        agent.score += 10
        agent.reset()
        
        assert agent.score == starting_score

    def test_consume_resource_adds_score(self, agent):
        """
        Verifies that consuming a resource:
        - Returns the correct reward value
        - Increases the agent's score accordingly
        """
        resource = DummyResource()
        reward = agent.consume_resource(resource)

        assert reward == 5
        assert agent.score == 15