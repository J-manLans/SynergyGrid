from enum import Enum
from synergygrid.core.resources import BaseResource
from typing import Final


class AgentAction(Enum):
    """Actions the Agent is capable of performing i.e. go in a certain direction"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class SynergyAgent:

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int, grid_cols: int, starting_score: int):
        """
        Initializes the agent.

        Defines the game world so the agent know its bounds, set its starting score and initializes the last action to an empty string so the renderer will have something to work with before its first action.
        """

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.score = starting_score

    def reset(self, starting_score: int) -> None:
        """Initialize Agents starting position at the center of the grid and reset its score"""

        self.position = [self.grid_rows // 2, self.grid_cols // 2]
        self.score = starting_score

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_action(self, agent_action: AgentAction) -> None:
        """Records current action and moves the agent according to it"""

        # Move Agent to the next cell
        if agent_action == AgentAction.LEFT:
            self._moveTowardsMinBound(1)
        elif agent_action == AgentAction.RIGHT:
            self._moveTowardsMaxBound(1, self.grid_cols - 1)
        elif agent_action == AgentAction.UP:
            self._moveTowardsMinBound(0)
        elif agent_action == AgentAction.DOWN:
            self._moveTowardsMaxBound(0, self.grid_rows - 1)

    def consume_resource(self, resource: BaseResource) -> int:
        """Consumes the resource, add its reward to its score and returns the reward"""

        reward = resource.consume()
        self.score += reward
        return reward

    # ================= #
    #      Helpers      #
    # ================= #

    def _moveTowardsMinBound(self, axis: int) -> None:
        self.position[axis] = max(self.position[axis] - 1, 0)

    def _moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.position[axis] = min(self.position[axis] + 1, bound)
