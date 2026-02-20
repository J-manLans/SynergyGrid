from enum import Enum


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

    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10):
        """
        Initializes the agent.

        Defines the game world so the agent know its bounds, set its starting score and initializes the last action to an empty string so the renderer will have something to work with before its first action.
        """

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.score = starting_score
        self.last_action = ""

    def reset(self) -> None:
        """Initialize Agents starting position at the center of the grid"""

        self.pos = [self.grid_rows // 2, self.grid_cols // 2]

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_action(self, agent_action: AgentAction) -> None:
        """Records current action and moves the agent according to it"""

        self.last_action = agent_action

        # Move Agent to the next cell
        if agent_action == AgentAction.LEFT:
            self.__moveTowardsMinBound(1)
        elif agent_action == AgentAction.RIGHT:
            self.__moveTowardsMaxBound(1, self.grid_cols - 1)
        elif agent_action == AgentAction.UP:
            self.__moveTowardsMinBound(0)
        elif agent_action == AgentAction.DOWN:
            self.__moveTowardsMaxBound(0, self.grid_rows - 1)

    # ================= #
    #      Helpers      #
    # ================= #

    def __moveTowardsMinBound(self, axis: int) -> None:
        self.pos[axis] = max(self.pos[axis] - 1, 0)

    def __moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.pos[axis] = min(self.pos[axis] + 1, bound)
