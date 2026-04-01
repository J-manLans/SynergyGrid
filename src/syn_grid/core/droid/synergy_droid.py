from syn_grid.config.models import DroidConf
from syn_grid.core.resources.base_resource import BaseResource
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.core.droid.digestion_engine import DigestionEngine


class SynergyDroid:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: DroidConf):
        """
        Initializes the agent.

        Defines the game world so the agent know its bounds, set its starting score and store it for later resetting.
        """

        self._grid_rows = conf.grid_rows
        self._grid_cols = conf.grid_cols
        self._starting_score = conf.starting_score
        self.digestion_engine = DigestionEngine()
        self.reset()

    def reset(self) -> None:
        """
        Initialize Agents starting position at the center of the grid and reset its score and the digestion engine.
        """

        self.position = [self._grid_rows // 2, self._grid_cols // 2]
        self.score = self._starting_score
        self.digestion_engine.reset()

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: DroidAction) -> int:
        """Performs current action"""

        # Move Agent to the next cell
        match agent_action:
            case DroidAction.LEFT:
                self._moveTowardsMinBound(1)
            case DroidAction.RIGHT:
                self._moveTowardsMaxBound(1, self._grid_cols - 1)
            case DroidAction.UP:
                self._moveTowardsMinBound(0)
            case DroidAction.DOWN:
                self._moveTowardsMaxBound(0, self._grid_rows - 1)
            case _:
                raise TypeError("This action isn't implemented")

        self.score -= 1
        return -1

    def consume_resource(self, resource: BaseResource) -> float:
        """Consumes the resource, add its reward to its score and returns the reward"""

        reward = self.digestion_engine.digest(resource.consume())
        self.score += reward
        return reward

    # ================= #
    #      Helpers      #
    # ================= #

    def _moveTowardsMinBound(self, axis: int) -> None:
        self.position[axis] = max(self.position[axis] - 1, 0)

    def _moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.position[axis] = min(self.position[axis] + 1, bound)
