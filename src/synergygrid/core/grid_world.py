from numpy.random import Generator
from synergygrid.core import AgentAction, SynergyAgent, BaseResource
import numpy as np


class GridWorld:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows=5, grid_cols=5, starting_score=10):
        '''
        Initializes the grid world. Defines the game world's size and initializes the agent and resource.
        '''

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent = SynergyAgent(
            grid_rows=5, grid_cols=5, starting_score=starting_score
        )
        self.resource = BaseResource(grid_rows=5, grid_cols=5)

    def reset(self, rng: Generator|None=None) -> None:
        '''
        Reset the agent to its starting position and re-spawns the resource at a random location
        '''

        self.agent.reset() # Initialize Agents starting position
        self.resource.reset(rng) # Initialize the resource's position

    # ================= #
    #       API         #
    # ================= #

    # === Logic === #

    def perform_agent_action(self, agent_action: AgentAction) -> bool:
        '''
        Perform an action through the agent and compare if its position is the same as the resource, if it is it returns True, otherwise False

        :param agent_action: the action the agent will perform
        '''

        self.agent.perform_action(agent_action)
        return self.agent.agent_pos == self.resource.resource_pos

    # === Getters === #

    def get_agent_pos(self) -> list[int]:
        return self.agent.agent_pos

    def get_resource_pos(self) -> list[np.int64]:
        return self.resource.resource_pos

    def get_last_action(self) -> str | AgentAction:
        return self.agent.last_action
