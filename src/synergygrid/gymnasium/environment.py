import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry, register

from synergygrid import GridWorld as gw
from synergygrid import AgentAction as act
import numpy as np


# The custom environment must inherit from gym.Env
class SynergyGridEnv(gym.Env):
    """
    A custom environment for SynergyGrid.
    """

    # Metadata required by Gym.
    # "human" for Pygame visualization, "ansi" for console output.
    # FPS set low since the agent moves discretely between grid cells.
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        grid_rows=5,
        grid_cols=5,
        max_steps=50,
        starting_score=10,
        render_mode=None,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = max_steps

        # Initialize the bench world
        self.agent = gw(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            starting_score=starting_score,
            fps=self.metadata["render_fps"],
        )

        # Gymnasium also requires us to define the action space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(act))

        # Gymnasium requires an observation space definition. Here we represent the state as a flat
        # vector: [agent_row, agent_col, resource_row, resource_col].
        # The space is used by Gymnasium to validate observations returned by reset() and step().
        self.observation_space = spaces.Box(
            dtype=np.int32,
            shape=(4,),  # 4 integers: agent position (2) + resource position (2)
            low=0,
            high=np.array(
                [
                    self.grid_rows - 1,
                    self.grid_cols - 1,  # Agent position bounds
                    self.grid_rows - 1,
                    self.grid_cols - 1,  # Resource position bounds
                ]
            ),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(
            seed=seed
        )  # gym requires this call to control randomness and reproduce scenarios.

        self.step_count = (
            0  # Reset so we don't get truncated right away on our second run.
        )

        # Reset the agent. Optionally, pass in seed to control randomness and reproduce scenarios.
        self.agent.reset(self.np_random)

        # Constructs the observation state: [agent_row, agent_col, resource_row, resource_col]
        obs = obs = np.concatenate(
            (self.agent.agent_pos, self.agent.resource_pos), dtype=np.int32
        )

        # Render environment if desired
        if self.render_mode == "human":
            self.render()

        # Return observation and info (not used)
        return obs, {}

    def step(self, action):
        # Perform action
        resource_consumed = self.agent.perform_action(act(action))

        # Determine reward and termination
        reward = 0
        terminated = False
        if resource_consumed:
            reward = 1
            terminated = True

        # Constructs the observation state: [agent_row, agent_col, resource_row, resource_col]
        obs = np.concatenate(
            (self.agent.agent_pos, self.agent.resource_pos), dtype=np.int32
        )

        # Render environment if desired
        if self.render_mode == "human":
            self.render(action=action)

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        # Return observation, reward, terminated, truncated (not used) and info (not used)
        return obs, reward, terminated, truncated, {}

    def render(self, mode="human", action=None):
        self.agent.renderer.render()

    def close(self):
        pass
