import gymnasium as gym
from gymnasium import spaces
import numpy as np
from synergygrid.core import GridWorld, AgentAction as act
from synergygrid.rendering import PygameRenderer


# The custom environment must inherit from gym.Env
class SynergyGridEnv(gym.Env):
    """
    SynergyGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # Metadata required by Gym.
    # "human" for Pygame visualization, "ansi" for console output.
    # FPS set low since the agent moves discretely between grid cells.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        grid_rows=5,
        grid_cols=5,
        max_steps=50,
        starting_score=20,
        render_mode=None,
    ):
        # Set up bench environment;

        self._init_configurable_vars(
            grid_rows, grid_cols, max_steps, render_mode, starting_score
        )
        self._init_episode_vars()
        self._init_world(grid_rows, grid_cols, starting_score)
        if render_mode == "human":
            self._init_renderer(grid_rows, grid_cols)

        # Set up Gymnasium environment:

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
                    # Agent position bounds
                    self.grid_rows - 1,
                    self.grid_cols - 1,
                    # Resource position bounds
                    self.grid_rows - 1,
                    self.grid_cols - 1,
                ]
            ),
        )

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the world.
        self._init_episode_vars()
        self.world.reset(self.np_random)

        if self.render_mode == "human":
            self.render()

        # Constructs the observation state: [agent_row, agent_col, resource_row, resource_col]
        obs = np.concatenate(
            (self.world.agent.position, self.world.resource.position), dtype=np.int32
        )

        # Return observation and info (not used)
        return obs, {}

    def step(self, action):
        # Perform action
        reward = (
            self.world.perform_agent_action(act(action))
        )
        self.step_count += 1

        # Render
        if self.render_mode == "human":
            self.render()

        # Prep Gymnasium variables
        obs = np.concatenate(
            (self.world.agent.position, self.world.resource.position), dtype=np.int32
        )
        truncated = self.step_count >= self.max_steps
        terminated = self.world.agent.score <= 0

        # Return observation, reward, terminated, truncated and info (not used)
        return obs, reward, terminated, truncated, {}

    def render(self):
        self.renderer.render(
            self.world.agent.position,
            self.world.resource.consumed,
            self.world.resource.position,
            self.world.resource.type,
            self.world.agent.score,
        )

    # ================== #
    #       Helpers      #
    # ================== #

    # === Init === #

    def _init_configurable_vars(
        self, grid_rows, grid_cols, max_steps, render_mode, starting_score
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_steps = max_steps
        self.render_mode = render_mode

    def _init_world(self, grid_rows, grid_cols, starting_score) -> None:
        self.world = GridWorld(
            grid_rows=grid_rows, grid_cols=grid_cols, starting_score=starting_score
        )

    def _init_renderer(self, grid_rows, grid_cols) -> None:
        self.renderer = PygameRenderer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata["render_fps"],
        )

    # === General === #

    def _init_episode_vars(self) -> None:
        self.step_count = 0


# =============================== #
#  Quick test for the game world  #
#  Click play button in your IDE  #
# =============================== #


# def testEnvironment():
#     import random

#     world = GridWorld()
#     renderer = PygameRenderer()
#     resource_consumed = False

#     world.reset()
#     _render(renderer, world, resource_consumed, 20)

#     while True:
#         action = random.randint(0, len(act) - 1)
#         resource_consumed, score, _ = world.perform_agent_action(act(action))

#         _render(renderer, world, resource_consumed, score)


# def _render(renderer, world, resource_consumed, score):
#     renderer.render(
#         world.get_agent_pos(),
#         resource_consumed,
#         world.get_resource_pos(),
#         score,
#     )


# if __name__ == "__main__":
#     testEnvironment()
