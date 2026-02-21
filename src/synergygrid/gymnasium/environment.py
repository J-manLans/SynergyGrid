import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry, register
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
        starting_score=10,
        render_mode=None,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.resource_consumed = True
        self.step_count = 0

        # Initialize the bench world
        self.world = GridWorld(
            grid_rows=grid_rows, grid_cols=grid_cols, starting_score=starting_score
        )

        if render_mode == "human":
            # Initialize the rendering
            self.renderer = PygameRenderer(
                grid_rows=grid_rows,
                grid_cols=grid_cols,
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

    # ================== #
    #    Gym contract    #
    # ================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)
        # Reset so we don't get truncated right away on our second episode.
        self.step_count = 0
        # Reset the world.
        self.world.reset(self.np_random)

        # Constructs the observation state: [agent_row, agent_col, resource_row, resource_col]
        obs = np.concatenate(
            (self.world.get_agent_pos(), self.world.get_resource_pos()), dtype=np.int32
        )

        if self.render_mode == "human":
            self.render()

        # Return observation and info (not used)
        return obs, {}

    def step(self, action):
        # Perform action
        self.resource_consumed, reward = self.world.perform_agent_action(act(action))

        # TODO: need to fix termination - that is when the agents point reach zero

        obs = np.concatenate(
            (self.world.get_agent_pos(), self.world.get_resource_pos()), dtype=np.int32
        )

        if self.render_mode == "human":
            self.render()

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        # Return observation, reward, terminated, truncated and info (not used)
        return obs, reward, False, truncated, {}

    def render(self):
        self.renderer.render(
            self.world.get_agent_pos(),
            self.resource_consumed,
            self.world.get_resource_pos(),
            self.world.get_last_action(),
        )

    def close(self):
        pass


# =============================== #
#  Quick test for the game world  #
#  Click play button in your IDE  #
# =============================== #


def testEnvironment():
    import random

    world = GridWorld()
    renderer = PygameRenderer()
    resource_consumed = False

    world.reset()
    __render(renderer, world, resource_consumed)

    while True:
        action = random.randint(0, len(act) - 1)
        resource_consumed, _ = world.perform_agent_action(act(action))

        __render(renderer, world, resource_consumed)


def __render(renderer, world, resource_consumed):
    renderer.render(
        world.get_agent_pos(),
        resource_consumed,
        world.get_resource_pos(),
        world.get_last_action(),
    )


if __name__ == "__main__":
    testEnvironment()
