import gymnasium as gym
from gymnasium import spaces
from synergygrid.core import GridWorld, AgentAction
from synergygrid.rendering import PygameRenderer
from synergygrid.gymnasium import ObservationHandler


class SynergyGridEnv(gym.Env):
    """
    SynergyGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # Metadata required by Gym.
    # "human" for Pygame visualization.
    # FPS caps the render() update rate; each call corresponds to one logic step, not full game fps.
    # Sub-loop in PygameRenderer.render() creates smooth animation between steps.
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        max_active_resources: int = 3,
        grid_rows: int = 5,
        grid_cols: int = 5,
        max_steps: int = 100,
        render_mode: str | None = None,
    ):
        # Set up bench environment;

        self._init_vars(
            max_active_resources, grid_rows, grid_cols, max_steps, render_mode
        )
        self._init_episode_vars()
        self._init_world(max_active_resources, grid_rows, grid_cols)
        if self.render_mode == "human":
            self._init_renderer(grid_rows, grid_cols)

        # Set up Gymnasium environment:

        # Gymnasium also requires us to define action_space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(AgentAction))

        # Same goes with observation_space: this provides the agent with a structured view
        # of the world that it uses to decide its actions.
        self.observation_handler = ObservationHandler(
            self._world, grid_rows, grid_cols, max_steps, self._step_count_down
        )
        self.observation_space = self.observation_handler.setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the world.
        self._init_episode_vars()
        self._world.reset(self.np_random)

        if self.render_mode == "human":
            self.render()

        obs = self.observation_handler.get_observation()

        # Return observation and info (not used)
        return self.observation_handler.normalize_obs(obs), {}

    def step(self, action: AgentAction):
        # Perform action and adjust variables affected by it
        reward = self._world.perform_agent_action(AgentAction(action))
        self._step_count_down -= 1
        truncated = self._step_count_down <= 0
        terminated = self._world._agent.score <= 0

        if self.render_mode == "human":
            self.render()

        obs = self.observation_handler.get_observation()

        # Return observation, reward, terminated, truncated and info (not used)
        return (
            self.observation_handler.normalize_obs(obs),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self) -> None:
        self._renderer.render(
            self._world._agent.position,
            self._world.get_resource_is_active_status(True),
            self._world.get_resource_positions(True),
            self._world.get_resource_types(True),
            self._world._agent.score,
        )

    # ================== #
    #       Helpers      #
    # ================== #

    # === Init === #

    def _init_vars(
        self,
        max_active_resources: int,
        grid_rows: int,
        grid_cols: int,
        max_steps: int,
        render_mode: str | None,
    ) -> None:
        self.max_active_resources = max_active_resources
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._max_steps = max_steps
        self.render_mode = render_mode

    def _init_world(
        self, max_active_resources: int, grid_rows: int, grid_cols: int
    ) -> None:
        self._world = GridWorld(
            max_active_resources, grid_rows=grid_rows, grid_cols=grid_cols
        )

    def _init_renderer(self, grid_rows: int, grid_cols: int) -> None:
        self._renderer = PygameRenderer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata["render_fps"],
        )

    # === Global === #

    def _init_episode_vars(self) -> None:
        self._step_count_down = self._max_steps
