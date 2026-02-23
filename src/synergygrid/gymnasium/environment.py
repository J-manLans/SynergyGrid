import gymnasium as gym
from gymnasium import spaces
import numpy as np
from synergygrid.core import GridWorld, AgentAction, SynergyType, DirectType
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
        max_active_resources: int = 3,
        grid_rows: int = 5,
        grid_cols: int = 5,
        max_steps: int = 100,
        starting_score: int = 20,
        render_mode: str | None = None,
    ):
        # Set up bench environment;

        self._init_configurable_vars(
            grid_rows, grid_cols, max_steps, starting_score, render_mode
        )
        self._init_episode_vars()
        self._init_world(max_active_resources, grid_rows, grid_cols, starting_score)
        if render_mode == "human":
            self._init_renderer(grid_rows, grid_cols)

        # Set up Gymnasium environment:

        # Gymnasium also requires us to define the action space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(AgentAction))

        self._setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the world.
        self._init_episode_vars()
        self.world.reset(self.starting_score, self.np_random)

        if self.render_mode == "human":
            self.render()

        # Constructs the observation state: [agent_row, agent_col, resource_row, resource_col]
        # obs = np.concatenate(
        #     (self.world.agent.position, self.world.resource.position), dtype=np.int32
        # )

        obs = self._get_observation()

        # print('-----------------NEW EPISODE-----------------', obs)

        # Return observation and info (not used)
        return self._normalize_obs(obs), {}

    def step(self, action):
        # Perform action
        reward = self.world.perform_agent_action(AgentAction(action))
        self.step_count_down -= 1

        # Render
        if self.render_mode == "human":
            self.render()

        # Prep Gymnasium variables
        # obs = np.concatenate(
        #     (self.world.agent.position, self.world.resource.position), dtype=np.int32
        # )
        obs = self._get_observation()
        norm_obs = self._normalize_obs(obs)

        # print(
        #     f"{obs}\nreward: {reward}\nscore: {self.world.agent.score}\nresource type: {self.world.resource.type.subtype}\n{'remaining time tile re-spawn: ' if self.world.resource.consumed else 'remaining time tile de-spawn: '}{self.world.resource.timer.remaining}\n"
        # )

        truncated = self.step_count_down <= 0
        terminated = self.world.agent.score <= 0

        # Return observation, reward, terminated, truncated and info (not used)
        return norm_obs, reward, terminated, truncated, {}

    def render(self):
        self.world.get_resource_is_active_status()

        self.renderer.render(
            self.world.agent.position,
            self.world.get_resource_is_active_status(),
            self.world.get_resource_positions(),
            self.world.get_resource_types(),
            self.world.agent.score,
        )

    # ================== #
    #       Helpers      #
    # ================== #

    # === Init === #

    def _init_configurable_vars(
        self,
        grid_rows: int,
        grid_cols: int,
        max_steps: int,
        starting_score: int,
        render_mode: str | None,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.starting_score = starting_score

    def _init_world(
        self, max_active_resources: int, grid_rows, grid_cols, starting_score
    ) -> None:
        self.world = GridWorld(
            max_active_resources,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            starting_score=starting_score,
        )

    def _init_renderer(self, grid_rows, grid_cols) -> None:
        self.renderer = PygameRenderer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata["render_fps"],
        )

    def _setup_obs_space(self):
        """
        Gymnasium requires an observation space definition. Here we represent the state as a flat
        vector. The space is used by Gymnasium to validate observations returned by reset() and step().

        Set up:
        - self._raw_low / self._raw_high: the original raw ranges (used for normalization)
        - self._sentinel_mask: boolean mask of indices that use -1 as "absent"
        - self.observation_space: the normalized observation space that agents will see
        (0..1 for active features, -1 for absent resource fields)
        """
        # original raw bounds (match your _get_observation() ordering)
        raw_low = np.array([0, 0, 0, -1, -1, -1, 0], dtype=np.float32)
        raw_high = np.array(
            [
                self.max_steps,  # episode length
                self.grid_rows - 1,  # agent_row max
                self.grid_cols - 1,  # agent_col max
                self.grid_rows - 1,  # resource_row max
                self.grid_cols - 1,  # resource_col max
                len(DirectType) - 1,  # resource_type max
                (self.grid_rows - 1) + (self.grid_cols - 1),  # timer max
            ],
            dtype=np.float32,
        )

        # store raw bounds for use in normalize_obs()
        self._raw_high = raw_high

        if np.any(raw_high - raw_low == 0):
            raise ValueError(
                "raw_high must be strictly greater than raw_low for all features"
            )

        # Absent resource mask: True where low == -1.0 (these fields mean "absent" when -1)
        # This mask will be used to keep -1 as a special value instead of normalizing it.
        self.resource_mask = raw_low == -1.0

        # normalized observation space the agent will receive:
        # inactive resources keep -1 as a valid "low" value; active features map to 0..1
        low_norm = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high_norm = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_norm, high=high_norm, dtype=np.float32
        )

    # === Global === #

    def _init_episode_vars(self) -> None:
        self.step_count_down = self.max_steps

    def _get_observation(self):
        return np.array(
            [
                self.step_count_down,
                self.world.agent.position[0],
                self.world.agent.position[1],
                (
                    self.world.get_resource_positions()[0][0]
                    if self.world.get_resource_is_active_status()[0]
                    else -1
                ),
                (
                    self.world.get_resource_positions()[0][1]
                    if self.world.get_resource_is_active_status()[0]
                    else -1
                ),
                (
                    self.world.get_resource_types()[0].subtype.value
                    if self.world.get_resource_is_active_status()[0]

                    else -1
                ),
                (
                    self.world.get_resource_timers()[0].remaining
                    if self.world.get_resource_is_active_status()[0]

                    else 0
                ),
            ],
            dtype=np.float32,
        )

    def _normalize_obs(self, obs):
        """
        Normalize observation to 0..1 while preserving sentinel values (-1) for absent resources.

        :param obs: raw observation array (shape: 7,) from _get_observation()
        Returns:
            normalized_obs: float32 np.array (7,) where:
            - regular features scaled 0..1
            - sentinel fields are -1.0 when absent, otherwise scaled 0..1
        """

        # Prepare output array
        normalized_obs = np.empty_like(obs, dtype=np.float32)
        # Create resource and non-resource indices
        resource_idx = np.where(self.resource_mask)[0]
        non_resource_idx = np.where(~self.resource_mask)[0]
        # Prep the non-resource indices
        normalized_obs[non_resource_idx] = (
            obs[non_resource_idx] / self._raw_high[non_resource_idx]
        )

        # If resource absent, keep -1, otherwise — normalize
        resource_values = obs[resource_idx]

        absent_mask = resource_values == -1.0
        present_mask = ~absent_mask

        # Keep absent resources
        normalized_obs[resource_idx[absent_mask]] = -1.0

        # Normalize only present ones
        normalized_obs[resource_idx[present_mask]] = (
            resource_values[present_mask] / self._raw_high[resource_idx[present_mask]]
        )

        return normalized_obs
