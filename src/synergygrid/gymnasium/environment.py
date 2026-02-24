import gymnasium as gym
from gymnasium import spaces
import numpy as np
from synergygrid.core import GridWorld, AgentAction, DirectType
from synergygrid.rendering import PygameRenderer
from numpy.typing import NDArray
from typing import Any


class SynergyGridEnv(gym.Env):
    """
    SynergyGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # Metadata required by Gym.
    # "human" for Pygame visualization.
    # FPS caps the render() update rate; each call corresponds to one logic step, not full game fps.
    # Sub-loop in PygameRenderer.render() creates smooth animation between steps.
    metadata = {"render_modes": ["human"], "render_fps": 2}

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

        # Gymnasium also requires us to define the action space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(AgentAction))
        self._setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(
        self, *, seed=None, options=None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the world.
        self._init_episode_vars()
        self._world.reset(self.np_random)

        if self.render_mode == "human":
            self.render()

        obs = self._get_observation()

        # Return observation and info (not used)
        return self._normalize_obs(obs), {}

    def step(
        self, action: AgentAction
    ) -> tuple[NDArray[np.float32], int, bool, bool, dict[str, Any]]:
        # Perform action and adjust variables affected by it
        reward = self._world.perform_agent_action(AgentAction(action))
        self._step_count_down -= 1
        truncated = self._step_count_down <= 0
        terminated = self._world.agent.score <= 0

        if self.render_mode == "human":
            self.render()

        obs = self._get_observation()

        # Return observation, reward, terminated, truncated and info (not used)
        return self._normalize_obs(obs), reward, terminated, truncated, {}

    def render(self) -> None:
        self._renderer.render(
            self._world.agent.position,
            self._world.get_resource_is_active_status(),
            self._world.get_resource_positions(),
            self._world.get_resource_types(),
            self._world.agent.score,
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

    def _setup_obs_space(self) -> None:
        """
        Gymnasium requires an observation space definition. Here we represent the state as a flat
        vector. The space is used by Gymnasium to validate observations returned by reset() and step().

        Set up:
        - self._raw_low / self._raw_high: the original raw ranges (used for normalization)
        - self._sentinel_mask: boolean mask that indicates "absent" resources
        - self.observation_space: the normalized observation space that agents will see
        (0..1 for active features, -1 for absent resource fields)
        """
        # original raw bounds — match _get_observation()
        raw_low, raw_high = self._build_observation_bounds(False)

        # store raw bounds for use in normalize_obs()
        self._raw_high = raw_high

        # Absent resource mask: True where low == -1.0 (these fields mean "absent" when -1)
        # This mask will be used to keep -1 as a special value instead of normalizing it.
        self._resource_mask = raw_low == -1.0

        # normalized bounds — match _normalize_obs()
        # inactive resources keep -1 as a valid "low" value; active features map to 0..1
        low_norm, high_norm = self._build_observation_bounds(True)

        self.observation_space = spaces.Box(
            low=low_norm, high=high_norm, dtype=np.float32
        )

    def _build_observation_bounds(
        self, normalized: bool
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if normalized:
            agent_and_ep_low = 0.0
            no_resource = -1.0
            r_timer_low = 0.0
            max_steps = 1.0
            max_row = 1.0
            max_col = 1.0
            max_resource_type = 1.0
            r_timer_high = 1.0
        else:
            agent_and_ep_low = 0
            no_resource = -1
            r_timer_low = 0
            max_steps = self._max_steps
            max_row = self.grid_rows - 1
            max_col = self.grid_cols - 1
            max_resource_type = len(DirectType) - 1
            r_timer_high = (self.grid_rows - 1) + (self.grid_cols - 1)

        low = [agent_and_ep_low, agent_and_ep_low, agent_and_ep_low]
        low.extend(
            [no_resource, no_resource, no_resource, r_timer_low]
            * self.max_active_resources
        )
        high = [max_steps, max_row, max_col]
        high.extend(
            [max_row, max_col, max_resource_type, r_timer_high]
            * self.max_active_resources
        )

        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)

        if low_arr.shape != high_arr.shape:
            raise ValueError(
                f"low/high shape mismatch: {low_arr.shape} != {high_arr.shape}"
            )
        if np.any(high_arr <= low_arr):
            raise ValueError(
                "All high bounds must be greater than low bounds for raw ranges"
            )

        return low_arr, high_arr

    # === Global === #

    def _init_episode_vars(self) -> None:
        self._step_count_down = self._max_steps

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Build a flat observation vector dynamically based on max_active_resources.
        - Uses sentinel -1 for absent resource position/type and 0 for absent timers.
        """
        # Get step + agent info
        agent_row, agent_col = self._world.agent.position
        obs: list[float] = [self._step_count_down, agent_row, agent_col]

        # Cache resource info
        positions = self._world.get_resource_positions()
        types = self._world.get_resource_types()
        timers = self._world.get_resource_timers()
        active = self._world.get_resource_is_active_status()

        # For each resource slot, append (row, col, type, timer) or absent values
        for i in range(self.max_active_resources):
            if i < len(active) and active[i]:
                # Resource is active: extract real values
                pos = positions[i]
                r_type = types[i].subtype.value
                r_timer = timers[i].remaining
                obs.extend(
                    [float(pos[0]), float(pos[1]), float(r_type), float(r_timer)]
                )
            else:
                # Resource inactive or slot unused: sentinel values
                obs.extend([-1, -1, -1, 0])

        return np.array(obs, dtype=np.float32)

    def _normalize_obs(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize observation to 0..1 while preserving sentinel values (-1) for absent resources.

        :param obs: raw observation array (shape: 15,) from _get_observation()
        Returns:
            normalized_obs: float32 np.array (15,) where:
            - regular features scaled 0..1
            - sentinel fields are -1.0 when absent, otherwise scaled 0..1
        """

        # Prepare output array
        normalized_obs = np.empty_like(obs, dtype=np.float32)

        # Create resource and non-resource indices
        resource_idx = np.where(self._resource_mask)[0]
        non_resource_idx = np.where(~self._resource_mask)[0]

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
