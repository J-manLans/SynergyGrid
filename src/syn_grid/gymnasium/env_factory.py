import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.registration import registry, register
from syn_grid.config.configs import load_config, RunConfig


def register_env():
    """Register the SynergyGrid Gym environment. Once registered, the id is usable in gym.make()."""

    if "syn_grid-v0" not in registry:
        register(
            id="syn_grid-v0",
            entry_point="syn_grid.gymnasium.environment:SYNGridEnv",
        )


def make(render_mode: str | None) -> Env:
    run_conf = load_config("run_config.yaml", RunConfig)

    return gym.make("syn_grid-v0", render_mode=render_mode, run_conf=run_conf)
