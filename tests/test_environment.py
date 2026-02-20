import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from synergygrid.gymnasium.environment import SynergyGridEnv


class TestEnvironment:
    @pytest.fixture
    def env(self):
        """Fixture for a fresh SynergyGridEnv"""
        return SynergyGridEnv()

    def test_check_env(self):
        env = gym.make("synergy_grid-v0", render_mode="human")
        check_env(env.unwrapped)
