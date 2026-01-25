import gymnasium as gym

class SynergyGridEnv(gym.Env):
    """
    A custom environment for SynergyGrid.
    """

    def __init__(self):
        pass

    def step(self, action):
        return None, 0, False, False, {}

    def reset(self,  *, seed: int | None = None, options: dict | None = None):
        return None, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
