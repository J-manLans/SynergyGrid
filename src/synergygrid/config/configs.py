from stable_baselines3 import PPO, DQN, A2C

environment = {
    "synergy_grid-v0": {
        "policy": "MultiInputPolicy",
        "device": "cpu",
        "ent_coef": 0.02,  # exploration
    }
}

algorithms = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
"""
Choose algorithm to use for training (also needed when running agent)
    0: PPO
    1: DQN
    2: A2C
"""
