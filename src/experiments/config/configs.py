from stable_baselines3 import PPO, DQN

environments = {
    "CarRacing-v3": {
        "policy": "CnnPolicy",
        "device": "cuda",
        "learning_rate": 1e-4,
        "gamma": 0.995,
        "gae_lambda": 0.97
    },"LunarLander-v3": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "BipedalWalker-v3": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "CartPole-v1": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "Acrobot-v1": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "MountainCar-v0": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "CliffWalking-v1": {
        "policy": "MlpPolicy",
        "device": "cpu"
    }, "FrozenLake-v1": {
        "policy": "MlpPolicy",
        "device": "cpu",
        "learning_rate": 1e-2,
        "n_steps": 32,
        "batch_size": 32,
        "buffer_size": 5000,
        "learning_starts": 100,
        "target_update_interval": 100
    }
}

algorithms = {
    "PPO": PPO,
    "DQN": DQN
}