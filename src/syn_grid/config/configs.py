from syn_grid.utils.paths import get_package_path

from stable_baselines3 import PPO, DQN, A2C
import yaml

# ================= #
#      Methods      #
# ================= #

# TODO: this is awful, wanna puke looking at it. Will make a YamlConfig class I think where I keep
# all yaml related stuff. But tomorrow, it's late now...


def load_config(model_class):
    path = get_package_path("config", "configs.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return model_class(**raw)


def update_from_args(): ...


# ================= #
#      Old shit     #
# ================= #

agent_config = {
    "policy": "MultiInputPolicy",
    "device": "cpu",
    "ent_coef": 0.02,  # exploration
}

algorithms = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
"""
Choose algorithm to use for training (also needed when running agent)
    0: PPO
    1: DQN
    2: A2C
"""
