from stable_baselines3 import PPO, DQN, A2C
import yaml
from importlib import resources
from pydantic import BaseModel

# ================= #
#      Models       #
# ================= #


class GridWorldConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_resources: int
    max_tier: int


class ObservationConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_steps: int


class RendererConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    fps: int


class AgentConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    starting_score: int


class RunConfig(BaseModel, frozen=True):
    grid_world: GridWorldConf
    observation_handler: ObservationConf
    renderer: RendererConf
    agent: AgentConf


# ================= #
#      Methods      #
# ================= #


def load_config(yaml_file, model_class):
    with resources.open_text("syn_grid.config", yaml_file) as f:
        raw = yaml.safe_load(f)
    return model_class(**raw)


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
