from pydantic import BaseModel

###########################
#    Run Configuration    #
###########################


class GridWorldConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_resources: int
    max_tier: int


class RendererConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int


class DroidConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    starting_score: int


class NegativeConf(BaseModel, frozen=True):
    reward: float
    cool_down: int


class TierConf(BaseModel, frozen=True):
    linear_reward_growth: bool
    step_wise_scoring: bool
    growth_factor: float
    base_reward: float
    cool_down: int


###########################
#    Obs Configuration    #
###########################


class ObservationConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_steps: int


###########################
#    Top Configurations   #
###########################


class RunConfig(BaseModel, frozen=True):
    grid_world_conf: GridWorldConf
    renderer_conf: RendererConf
    droid_conf: DroidConf
    negative_resource_conf: NegativeConf
    tier_resource_conf: TierConf


class ObsConfig(BaseModel, frozen=True):
    observation_handler: ObservationConf


class AgentConfig(BaseModel, frozen=True): ...
