from syn_grid.config.configs import load_config
from syn_grid.config.models import RunConfig, ObsConfig
from syn_grid.runners.agent_runner.agent_runner import AgentRunner
from syn_grid.runners.agent_runner.train_agent import train_agent
from syn_grid.runners.agent_runner.evaluate_agent import evaluate_agent
from syn_grid.utils.parse_args import parse_args
from syn_grid.runners.human_runner.human_runner import HumanRunner

import sys


def main():
    run_conf = load_config(RunConfig)
    obs_conf = load_config(ObsConfig)

    if len(sys.argv) == 1:
        # Pick algorithm to train or evaluate
        algorithm = 0
        # Choose to use an agent or just random sampling (for debugging the environment)
        agent = True
        # If we want to test the game our selves
        # Choose to train or run the agent
        training = False
        # Human control to test the game
        human_control = True
        # Continue training from a saved model
        continue_training = False
        # Model that we shall continue to train
        agent_steps = "1484800"
        # Num of timesteps for training or model selection when running
        timesteps = 50000
        # Number of training iterations
        iterations = 30
    else:
        args = parse_args()  # python -m experiments -h for info
        algorithm = args.alg
        agent = args.agent
        training = args.train
        human_control = args.human_controls
        continue_training = args.cont
        agent_steps = args.steps
        timesteps = args.timesteps
        iterations = args.iterations

    if human_control:
        runner = HumanRunner(run_conf, obs_conf.observation_handler.max_steps)
        runner.human_player_loop()
    else:
        runner = AgentRunner(algorithm, "-1_reward_tier_1", run_conf, obs_conf)

        if training:
            # Train agent
            train_agent(
                runner,
                continue_training=continue_training,
                agent_steps=agent_steps,
                timesteps=timesteps,
                iterations=iterations,
            )
        else:
            # Run environment with agent
            evaluate_agent(runner, agent_steps, agent)


if __name__ == "__main__":
    main()
