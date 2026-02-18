import argparse
from experiments import GymAgentRunner, environments, algorithms

def parse_args():
    parser = argparse.ArgumentParser(description="Run agent experiments.")

    parser.add_argument(
        "--env",
        type=int,
        default=0,
        choices=range(len(environments)),
        help="Environment index"
    )

    parser.add_argument(
        "--alg",
        type=int,
        default=0,
        choices=range(len(algorithms)),
        help="Algorithm index"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the agent"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Number of timesteps"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of iterations"
    )

    return parser.parse_args()

def main():
    # Chose environment
    # 0: CarRacing-v3
    # 1: LunarLander-v3
    # 2: BipedalWalker-v3
    # 3: CartPole-v1
    # 4: Acrobot-v1
    # 5: MountainCar-v0
    # 6: CliffWalking-v1
    # 7: FrozenLake-v1

    #env = list(environments.keys())
    #environment = env[0]
    
    # Choose algorithm to use for training (also needed when running agent)
    # 0: PPO
    # 1: DQN

    #algorithm = alg[0]

    args = parse_args()
    # pyhon -m experiments --env 0 --alg 0 --train --timesteps 10000 --iterations 10000

    environment = list(environments.keys())[args.env] # Choose the environment
    algorithm = list(algorithms.keys())[args.alg] # Choose the algorithm
    training = args.train # Choose to train or run the agent
    timesteps = args.timesteps # Choose the number of timesteps
    iterations = args.iterations # Choose the number of iterations
    agent = True  # Choose to use an agent or not
    runner = GymAgentRunner(environment=environment, algorithm=algorithm)

    if agent:
        if training:
            # Train agent
            runner.train(continue_training=continue_training, agent_steps=agent_steps, timesteps=timesteps, iterations=iterations)
        else:
            # Run environment with agent
            runner.agentRun(agent_steps)
    else:
        # Run environment without agent
        runner.randomRun()

if __name__ == "__main__":
    main()
