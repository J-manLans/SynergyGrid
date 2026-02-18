from experiments import GymAgentRunner, environments, algorithms

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
    env = list(environments.keys())
    environment = env[1]
    # Choose algorithm to use for training (also needed when running agent)
    # 0: PPO
    # 1: DQN
    alg = list(algorithms.keys())
    algorithm = alg[0]
    agent = True  # Choose to use an agent or not
    training = False  # Choose to train or run the agent
    timesteps = 220000  # Num of timesteps for training or model selection when running
    iterations = 6  # Number of training iterations

    runner = GymAgentRunner(environment=environment, algorithm=algorithm)

    if agent:
        if training:
            # Train agent
            runner.train(timesteps=timesteps, iterations=iterations)
        else:
            # Run environment with agent
            runner.agentRun(f"{timesteps}")
    else:
        # Run environment without agent
        runner.randomRun()

if __name__ == "__main__":
    main()
