from synergygrid.experiment import AgentRunner
import gymnasium as gym


def evaluate_agent(runner: AgentRunner, agent_steps: str, random_model: bool):
    """
    Run the benchmark with the specified model.

    :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
    """

    # Create the environment with human rendering and load the model
    env = gym.make(runner.environment, render_mode="human")

    # Define get_action() depending on type of model
    if random_model:

        def get_action(obs):
            # Sample a random action
            return env.action_space.sample()

    else:
        model = runner.get_model(agent_steps, env)

        def get_action(obs):
            # Predict action from the model
            return model.predict(obs)[0]

    # Reset the environment
    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        # Reset if resource is found or exit environment if truncated.
        if terminated: env.reset()
        done = truncated

    env.close()
