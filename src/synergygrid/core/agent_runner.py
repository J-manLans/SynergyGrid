import gymnasium as gym
import pygame
import os
import datetime
import sys
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from synergygrid import environment, algorithms


class AgentRunner:
    def __init__(self, environment: str, algorithm: str):
        self.environment = environment
        self.model = None
        self.algorithm = algorithm
        self.AlgorithmClass = algorithms.get(self.algorithm, {})
        if not self.AlgorithmClass:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(
        self, continue_training=False, agent_steps="", timesteps=20000, iterations=10
    ):
        '''
        Train an agent, either from scratch or by continuing from a saved checkpoint.

        :param continue_training: If True, the training continues from an existing model checkpoint.
        :type continue_training: bool
        :param agent_steps: The specific checkpoint steps of the model to continue training from.
        :type agent_steps: str
        :param timesteps: Number of steps to train before saving the agent.
        :type timesteps: int
        :param iterations: Number of training loops, each consisting of `timesteps` steps.
        :type iterations: int
        '''

        # Get current date and time for unique file naming
        date = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        # Create directories for saving models and logs
        model_dir = os.path.join("results", "models", self.environment)
        log_dir = os.path.join("results", "logs", self.environment)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the training environment
        env = gym.make(self.environment, render_mode=None)
        # Wrap the environment with a Monitor for logging.
        # The created csv is needed for plotting our own graphs with matplotlib later.
        monitor_file = os.path.join(
            log_dir, f"{self.environment}_{self.algorithm}_{date}.csv"
        )
        env = Monitor(env, filename=monitor_file)

        if continue_training:
            # Get the model with the desired steps to continue its training
            agent_path = self.__get_agent(agent_steps)
            print("Loading existing training data", agent_path)
            self.model = self.AlgorithmClass.load(agent_path, env=env)
        else:
            # Initialize a fresh model
            self.model = self.AlgorithmClass(
                env=env,
                verbose=1,
                tensorboard_log=log_dir,
                **environment.get(self.environment, {}),
            )

        try:
            # This loop will keep training based on timesteps and iterations.
            # After the timesteps are completed, the model is saved and training
            # continues for the next iteration. When training is done, start another
            # cmd prompt and launch Tensorboard:
            # tensorboard --logdir results/logs/<env_name>
            # Once Tensorboard is loaded, it will print a URL. Follow the URL to see
            # the status of the training.
            for i in range(1, iterations + 1):
                print(
                    f"\nTraining starting for iteration: {i}, environment: {self.environment}\n"
                )

                # Train the model
                self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

                # Save the model
                save_path = os.path.join(
                    model_dir,
                    f"{self.algorithm}_{self.environment}_{self.model.num_timesteps}_{date}",
                )
                self.model.save(save_path)
                print(f"\nModel saved with {self.model.num_timesteps} time steps")
        finally:
            env.close()

    def agentRun(self, agent_steps: str):
        '''
        Run the benchmark with the specified model.

        :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
        :type agent_steps: str
        '''

        model_dir = self.__get_agent(agent_steps)
        # Create the environment with human rendering and load the model
        env = gym.make(self.environment, render_mode="human")
        model = self.AlgorithmClass.load(model_dir, env=env)

        # Reset the environment, just in case
        obs, info = env.reset()

        done = False
        while not done:
            # Predict action from the trained model
            action, _ = model.predict(obs)
            # Convert action to int if the action space is discrete (simple
            # environments like FrozenLake or CartPole require this)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(action)

            obs, _, terminated, truncated, _ = env.step(action)

            # Exit environment if terminated or truncated. ESC exit doesn't work with
            # models because of the rendering loop
            done = terminated or truncated

        env.close()

    def randomRun(self):
        """Run a random actions with human-visible rendering."""

        # Create the environment with human rendering
        env = gym.make(self.environment, render_mode="human")

        # Reset the environment, just in case
        obs, info = env.reset()

        done = False
        while not done:
            # Sample a random action and perform a step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Reset environment if terminated or truncated
            if terminated or truncated:
                obs, info = env.reset()

            # Check for ESC key press to exit
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("Escape pressed! Exiting.")
                    done = True

        env.close()

    def __get_agent(self, agent_steps: str) -> Path:
        """Create a path to match the latest model of the specified timesteps"""
        if agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        base_dir = Path("results") / "models" / self.environment
        file_name = f"{self.algorithm}_{self.environment}_{agent_steps}*"
        return list(base_dir.glob(file_name))[-1]
