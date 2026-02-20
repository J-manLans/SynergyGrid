import gymnasium as gym
import pygame
import datetime
import sys
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from synergygrid import environment, algorithms


# TODO: split this up in run and train files and move from core into scripts or something
class AgentRunner:
    def __init__(self, environment: str, algorithm: str):
        self.environment = environment
        self.algorithm = algorithm
        self.AlgorithmClass = algorithms.get(self.algorithm, {})
        if not self.AlgorithmClass:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(
        self, continue_training=False, agent_steps="", timesteps=20000, iterations=10
    ):
        """
        Train an agent, either from scratch or by continuing from a saved checkpoint.

        :param continue_training: If True, the training continues from an existing model checkpoint.
        :type continue_training: bool
        :param agent_steps: The specific checkpoint steps of the model to continue training from.
        :type agent_steps: str
        :param timesteps: Number of steps to train before saving the agent.
        :type timesteps: int
        :param iterations: Number of training loops, each consisting of `timesteps` steps.
        :type iterations: int
        """

        # Get current date and time for unique file naming
        date = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        # Create directories for saving models and logs
        model_dir = Path("results/models") / self.environment
        log_dir = Path("results/logs") / self.environment

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Create and wrap the training environment
        env = gym.make(self.environment, render_mode=None)
        # Wrap the environment with a Monitor for logging.
        # The created csv is needed for plotting our own graphs with matplotlib later.
        monitor_file = Path(log_dir) / f"{self.environment}_{self.algorithm}_{date}.csv"
        env = Monitor(env, filename=str(monitor_file))

        model = None
        if continue_training:
            print("Loading existing training data")
            # Get the model with the desired steps to continue its training
            model = self.__get_model(agent_steps, env)
        else:
            # Initialize a fresh model
            model = self.AlgorithmClass(
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
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

                # Save the model
                save_path = (
                    Path(model_dir)
                    / f"{self.algorithm}_{self.environment}_{model.num_timesteps}_{date}"
                )
                model.save(save_path)
                print(f"\nModel saved with {model.num_timesteps} time steps")
        finally:
            env.close()

    def evaluate(self, agent_steps: str):
        """
        Run the benchmark with the specified model.

        :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
        """

        # Create the environment with human rendering and load the model
        env = gym.make(self.environment, render_mode="human")
        model = self.__get_model(agent_steps, env)

        # Reset the environment
        obs, _ = env.reset()

        done = False
        while not done:
            # Predict action from the trained model
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            # Reset if resource is found and exit environment if truncated.
            if terminated:
                env.reset()
            done = truncated

        env.close()

    def randomRun(self):
        """Run a random actions with human-visible rendering."""

        # Create the environment with human rendering
        env = gym.make(self.environment, render_mode="human")

        # Reset the environment
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

    def __get_model(self, agent_steps: str, env):
        """Create a path to match the latest model of the specified timesteps and load it"""

        if agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        base_dir = Path("results/models") / self.environment
        file_name = f"{self.algorithm}_{self.environment}_{agent_steps}*"
        return self.AlgorithmClass.load(list(base_dir.glob(file_name))[-1], env=env)
