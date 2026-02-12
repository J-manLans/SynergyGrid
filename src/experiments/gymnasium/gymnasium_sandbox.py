import gymnasium as gym
import pygame
import os
import datetime
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from experiments import environments, algorithms

class GymAgentRunner:
    def __init__(self, environment: str, algorithm: str):
        self.environment = environment
        self.model = None
        self.algorithm = algorithm
        self.AlgorithmClass = algorithms.get(self.algorithm, {})
        if not self.AlgorithmClass:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, timesteps=20000, iterations=11):
        """Train a PPO agent on the environment."""

        # Get current date and time for unique file naming
        date = datetime.datetime.now().strftime("%y.%m.%d_%H:%M:%S")

        # Create directories for saving models and logs
        model_dir = os.path.join("results", "models", self.environment)
        log_dir = os.path.join("results", "logs", self.environment)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Get the latest model
        latest_model_path = os.path.join(
            model_dir, f"{self.algorithm}_{self.environment}_latest"
        )

        # Create and wrap the training environment
        env = gym.make(self.environment, render_mode=None)
        # Wrap the environment with a Monitor for logging.
        # The created csv is needed for plotting our own graphs with matplotlib later.
        monitor_file = os.path.join(
            log_dir, f"{self.environment}_{self.algorithm}_{date}.csv"
        )
        env = Monitor(env, filename=monitor_file)

        try:
            # Initialize the model
            if os.path.exists(latest_model_path + ".zip"):
                print("Loading existing training data", latest_model_path)
                self.model = self.AlgorithmClass.load(latest_model_path, env=env)
            else:
                print("Creating new training data")
                self.model = self.AlgorithmClass(
                    env=env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    **environments.get(self.environment, {}),
                )

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
                    f"Current timestep: ", self.model.num_timesteps
                )

                # Train the model
                self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

                # Save the model
                self.model.save(latest_model_path)
                #versioned_path = os.path.join(
                #    model_dir,
                #    #f"{self.algorithm}_{self.environment}_{timesteps*i}_{date}",
                #    f"{self.algorithm}_{self.environment}_{self.model.num_timesteps}"
                #)
                #self.model.save(versioned_path)
        finally:
            env.close()

    def agentRun(self, agent_steps: str):
        """Run the trained agent with human-visible rendering."""

        # Create a path to match the latest model of the specified timesteps
        base_dir = Path("results")/ "models"/self.environment
        #file_name = f"{self.algorithm}_{self.environment}_{agent_steps}*"
        file_name = f"{self.algorithm}_{self.environment}_latest.zip"
        model_dir = list(base_dir.glob(file_name))[-1]
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

            obs, reward, terminated, truncated, info = env.step(action)

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
