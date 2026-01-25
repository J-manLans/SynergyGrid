import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

class GymAgentRunner:
    def __init__(self, env_name:str, seed:int=42, algorithm:str='PPO'):
        self.env_name = env_name
        self.seed = seed
        self.model = None
        self.algorithm = algorithm


    def train(self, timesteps=20000, iterations=11):
        '''Train a PPO agent on the environment.'''
        print('\nTraining agent...\n')

        model_dir = os.path.join('results', 'models', self.env_name)
        log_dir = os.path.join('results', 'logs', self.env_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        env = gym.make(self.env_name, render_mode=None)
        monitor_file = os.path.join(log_dir, "monitor.csv")
        env = Monitor(env, filename=monitor_file)

        if self.algorithm == 'PPO':
            self.model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device="cpu")
        else:
            raise NotImplementedError(f'{self.algorithm} not supported yet.')

        try:
            # This loop will keep training based on timesteps and iterations.
            # After the timesteps are completed, the model is saved and training continues for the next iteration.
            # Start another cmd prompt and launch Tensorboard: tensorboard --logdir results/logs/<env_name>
            # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
            for i in range(1, iterations + 1):
                print('Training starting for iteration:', i)

                # Train the model
                self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

                # Save the model
                save_path = os.path.join(model_dir, f"{self.algorithm}_{self.env_name}_{timesteps*i}")
                self.model.save(save_path)
                print(f'\nModel saved with {timesteps*i} time steps\n')
        finally:
            env.close()


    def agentRun(self, agent_path: str):
        '''Run the trained agent with human-visible rendering.'''
        env = gym.make(self.env_name, render_mode='human')
        model = PPO.load(os.path.join('results', 'models', self.env_name, f'{self.algorithm}_{agent_path}'), env=env)
        obs, info = env.reset()
        done = False

        while not done:
            # Predict action from the trained model
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Exit environment if terminated or truncated
            done = terminated or truncated

        env.close()


    def randomRun(self):
        '''Run a random actions with human-visible rendering.'''
        env = gym.make(self.env_name, render_mode='human')

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
                    print('Escape pressed! Exiting.')
                    done = True

        env.close()


if __name__ == '__main__':
    # Chose environment
    environments = {
        1: 'LunarLander-v3',
        2: 'CartPole-v1',
        3: 'Acrobot-v1',
        4: 'MountainCar-v0',
        5: 'CarRacing-v3',
        6: 'BipedalWalker-v3',
        7: 'CliffWalking-v1',
        8: 'FrozenLake-v1'
    }
    environment = environments[6]

    agent = True # Choose to use an agent or not
    training = False # Choose to train or run the agent
    model = f'{environment}_220000' # Model to load

    runner = GymAgentRunner(env_name=environment)

    if agent:
        if training:
            # Train agent
            runner.train()
        else:
            # Run agent
            runner.agentRun(model)
    else:
        # Run environment without agent
        runner.randomRun()