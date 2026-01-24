import gymnasium as gym
import pygame

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

env = gym.make(environments[9], render_mode='human')
obs, info = env.reset(seed=42)
done = False

while not done:
    # Sample a random action and perform a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Reset environment if terminated or truncated
    if terminated or truncated:
        obs = env.reset()

    # Check for ESC key press to exit
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            print("Escape pressed! Exiting.")
            done = True

env.close()
