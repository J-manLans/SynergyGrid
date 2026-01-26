import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

plots_dir = os.path.join("results", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Set path to the monitor CSV files
csv_files = glob.glob('results/logs/BipedalWalker-v3/*.csv')

plt.figure(figsize=(10,6))

for file in csv_files:
    # Read CSV (Monitor have a header with r,l,t)
    data = pd.read_csv(file, comment='#')
    rewards = data['r']  # Total reward per episode
    plt.plot(rewards, label=file.split('/')[-1])  # Plot and use filename as label

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episod Rewards from Multiple Monitor CSVs')
plt.legend()
plt.savefig('results/plots/reward_plot.png')
plt.close()