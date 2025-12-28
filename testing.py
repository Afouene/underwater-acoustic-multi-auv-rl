import torch
import gym
from stable_baselines3 import PPO
from environment2auv import MultiAUVEnvironment  
import csv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path='rician_dir_5_nodes_2_auvs/best_model/best_model.zip'
model_path='10_nodes_2_auvs/best_model/best_model.zip'

# Create environment
env = MultiAUVEnvironment()
model = PPO.load(model_path, env=env, device=device)

num_episodes = 100
num_auvs=2
# Separate lists for each AUV
auv1_positions = []
auv2_positions = []

sensor_nodes = [list(node) for node in env.sensor_node_positions]  # Save sensor node positions

average_age_over_episodes = []
average_energy_harvested_over_episodes = []
jain_index_over_episodes = []
avg_collision = []
avg_number_data_collected_over_episodes = []
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward
        #env.render()
        # Separate AUV positions
        auv1_positions.append(env.auv_positions[0].copy())  
        auv2_positions.append(env.auv_positions[1].copy())  
    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)

    Jain_index = ((sum_of_values**2) / (sum_of_squares * env.num_devices)) if sum_of_squares != 0 else 0
    #print('auvs positions',env.auv_positions)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(Jain_index)
    avg_collision.append(env.num_collision)
    avg_number_data_collected_over_episodes.append(np.sum(env.occurence)/num_auvs)

# Final report (RL PPO, 2 AUVs)
# Final report (RL PPO, 2 AUVs)
print(f"RL 2‑AUV, {env.num_devices} nodes: Avg AoI           = {np.mean(average_age_over_episodes):.3f}")
print(f"RL 2‑AUV, {env.num_devices} nodes: Avg Energy Harvest = {np.mean(average_energy_harvested_over_episodes):.3f}")
print(f"RL 2‑AUV, {env.num_devices} nodes: Avg Jain's Index   = {np.mean(jain_index_over_episodes):.3f}")
print(f"RL 2‑AUV, {env.num_devices} nodes: Avg Collisions     = {np.mean(avg_collision):.3f}")
print(f"RL 2‑AUV, {env.num_devices} nodes: Avg Data Collected = {np.mean(avg_number_data_collected_over_episodes):.3f}")
print(f"RL 2‑AUV, {env.num_devices} nodes: Total Steps        = {env.total_steps}")

