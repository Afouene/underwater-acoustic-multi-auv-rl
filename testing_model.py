import torch
import gym
from stable_baselines3 import PPO
from environment1auv import AUVEnvironment  
import csv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path ='rician_model_7_nodes_1_auv/PPO_first_model_final.zip'
model_path = '10_nodes_1_auvs/best_model/best_model.zip'

# Create environment
env = AUVEnvironment()
model = PPO.load(model_path, env=env, device=device)

num_episodes = 100
num_auvs=1

# Separate lists for each AUV
auv1_positions = []

sensor_nodes = [list(node) for node in env.sensor_node_positions]  # Save sensor node positions

average_age_over_episodes = []
average_energy_harvested_over_episodes = []
jain_index_over_episodes = []

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
        auv1_positions.append(env.auv_position.copy())  

    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)

    Jain_index = ((sum_of_values**2) / (sum_of_squares * env.num_devices)) if sum_of_squares != 0 else 0
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(Jain_index)

print("This is for the average age for RL algorithm ", env.num_devices, " nodes ", np.mean(average_age_over_episodes))
print("Total Average number of communications RL 3D with", env.num_devices, "nodes ", np.sum(env.occurence)/num_auvs)
print("This is the average cumulative energy harvested for RL algorithm ", env.num_devices, " nodes ", np.mean(average_energy_harvested_over_episodes))
print("This is the average Jain's fairness index for RL algorithm ", env.num_devices, " nodes ", np.mean(jain_index_over_episodes))
print("This is the total nbr of steps for RL 3DD algorithm with", env.total_steps)

# Save AUV1 positions
'''with open('auv1_positions_5nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(auv1_positions)

# Save AUV2 positions
with open('auv2_positions_5nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(auv2_positions)

# Save sensor node positions
with open('sensor_nodes_5nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sensor_nodes)

print("CSV files saved: 'auv1_positions_5nodes.csv', 'auv2_positions_5nodes.csv', and 'sensor_nodes_5nodes.csv'")'''