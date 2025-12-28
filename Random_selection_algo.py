import gym
import numpy as np
import csv
from environment1auv import AUVEnvironment  # Ensure this is correctly imported

# Create environment
env = AUVEnvironment()

num_episodes = 100 # Number of episodes to run
average_age_over_episodes = []
average_energy_harvested_over_episodes = []
jain_index_over_episodes = []
num_auvs=1
# Separate lists for AUV positions
auv1_positions = []
auv2_positions = []
sensor_nodes = [list(node) for node in env.sensor_node_positions]  # Store sensor node positions

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Random action selection
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Store AUV positions
        auv1_positions.append(env.auv_position.copy())
    # Calculate Jain's fairness index
    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)
    Jain_index = ((sum_of_values**2) / (sum_of_squares * env.num_devices)) if sum_of_squares != 0 else 0

    # Store results for each episode
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(Jain_index)

# Print final results
print("This is for the average age for Random Selection algorithm with", env.num_devices, "nodes:", np.mean(average_age_over_episodes))
print("Total Average number of communications Random Selection 3D with", env.num_devices, "nodes:", np.sum(env.occurence)/num_auvs)
print("This is the average cumulative energy harvested for Random Selection algorithm with", env.num_devices, "nodes:", np.mean(average_energy_harvested_over_episodes))
print("This is the average Jain's fairness index for Random Selection algorithm with", env.num_devices, "nodes:", np.mean(jain_index_over_episodes))
print("This is the total nbr of steps for Random Selection algorithm with", env.total_steps)


