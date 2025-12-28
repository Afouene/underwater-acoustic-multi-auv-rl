import numpy as np
from environment1auv import AUVEnvironment

num_episodes = 1

# Storage for metrics
average_age_over_episodes               = []
average_energy_harvested_over_episodes  = []
jain_index_over_episodes                = []
avg_number_data_collected_over_episodes = []

for ep in range(num_episodes):
    env = AUVEnvironment()
    obs = env.reset()
    done = False

    rr_idx   = 0
    # Keep track of previous occurrence counts to detect a new data collection
    prev_occ = [0] * env.num_devices

    while not done:
        N      = env.num_devices
        target = rr_idx % N
        node   = env.sensor_node_positions[target]
        pos    = env.auv_position
        diff1   = node - pos
        if abs(diff1[0]) >= abs(diff1[1]) and abs(diff1[0]) >= abs(diff1[2]):
            direction = 0 if diff1[0] > 0 else 1
        elif abs(diff1[1]) >= abs(diff1[2]):
            direction = 2 if diff1[1] > 0 else 3
        else:
            direction = 4 if diff1[2] > 0 else 5

      

        # WET and DATA both target the same node
        action = np.array([direction, target, target], dtype=int)
        print("action", action)
        obs, reward, done, _ = env.step(action)

        # if env.occurence[target] increased, we've just collected data → advance rr_idx
        if env.occurence[target] > prev_occ[target]:
            rr_idx += 1

        prev_occ = env.occurence.copy()

    # compute Jain’s fairness
    sum_sq  = sum(x**2 for x in env.occurence)
    sum_val = sum(env.occurence)
    jain    = (sum_val**2) / (sum_sq * env.num_devices) if sum_sq else 0

    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(jain)
    avg_number_data_collected_over_episodes.append(np.sum(env.occurence) / 1)

# final report
print(f"RR 1‑AUV, {env.num_devices} nodes: Avg AoI = {np.mean(average_age_over_episodes):.3f}")
print(f"RR 1‑AUV, {env.num_devices} nodes: Avg Energy Harvest = {np.mean(average_energy_harvested_over_episodes):.3f}")
print(f"RR 1‑AUV, {env.num_devices} nodes: Avg Jain's Index   = {np.mean(jain_index_over_episodes):.3f}")
print(f"RR 1‑AUV, {env.num_devices} nodes: Avg Data Collected = {np.mean(avg_number_data_collected_over_episodes):.3f}")
