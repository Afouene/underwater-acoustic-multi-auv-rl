import numpy as np
from environment2auv import MultiAUVEnvironment

# Number of episodes to average over
num_episodes = 100

# Storage for metrics
average_age_over_episodes               = []
average_energy_harvested_over_episodes  = []
jain_index_over_episodes                = []
avg_collision                           = []
avg_number_data_collected_over_episodes = []

for ep in range(num_episodes):
    env = MultiAUVEnvironment()
    obs = env.reset()
    done = False

    # Round‑robin indices for each AUV
    rr_idx1 = 0
    rr_idx2 = env.num_devices // 2    # start AUV2 halfway around the list
    # Track previous occurrence counts
    prev_occ = [0] * env.num_devices

    while not done:
        N = env.num_devices

        # --- AUV 1 target & direction ---
        target1 = rr_idx1 % N
        pos1    = env.auv_positions[0]
        diff1   = env.sensor_node_positions[target1] - pos1
        if abs(diff1[0]) >= abs(diff1[1]) and abs(diff1[0]) >= abs(diff1[2]):
            dir1 = 0 if diff1[0] > 0 else 1
        elif abs(diff1[1]) >= abs(diff1[2]):
            dir1 = 2 if diff1[1] > 0 else 3
        else:
            dir1 = 4 if diff1[2] > 0 else 5

        # --- AUV 2 target & direction ---
        target2 = rr_idx2 % N
        pos2    = env.auv_positions[1]
        diff2   = env.sensor_node_positions[target2] - pos2
        if abs(diff2[0]) >= abs(diff2[1]) and abs(diff2[0]) >= abs(diff2[2]):
            dir2 = 0 if diff2[0] > 0 else 1
        elif abs(diff2[1]) >= abs(diff2[2]):
            dir2 = 2 if diff2[1] > 0 else 3
        else:
            dir2 = 4 if diff2[2] > 0 else 5

        # Pack action: [dir1,W1,I1, dir2,W2,I2]
        action = np.array([dir1, target1, target1,
                           dir2, target2, target2], dtype=int)

        obs, reward, done, _ = env.step(action)

        # Advance each RR index only if that AUV just collected data
        if env.occurence[target1] > prev_occ[target1]:
            rr_idx1 += 1
        if env.occurence[target2] > prev_occ[target2]:
            rr_idx2 += 1

        prev_occ = env.occurence.copy()
    # Compute Jain's fairness
    sum_sq  = sum(x**2 for x in env.occurence)
    sum_val = sum(env.occurence)
    jain    = (sum_val**2) / (sum_sq * env.num_devices) if sum_sq else 0

    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(jain)
    avg_collision.append(env.num_collision)
    avg_number_data_collected_over_episodes.append(np.sum(env.occurence) / 2)

# Final report (Round‑Robin, 2 AUVs)
print(f"RR 2‑AUV, {env.num_devices} nodes: Avg AoI           = {np.mean(average_age_over_episodes):.3f}")
print(f"RR 2‑AUV, {env.num_devices} nodes: Avg Energy Harvest = {np.mean(average_energy_harvested_over_episodes):.3f}")
print(f"RR 2‑AUV, {env.num_devices} nodes: Avg Jain's Index   = {np.mean(jain_index_over_episodes):.3f}")
print(f"RR 2‑AUV, {env.num_devices} nodes: Avg Collisions     = {np.mean(avg_collision):.3f}")
print(f"RR 2‑AUV, {env.num_devices} nodes: Avg Data Collected = {np.mean(avg_number_data_collected_over_episodes):.3f}")
print(f"RR 2‑AUV, {env.num_devices} nodes: Total Steps        = {env.total_steps}")