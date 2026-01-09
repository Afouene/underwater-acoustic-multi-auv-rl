import torch
import numpy as np
from stable_baselines3 import PPO

from environment1auv import AUV2DEnv   # your SB3-compatible env


# ---------------- Config ----------------
MODEL_PATH = "auv2d_runs/models/PPO_large_model_final.zip"
MODEL_PATH ="auv2d_runs_new_aoi_update/models/PPO_large_model_final.zip"
NUM_EPISODES = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Load env & model ----------------
env = AUV2DEnv()
model = PPO.load(MODEL_PATH, device=device)


print("\n" + "=" * 80)
print("TESTING PPO POLICY (SIMPLE MODE)")
print("=" * 80)

episode_rewards = []
final_mean_aoi = []
goal_reached = []


for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        total_reward += reward
        step += 1

        if step >= env.max_steps:
            break

    episode_rewards.append(total_reward)
    final_mean_aoi.append(np.mean(env.AoI))
    goal_reached.append(
        np.linalg.norm(env.pos - env.goal) <= env.goal_radius
    )

    print(
        f"EP {ep+1:02d} | "
        f"reward={total_reward:8.2f} | "
        f"steps={step:3d} | "
        f"mean_AoI={final_mean_aoi[-1]:6.2f} | "
        f"goal={goal_reached[-1]}"
    )

    env.plot_trajectory()


# ---------------- Summary ----------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Avg reward:", np.mean(episode_rewards))
print("Avg final AoI:", np.mean(final_mean_aoi))
print("Goal success rate:", np.mean(goal_reached))
