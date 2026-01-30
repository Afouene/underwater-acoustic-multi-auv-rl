import torch
import numpy as np
from stable_baselines3 import PPO

from environment2auv import AUV2DEnv2AUV   # your 2-AUV SB3-compatible env


# ---------------- Config ----------------
# TODO: update this path to the model you actually want to test
# e.g. "auv2d_runs_2auvs/models/PPO_large_model_2auv_final.zip"
MODEL_PATH = "auv2d_runs_2auvs_13nodes/models/PPO_small_model_2auv_final.zip"
#MODEL_PATH="auv2d_runs_2auvs_13nodes/best_model/PPO_large_model_2auv/best_model.zip"
#MODEL_PATH ="auv2d_runs_2auvs_10nodes/models/PPO_large_model_2auv_final.zip"
NUM_EPISODES = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Load env & model ----------------
env = AUV2DEnv2AUV()
model = PPO.load(MODEL_PATH, device=device)


print("\n" + "=" * 80)
print("TESTING 2-AUV PPO POLICY (SIMPLE MODE)")
print("=" * 80)

episode_rewards = []
final_mean_aoi = []
goals_reached_all = []  # list of arrays, one per episode


for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        step += 1

        if hasattr(env, "max_steps") and step >= env.max_steps:
            break

    episode_rewards.append(total_reward)

    # ---- final AoI ----
    if hasattr(env, "AoI"):
        final_mean_aoi.append(float(np.mean(env.AoI)))
    else:
        final_mean_aoi.append(np.nan)

    # ---- goal reached per AUV ----
    reached_flags = None

    if hasattr(env, "reached"):
        # expected shape (2,)
        reached_flags = np.array(env.reached, dtype=bool)
    elif hasattr(env, "pos") and hasattr(env, "goal") and hasattr(env, "goal_radius"):
        pos = np.array(env.pos)      # shape (2, 2) or (2, 3)
        goal = np.array(env.goal)    # shape (2,) or (3,)
        dists = np.linalg.norm(pos - goal, axis=-1)
        reached_flags = dists <= float(env.goal_radius)
    else:
        # fallback: unknown, mark as False
        reached_flags = np.array([False, False])

    goals_reached_all.append(reached_flags)

    print(
        f"EP {ep+1:02d} | "
        f"reward={total_reward:8.2f} | "
        f"steps={step:3d} | "
        f"mean_AoI={final_mean_aoi[-1]:6.2f} | "
        f"goals={reached_flags.tolist()}"
    )

    # assumes your 2-AUV env also has a trajectory plotting helper
    if hasattr(env, "plot_trajectory"):
        env.plot_trajectory()


# ---------------- Summary ----------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Avg reward:", float(np.mean(episode_rewards)))
print("Avg final AoI:", float(np.mean(final_mean_aoi)))

# goal success rate per AUV (over episodes)
goals_arr = np.stack(goals_reached_all, axis=0)  # shape (episodes, 2)
per_auv_success = np.mean(goals_arr, axis=0)

print("Goal success rate AUV 1:", float(per_auv_success[0]))
print("Goal success rate AUV 2:", float(per_auv_success[1]))
print("Joint success rate (both reached):", float(np.mean(np.all(goals_arr, axis=1))))
