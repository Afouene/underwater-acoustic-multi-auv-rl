import numpy as np
from collections import defaultdict

from environment1auv import AUV2DEnv   # <-- adjust import if needed


def run_random_episode(env, max_steps=200, seed=0):
    np.random.seed(seed)

    obs = env.reset()
    done = False
    step = 0

    print("\n" + "="*80)
    print("RANDOM POLICY DEBUG RUN")
    print("="*80)

    while not done and step < max_steps:
        # ---- Sample random Dict action ----
        action = {
            "motion": env.action_space["motion"].sample(),
            "sel_wet": env.action_space["sel_wet"].sample(),
            "sel_data": env.action_space["sel_data"].sample(),
        }

        obs, reward, done, _ = env.step(action)

        # ---- Extract diagnostics ----
        pos = env.pos
        theta = env.theta
        v = env.v

        mean_AoI = np.mean(env.AoI)

        sum_occ = np.sum(env.occ)
        sum_sq = np.sum(env.occ ** 2)
        Jain = (sum_occ**2)/(env.N*sum_sq) if sum_sq > 0 else 0.0

        dist_goal = np.linalg.norm(env.pos - env.goal)

        # ---- Reward components (recomputed for clarity) ----
        smooth_penalty = 0.05 * (abs(action["motion"][0]) + abs(action["motion"][1]))
        aoi_penalty = mean_AoI
        fairness_penalty = env.lambda_fair * (1 - Jain)
        goal_penalty = 0.1 * dist_goal

        # ---- Print step summary ----
        print(f"\nSTEP {step}")
        print("-"*60)
        print(f"Action:")
        print(f"  Δθ~ = {action['motion'][0]: .3f}, Δv~ = {action['motion'][1]: .3f}")
        print(f"  sel_wet  = {action['sel_wet']}")
        print(f"  sel_data = {action['sel_data']}")

        print(f"\nState:")
        print(f"  pos = ({pos[0]:.2f}, {pos[1]:.2f})")
        print(f"  θ = {theta:.2f} rad, v = {v:.2f}")
        print(f"  Energy AUV = {env.E:.2e}")

        print(f"\nMetrics:")
        print(f"  Mean AoI = {mean_AoI:.2f}")
        print(f"  Jain fairness = {Jain:.3f}")
        print(f"  Distance to goal = {dist_goal:.2f}")

        print(f"\nReward breakdown:")
        print(f"  - Smoothness penalty   = {-smooth_penalty:.3f}")
        print(f"  - AoI penalty          = {-aoi_penalty:.3f}")
        print(f"  - Fairness penalty     = {-fairness_penalty:.3f}")
        print(f"  - Goal distance penalty= {-goal_penalty:.3f}")
        print(f"  --------------------------------")
        print(f"  TOTAL reward           = {reward:.3f}")

        step += 1

    print("\n" + "="*80)
    print("EPISODE FINISHED")
    print("="*80)

    env.plot_trajectory()


if __name__ == "__main__":
    env = AUV2DEnv()
    run_random_episode(env, max_steps=100, seed=42)
