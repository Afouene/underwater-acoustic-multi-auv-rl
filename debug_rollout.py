import numpy as np
import matplotlib.pyplot as plt

from environment1auv import AUV2DEnv   


def run_random_debug(seed=0):
    np.random.seed(seed)

    env = AUV2DEnv()
    obs = env.reset()

    print("\n" + "="*90)
    print("RANDOM POLICY DEBUG RUN")
    print("="*90)

    total_reward = 0.0

    for t in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        dtheta_i, dv_i, sel_w, sel_d = action

        dist_goal = np.linalg.norm(env.pos - env.goal)
        mean_aoi  = np.mean(env.AoI)

        # ----- Fairness (if defined) -----
        if np.sum(env.occ) > 0:
            s1 = np.sum(env.occ)
            s2 = np.sum(env.occ**2)
            jain = (s1**2) / (env.N * s2) if s2 > 0 else 0.0
        else:
            jain = 0.0

        print(f"\nSTEP {t}")
        print("-"*60)
        print(f"Action indices:")
        print(f"  dtheta_idx={dtheta_i}, dv_idx={dv_i}, sel_w={sel_w}, sel_d={sel_d}")

        print(f"AUV state:")
        print(f"  pos=({env.pos[0]:.2f},{env.pos[1]:.2f})  θ={env.theta:.2f} rad  v={env.v:.2f}")
        print(f"  dist_to_goal={dist_goal:.2f}")

        print(f"AUV energy:")
        print(f"  E_AUV={env.E:.2e}")

        print(f"Node energies:")
        print(f"  E_nodes={np.round(env.E_nodes, 6)}")

        print(f"AoI:")
        print(f"  AoI={env.AoI.astype(int)}  mean_AoI={mean_aoi:.2f}")

        print(f"Selection counters:")
        print(f"  occ={env.occ.astype(int)}  Jain={jain:.3f}")

        print(f"Reward:")
        print(f"  step_reward={reward:.3f}  cumulative={total_reward:.3f}")

        if done:
            print("\n--- TERMINATED ---")
            print(f"Reason: {'GOAL reached' if dist_goal <= env.goal_radius else 'max_steps or energy'}")
            break

    print("\n" + "="*90)
    print("EPISODE SUMMARY")
    print("="*90)
    print(f"Total reward = {total_reward:.3f}")
    print(f"Final mean AoI = {np.mean(env.AoI):.2f}")
    print(f"Final AUV energy = {env.E:.2e}")

    # ---- Trajectory plot ----
    env.plot_trajectory()


if __name__ == "__main__":
    run_random_debug(seed=1)
