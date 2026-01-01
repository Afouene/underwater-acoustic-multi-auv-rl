import numpy as np
from environment1auv import AUV2DEnv   # adjust filename if needed


def greedy_aoi_goal_policy(env):
    """
    Greedy AoI + Goal baseline compatible with SCORE-BASED action space.

    Action =
      [dtheta_norm,
       dv_norm,
       score_w[0..N-1],
       score_d[0..N-1]]
    """

    pos   = env.pos
    theta = env.theta
    goal  = env.goal
    N     = env.N

    # ==================================================
    # Motion: go toward goal
    # ==================================================
    phi = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])

    dtheta = phi - theta
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
    dtheta_norm = np.clip(dtheta / env.w_theta, -1.0, 1.0)

    # constant speed
    dv_norm = 0.0

    # ==================================================
    # Scheduling (greedy rules)
    # ==================================================

    # DATA node = highest AoI
    sel_data = int(np.argmax(env.AoI))

    # WET node = closest node
    dists = np.linalg.norm(env.nodes - pos, axis=1)
    sel_wet = int(np.argmin(dists))

    # ==================================================
    # Score vectors (argmax selection)
    # ==================================================
    score_w = -np.ones(N, dtype=np.float32)
    score_d = -np.ones(N, dtype=np.float32)

    score_w[sel_wet] = 1.0
    score_d[sel_data] = 1.0

    # ==================================================
    # Final action vector
    # ==================================================
    action = np.concatenate(
        ([dtheta_norm, dv_norm], score_w, score_d),
        axis=0
    ).astype(np.float32)

    return action


def run_baseline_episode(env, max_steps=200, verbose=True):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    if verbose:
        print("\n" + "="*80)
        print("GREEDY AoI + GOAL BASELINE (SCORE-BASED)")
        print("="*80)

    while not done and step < max_steps:
        action = greedy_aoi_goal_policy(env)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if verbose:
            mean_AoI = np.mean(env.AoI)
            dist_goal = np.linalg.norm(env.pos - env.goal)

            sel_w = int(np.argmax(action[2 : 2 + env.N]))
            sel_d = int(np.argmax(action[2 + env.N : 2 + 2*env.N]))

            print(f"\nSTEP {step}")
            print("-"*60)
            print(f"pos = ({env.pos[0]:.2f}, {env.pos[1]:.2f})")
            print(f"θ = {env.theta:.2f}, v = {env.v:.2f}")
            print(f"sel_wet = {sel_w}, sel_data = {sel_d}")
            print(f"Mean AoI = {mean_AoI:.2f}")
            print(f"Distance to goal = {dist_goal:.2f}")
            print(f"Reward = {reward:.3f}")

        step += 1

    if verbose:
        print("\n" + "="*80)
        print(f"EPISODE FINISHED | Total reward = {total_reward:.2f}")
        print("="*80)

    env.plot_trajectory()
    return total_reward


if __name__ == "__main__":
    env = AUV2DEnv()
    run_baseline_episode(env, max_steps=200, verbose=True)
