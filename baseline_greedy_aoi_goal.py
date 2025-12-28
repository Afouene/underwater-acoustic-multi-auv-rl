import numpy as np
from environment1auv import AUV2DEnv   # adjust path if needed


def greedy_aoi_goal_policy(env):
    """
    Returns a Dict action compatible with AUV2DEnv:
      - motion: [Δθ~, Δv~]
      - sel_wet: closest node
      - sel_data: node with max AoI
    """

    # ---------------- Motion: go toward goal ----------------
    pos = env.pos
    theta = env.theta
    goal = env.goal

    # Desired heading to goal
    phi = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])

    # Heading error (wrapped to [-pi, pi])
    dtheta = phi - theta
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi

    # Normalize heading change
    dtheta_norm = np.clip(dtheta / env.w_theta, -1.0, 1.0)

    # Speed control: try to keep constant forward motion
    dv_norm = 0.0   # constant speed baseline

    # ---------------- Scheduling ----------------
    # Data node: highest AoI
    sel_data = int(np.argmax(env.AoI))

    # Harvest node: closest node
    dists = np.linalg.norm(env.nodes - pos, axis=1)
    sel_wet = int(np.argmin(dists))

    return {
        "motion": np.array([dtheta_norm, dv_norm], dtype=np.float32),
        "sel_wet": sel_wet,
        "sel_data": sel_data,
    }


def run_baseline_episode(env, max_steps=200, verbose=True):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    if verbose:
        print("\n" + "="*80)
        print("GREEDY AoI + GOAL BASELINE")
        print("="*80)

    while not done and step < max_steps:
        action = greedy_aoi_goal_policy(env)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if verbose:
            mean_AoI = np.mean(env.AoI)
            dist_goal = np.linalg.norm(env.pos - env.goal)

            print(f"\nSTEP {step}")
            print("-"*60)
            print(f"pos = ({env.pos[0]:.2f}, {env.pos[1]:.2f})")
            print(f"θ = {env.theta:.2f}, v = {env.v:.2f}")
            print(f"sel_data = {action['sel_data']}, sel_wet = {action['sel_wet']}")
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
