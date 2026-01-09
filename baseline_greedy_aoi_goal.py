import numpy as np
from environment1auv import AUV2DEnv

# -------------------------
# Helpers: continuous [-1,1] <-> discrete index
# -------------------------
def cont_to_idx(u, K):
    """
    Map u in [-1, 1] to discrete index in {0,...,K-1}
    """
    u = float(np.clip(u, -1.0, 1.0))
    return int(np.round((u + 1.0) * 0.5 * (K - 1)))

def idx_to_cont(i, K):
    """
    Map discrete index to u in [-1,1]
    """
    if K <= 1:
        return 0.0
    return 2.0 * i / (K - 1) - 1.0


# -------------------------
# Greedy baseline (compatible with your env)
# -------------------------
def greedy_aoi_goal_policy(env: AUV2DEnv):
    """
    Returns an action compatible with:
      spaces.MultiDiscrete([K_theta, K_v, N, N])

    Strategy:
      - Move toward goal (discretized)
      - sel_wet: closest node (maximize harvesting)
      - sel_data: pick node with high AoI but also "more feasible"
                (prefer nodes that already have some energy and are close)
    """

    pos   = env.pos
    theta = env.theta
    goal  = env.goal
    N     = env.N

    # ==================================================
    # 1) MOTION: steer toward goal
    # ==================================================
    phi = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])

    # heading error wrapped to [-pi, pi]
    dtheta = phi - theta
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi

    # convert to normalized action in [-1, 1]
    # env.w_theta is max heading change per step (scaled by dtheta_norm)
    dtheta_norm = np.clip(dtheta / (env.w_theta + 1e-12), -1.0, 1.0)
    dtheta_idx  = cont_to_idx(dtheta_norm, env.K_theta)

    # Speed control: keep a reasonable speed, slow down when near borders
    # (prevents boundary penalty -20 and "do not move")
    margin = 0.7
    near_border = (
        (pos[0] < env.xmin + margin) or (pos[0] > env.xmax - margin) or
        (pos[1] < env.ymin + margin) or (pos[1] > env.ymax - margin)
    )

    # also slow down when close to goal to avoid overshooting
    dist_goal = np.linalg.norm(pos - goal)
    if dist_goal < 1.2:
        target_v = 0.3
    elif near_border:
        target_v = 0.65
    else:
        target_v = 1

    # convert desired speed to dv_norm:
    # v_new = clip(v + dv_norm*w_v, 0, v_max)  => dv_norm ≈ (target_v - v)/w_v
    dv_norm = (target_v - env.v) / (env.w_v + 1e-12)
    dv_norm = np.clip(dv_norm, -1.0, 1.0)
    dv_idx  = cont_to_idx(dv_norm, env.K_v)

    # ==================================================
    # 2) WET selection: closest node
    # ==================================================
    dists = np.linalg.norm(env.nodes - pos, axis=1)
    sel_wet = int(np.argmin(dists))

    # ==================================================
    # 3) DATA selection: AoI + "feasibility-ish"
    # ==================================================
    # We want high AoI, but also avoid always choosing a node that can't transmit.
    # Use a simple score: AoI - c1*distance + c2*log(energy+eps)
    # (no exact E_req computation needed; cheap and stable)
    eps = 1e-12
    score = env.AoI.astype(np.float64) - 0.8 * dists + 0.4 * np.log(env.E_nodes + eps)

    # optional: discourage selecting same node as WET every time (not required)
    # score[sel_wet] += 0.05

    sel_data = int(np.argmax(score))

    return np.array([dtheta_idx, dv_idx, sel_wet, sel_data], dtype=np.int64)


def run_baseline_episode(env, max_steps=200, verbose=True):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    if verbose:
        print("\n" + "="*80)
        print("GREEDY AoI + GOAL BASELINE (MultiDiscrete)")
        print("="*80)

    while not done and step < max_steps:
        action = greedy_aoi_goal_policy(env)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if verbose:
            dtheta_idx, dv_idx, sel_w, sel_d = action
            dtheta_norm = idx_to_cont(dtheta_idx, env.K_theta)
            dv_norm     = idx_to_cont(dv_idx, env.K_v)

            mean_AoI = np.mean(env.AoI)
            dist_goal = np.linalg.norm(env.pos - env.goal)

            print(f"\nSTEP {step}")
            print("-"*60)
            print(f"action idx = {action.tolist()}  |  (dθ~={dtheta_norm:+.2f}, dv~={dv_norm:+.2f})")
            print(f"pos = ({env.pos[0]:.2f}, {env.pos[1]:.2f})")
            print(f"θ = {env.theta:.2f}, v = {env.v:.2f}")
            print(f"sel_wet = {int(sel_w)}, sel_data = {int(sel_d)}")
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
