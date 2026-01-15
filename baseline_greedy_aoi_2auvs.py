import numpy as np
from environment2auv import AUV2DEnv2AUV   # <-- update filename if needed


# -------------------------
# Helpers: continuous [-1,1] <-> discrete index
# -------------------------
def cont_to_idx(u, K):
    u = float(np.clip(u, -1.0, 1.0))
    return int(np.round((u + 1.0) * 0.5 * (K - 1)))

def idx_to_cont(i, K):
    if K <= 1:
        return 0.0
    return 2.0 * i / (K - 1) - 1.0


# ==================================================
# Greedy policy for ONE AUV (used twice)
# ==================================================
def greedy_single_auv(env, i, other_i):
    """
    Greedy control for AUV i in a 2-AUV environment.
    """

    pos   = env.pos[i]
    theta = env.theta[i]
    goal  = env.goal
    N     = env.N

    # --------------------------------------------------
    # 1) MOTION: steer toward goal + mild collision avoidance
    # --------------------------------------------------
    phi = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
    dtheta = (phi - theta + np.pi) % (2*np.pi) - np.pi

    # simple repulsion if too close to the other AUV
    d_ij = np.linalg.norm(env.pos[i] - env.pos[other_i])
    if d_ij < 1.2:
        repulse = np.arctan2(
            pos[1] - env.pos[other_i][1],
            pos[0] - env.pos[other_i][0]
        )
        dtheta += 0.4 * ((repulse - theta + np.pi) % (2*np.pi) - np.pi)

    dtheta_norm = np.clip(dtheta / (env.w_theta + 1e-12), -1.0, 1.0)
    dtheta_idx  = cont_to_idx(dtheta_norm, env.K_theta)

    # --------------------------------------------------
    # Speed control
    # --------------------------------------------------
    dist_goal = np.linalg.norm(pos - goal)

    if dist_goal < 1.2:
        target_v = 0.3
    elif d_ij < 1.2:
        target_v = 0.5
    else:
        target_v = 1.0

    dv_norm = (target_v - env.v[i]) / (env.w_v + 1e-12)
    dv_norm = np.clip(dv_norm, -1.0, 1.0)
    dv_idx  = cont_to_idx(dv_norm, env.K_v)

    # --------------------------------------------------
    # 2) WET selection: closest node
    # --------------------------------------------------
    dists = np.linalg.norm(env.nodes - pos, axis=1)
    sel_wet = int(np.argmin(dists))

    # --------------------------------------------------
    # 3) DATA selection: AoI + feasibility
    # --------------------------------------------------
    eps = 1e-12
    score = (
        env.AoI.astype(float)
        - 0.7 * dists
        + 0.4 * np.log(env.E_nodes + eps)
    )

    sel_data = int(np.argmax(score))

    return np.array([dtheta_idx, dv_idx, sel_wet, sel_data], dtype=np.int64)


# ==================================================
# Greedy policy for BOTH AUVs
# ==================================================
def greedy_2auv_policy(env: AUV2DEnv2AUV):
    """
    Returns an 8D action:
    [AUV1_dθ, AUV1_dv, AUV1_w, AUV1_d,
     AUV2_dθ, AUV2_dv, AUV2_w, AUV2_d]
    """

    a0 = greedy_single_auv(env, i=0, other_i=1)
    a1 = greedy_single_auv(env, i=1, other_i=0)

    return np.hstack([a0, a1])


# ==================================================
# Run one greedy episode
# ==================================================
def run_baseline_episode(env, max_steps=200, verbose=True):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    if verbose:
        print("\n" + "="*80)
        print("GREEDY 2-AUV AoI + GOAL BASELINE")
        print("="*80)

    while not done and step < max_steps:
        action = greedy_2auv_policy(env)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            print(f"\nSTEP {step}")
            print("-"*60)
            print(f"AUV1 pos = ({env.pos[0][0]:.2f}, {env.pos[0][1]:.2f}) | v={env.v[0]:.2f}")
            print(f"AUV2 pos = ({env.pos[1][0]:.2f}, {env.pos[1][1]:.2f}) | v={env.v[1]:.2f}")
            print(f"d_auv = {info['d_auv']:.2f} | collision = {info['collision']}")
            print(f"Mean AoI = {np.mean(env.AoI):.2f}")
            print(f"Reward = {reward:.3f}")
           

        step += 1

    if verbose:
        print("\n" + "="*80)
        print(f"EPISODE FINISHED | Total reward = {total_reward:.2f}")
        print("="*80)

    env.plot_trajectory()
    

    return total_reward


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    env = AUV2DEnv2AUV()
    run_baseline_episode(env, max_steps=200, verbose=True)
    
