import numpy as np


def communication_first_policy(env, aoi_threshold=4.0):
    """
    Communication-first baseline for the SCORE-BASED action space:

      action = [dtheta_norm, dv_norm, score_w[0..N-1], score_d[0..N-1]]

    Strategy:
    - If mean AoI is high: target the node with max AoI and set its score highest (for both WET and DATA)
    - Else: target the goal and pick a default node (0) for scores
    """

    pos   = env.pos
    theta = env.theta
    goal  = env.goal
    N     = env.N

    # ===============================
    # Decide target
    # ===============================
    mean_aoi = float(np.mean(env.AoI))

    if mean_aoi > aoi_threshold:
        target_idx = int(np.argmax(env.AoI))
        target = env.nodes[target_idx]
        sel_w = target_idx
        sel_d = target_idx
    else:
        target = goal
        sel_w = 0
        sel_d = 0

    # ===============================
    # Motion control (same as before)
    # ===============================
    desired_heading = np.arctan2(
        target[1] - pos[1],
        target[0] - pos[0]
    )

    dtheta = desired_heading - theta
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
    dtheta_norm = np.clip(dtheta / env.w_theta, -1.0, 1.0)

    desired_v = 1.2
    dv = (desired_v - env.v) / env.w_v
    dv_norm = np.clip(dv, -1.0, 1.0)

    # ===============================
    # Score vectors (argmax will pick)
    # ===============================
    score_w = -np.ones(N, dtype=np.float32)
    score_d = -np.ones(N, dtype=np.float32)

    score_w[sel_w] = 1.0
    score_d[sel_d] = 1.0

    # ===============================
    # Final action vector
    # ===============================
    action = np.concatenate(
        ([dtheta_norm, dv_norm], score_w, score_d),
        axis=0
    ).astype(np.float32)

    return action
