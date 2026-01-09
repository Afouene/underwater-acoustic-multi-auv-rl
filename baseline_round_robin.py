import numpy as np
from environment1auv import AUV2DEnv


# ==================================================
# Helpers
# ==================================================
def cont_to_idx(u, K):
    """
    Map u in [-1,1] to discrete index {0,...,K-1}
    """
    u = np.clip(u, -1.0, 1.0)
    return int(np.round((u + 1.0) * 0.5 * (K - 1)))


# ==================================================
# Round-Robin Baseline with Dwell
# ==================================================
class RoundRobinBaseline:
    """
    Phase 1: Round-robin service
      - Move to node k
      - Harvest + transmit from SAME node
      - DWELL until AoI[k] == 0
    Phase 2: Go to goal
    """

    def __init__(self, env: AUV2DEnv):
        self.env = env
        self.N = env.N
        self.current_node = 0
        self.served = np.zeros(self.N, dtype=bool)
        self.phase = "service"   # "service" or "goal"

    def reset(self):
        self.current_node = 0
        self.served[:] = False
        self.phase = "service"

    def step(self):
        env = self.env
        pos = env.pos
        theta = env.theta

        # --------------------------------------------------
        # Phase switch
        # --------------------------------------------------
        if self.phase == "service" and np.all(self.served):
            self.phase = "goal"

        # --------------------------------------------------
        # Target selection
        # --------------------------------------------------
        if self.phase == "service":
            k = self.current_node
            target = env.nodes[k]
        else:
            target = env.goal
            k = None

        # --------------------------------------------------
        # Check service completion
        # --------------------------------------------------
        if self.phase == "service":
            if env.AoI[self.current_node] == 0:
                self.served[self.current_node] = True
                self.current_node = (self.current_node + 1) % self.N

        # --------------------------------------------------
        # Motion: steer toward target
        # --------------------------------------------------
        phi = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        dtheta = phi - theta
        dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
        dtheta_norm = np.clip(dtheta / (env.w_theta + 1e-12), -1.0, 1.0)
        dtheta_idx = cont_to_idx(dtheta_norm, env.K_theta)

        # --------------------------------------------------
        # SPEED CONTROL (THIS IS THE KEY FIX)
        # --------------------------------------------------
        if self.phase == "service" and env.AoI[self.current_node] > 0:
            # DWELL near node to harvest + transmit
            dv_idx = cont_to_idx(-1.0, env.K_v)   # stop / very slow
        else:
            # Move when switching nodes or going to goal
            dv_idx = cont_to_idx(0.6, env.K_v)

        # --------------------------------------------------
        # Scheduling
        # --------------------------------------------------
        if self.phase == "service":
            sel_wet  = self.current_node
            sel_data = self.current_node
        else:
            # after service, just transmit freshest AoI
            sel_wet  = int(np.argmax(env.AoI))
            sel_data = sel_wet

        return np.array(
            [dtheta_idx, dv_idx, sel_wet, sel_data],
            dtype=np.int64
        )


# ==================================================
# SIMPLE POLICY FUNCTION (IMPORT LIKE GREEDY)
# ==================================================
_baseline = None


def round_robin_policy(env: AUV2DEnv):
    """
    Usage:
        from baseline_round_robin import round_robin_policy
        action = round_robin_policy(env)
    """
    global _baseline

    # Give enough time for full service
    if env.step_count == 0:
        env.max_steps = max(env.max_steps, 500)

    if _baseline is None or env.step_count == 0:
        _baseline = RoundRobinBaseline(env)
        _baseline.reset()

    return _baseline.step()
