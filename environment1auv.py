import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
#                ACOUSTIC & ENERGY HELPERS
# ======================================================

def thorp_absorption(f_kHz):
    f2 = f_kHz**2
    return (0.11*f2/(f2+1) + 44*f2/(f2+4100) + 2.75e-4*f2 + 0.003) * 1e-3

def transmission_loss(f_kHz, r_m, k=1.5):
    return 10*k*np.log10(max(r_m, 1.0)) + thorp_absorption(f_kHz)*r_m

def received_level(SL_dB, TL_dB, DI_rx=10):
    return SL_dB - TL_dB + DI_rx

def harvested_power(RL_dB, RVS_dB=-150, Rp=125, n_hydro=4, harv_eff=0.7):
    p_Pa  = 10**(RL_dB / 20)
    V_ind = p_Pa * 10**(RVS_dB / 20)
    P_av  = n_hydro * (V_ind**2) / (4 * Rp)
    return harv_eff * P_av

def required_source_level(SNR_dB, TL_dB, NL_band_dB):
    return SNR_dB + TL_dB + NL_band_dB

def electrical_power_from_SL(SL_dB, eta_tx=0.7, DI_tx=10):
    return 10**((SL_dB - 170.8 - 10*np.log10(eta_tx) - DI_tx) / 10)

def propulsion_energy(V, d):
    rho, Cd, S, eta, H = 1000, 0.006, 3, 0.7, 40
    return (((rho * Cd * S * V**3) / (2 * eta)) + H) * (d / max(V, 1e-6))


# ======================================================
#                    ENVIRONMENT
# ======================================================

class AUV2DEnv(gym.Env):
    """
    MultiDiscrete action space:
      action = (dtheta_idx, dv_idx, sel_w, sel_d)
    """

    def __init__(self):
        super().__init__()

        # -------- Area --------
        self.xmin, self.xmax = 0.0, 10.0
        self.ymin, self.ymax = 0.0, 10.0

        # -------- Goal --------
        self.goal = np.array([8.3, 8.3])
        self.goal_radius = 0.3

        # -------- Sensor Nodes --------
        self.nodes = np.array([
            [1, 1],
            [8, 5],
            [5, 9],
            [3, 7],
            [5, 2],
            [9, 2],
            [2, 4],
        ])
        self.N = len(self.nodes)

        # -------- AUV State --------
        self.start_pos = np.array([0.5, 0.5])
        self.pos = self.start_pos.copy()
        self.theta = 0.0
        self.v = 1.0

        self.dt = 0.25
        self.w_theta = np.deg2rad(25)
        self.w_v = 0.4
        self.v_max = 4.0

        # -------- Control discretization --------
        self.K_theta = 11
        self.K_v = 7

        # -------- AoI & Fairness --------
        self.AoI = np.ones(self.N)
        self.AoI_max = 50
        self.occ = np.zeros(self.N)
        self.lambda_fair = 2.0

        # -------- NEW: service accumulation --------
        self.K_reset = 3                          # REQUIRED number of services
        self.service_count = np.zeros(self.N)    # per-node counter

        # -------- Energy --------
        self.E_init = 2e6
        self.E = self.E_init
        self.E_nodes = 0.1 * np.ones(self.N)

        # -------- Episode --------
        self.max_steps = 80
        self.step_count = 0
        self.trajectory = []

        # -------- Action space --------
        self.action_space = spaces.MultiDiscrete([
            self.K_theta,
            self.K_v,
            self.N,
            self.N
        ])

        # -------- Observation --------
        obs_dim = 4 + 2*self.N + 2*self.N + self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ======================================================

    def reset(self):
        self.pos = self.start_pos.copy()
        self.theta = 0.0
        self.v = 1.0

        self.AoI[:] = 1
        self.occ[:] = 0
        self.service_count[:] = 0
        self.E_nodes[:] = 0.1
        self.E = self.E_init

        self.step_count = 0
        self.trajectory = [self.pos.copy()]
        return self._obs()

    # ======================================================

    def step(self, action):
        self.step_count += 1
        reward = 0.0

        # -------- Decode action --------
        dtheta_idx, dv_idx, sel_w, sel_d = action
        dtheta = 2.0 * dtheta_idx / (self.K_theta - 1) - 1.0
        dv     = 2.0 * dv_idx     / (self.K_v     - 1) - 1.0

        # -------- Motion --------
        theta_new = self.theta + dtheta * self.w_theta
        v_new = np.clip(self.v + dv * self.w_v, 0.0, self.v_max)

        pos_new = self.pos + v_new * self.dt * np.array([
            np.cos(theta_new),
            np.sin(theta_new)
        ])

        if not (self.xmin <= pos_new[0] <= self.xmax and
                self.ymin <= pos_new[1] <= self.ymax):
            reward -= 20.0
            pos_new = self.pos.copy()
            v_new = self.v
            theta_new = self.theta

        prev = self.pos.copy()
        self.pos = pos_new
        self.theta = theta_new
        self.v = v_new

        self.trajectory.append(self.pos.copy())

        # -------- Energy cost --------
        dist = np.linalg.norm(self.pos - prev)
        if dist < 1e-2:
            reward -= 1.0
        self.E -= propulsion_energy(self.v, dist)

        # -------- Goal shaping --------
        reward += np.linalg.norm(prev - self.goal) - np.linalg.norm(self.pos - self.goal)

        # ==================================================
        #              ENERGY HARVESTING
        # ==================================================
        r = np.linalg.norm(self.nodes[sel_w] - self.pos)
        r_m = 100 * r

        TL = transmission_loss(70, r_m)
        SL = 170.8 + 10*np.log10(5.0) + 10*np.log10(0.7) + 10
        RL = received_level(SL, TL, 10)

        self.E_nodes[sel_w] += harvested_power(RL) * 25

        # ==================================================
        #              DATA TRANSMISSION
        # ==================================================
        r = np.linalg.norm(self.nodes[sel_d] - self.pos)
        r_m = 100 * r

        TL = transmission_loss(50, r_m)
        NL_band = 40 + 10*np.log10(1000)
        gamma = 2**(12000 / 1000) - 1
        SL_req = required_source_level(10*np.log10(gamma), TL, NL_band)
        E_req = electrical_power_from_SL(SL_req) * 25

        margin = self.E_nodes[sel_d] - E_req
        reward += 2.0 * np.tanh(margin / (E_req + 1e-12))

        if self.E_nodes[sel_d] >= E_req:
            self.E_nodes[sel_d] -= E_req
            self.service_count[sel_d] += 1
            self.occ[sel_d] += 1

            # -------- AoI reset ONLY after K services --------
            if self.service_count[sel_d] >= self.K_reset:
                self.AoI[sel_d] = 1
                self.service_count[sel_d] = 0
        else:
            reward -= 0.02

        # -------- AoI aging --------
        self.AoI = np.minimum(self.AoI + 1, self.AoI_max)

        # -------- Fairness --------
        if np.sum(self.occ) > 3:
            s1 = np.sum(self.occ)
            s2 = np.sum(self.occ**2)
            Jain = (s1**2) / (self.N * s2)
            reward -= self.lambda_fair * (1 - Jain)

        # -------- AoI penalty --------
        reward -= 0.1 * np.mean(self.AoI)

        # -------- Goal --------
        if np.linalg.norm(self.pos - self.goal) <= self.goal_radius:
            reward += 200.0
            done = True
        else:
            done = (self.step_count >= self.max_steps) or (self.E <= 0)

        return self._obs(), reward, done, {}

    # ======================================================

    def _obs(self):
        rel_pos = self.nodes - self.pos
        dists = np.linalg.norm(rel_pos, axis=1)
        return np.hstack([
            self.pos,
            [self.theta, self.v],
            self.AoI,
            self.E_nodes,
            rel_pos.flatten(),
            dists
        ]).astype(np.float32)
    def seed(self, seed=None): 
        np.random.seed(seed) 
        return [seed]
    # ======================================================

    def plot_trajectory(self):
        traj = np.array(self.trajectory)
        plt.figure(figsize=(6,6))
        plt.plot(traj[:,0], traj[:,1], '-b', label='AUV')
        plt.scatter(self.nodes[:,0], self.nodes[:,1], c='r', marker='s', label='Nodes')
        plt.scatter(*self.start_pos, c='g', label='Start')
        plt.scatter(*self.goal, c='k', marker='*', s=150, label='Goal')
        plt.grid()
        plt.legend()
        plt.show()
