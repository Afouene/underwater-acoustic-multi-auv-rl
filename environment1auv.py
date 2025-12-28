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

    def __init__(self):
        super().__init__()

        # -------- Area --------
        self.xmin, self.xmax = 0.0, 10.0
        self.ymin, self.ymax = 0.0, 10.0

        # -------- Goal --------
        self.goal = np.array([9.5, 9.5])
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

        self.dt = 1.0
        self.w_theta = np.deg2rad(25)
        self.w_v = 0.4
        self.v_max = 4.0

        # -------- AoI & Fairness --------
        self.AoI = np.ones(self.N)
        self.AoI_max = 50
        self.occ = np.zeros(self.N)
        self.lambda_fair = 2.0

        # -------- Energy --------
        self.E_init = 2e6
        self.E = self.E_init
        self.E_nodes = np.zeros(self.N)

        # -------- Episode --------
        self.max_steps = 200
        self.step_count = 0
        self.trajectory = []

        # -------- Action space --------
        self.action_space = spaces.Dict({
            "motion": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "sel_wet": spaces.Discrete(self.N),
            "sel_data": spaces.Discrete(self.N),
        })

        # -------- Observation --------
        obs_dim = 4 + 2*self.N
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
        self.E_nodes[:] = 0
        self.E = self.E_init
        self.step_count = 0
        self.trajectory = [self.pos.copy()]
        return self._obs()

    # ======================================================

    def step(self, action):
        self.step_count += 1
        reward = 0.0

        dtheta, dv = action["motion"]
        sel_w = int(action["sel_wet"])
        sel_d = int(action["sel_data"])

        # -------- Motion --------
        self.theta += dtheta * self.w_theta
        self.v = np.clip(self.v + dv*self.w_v, 0.0, self.v_max)

        prev = self.pos.copy()
        self.pos += self.v * np.array([np.cos(self.theta), np.sin(self.theta)])
        dist = np.linalg.norm(self.pos - prev)
        self.E -= propulsion_energy(self.v, dist)

        if not (self.xmin <= self.pos[0] <= self.xmax and
                self.ymin <= self.pos[1] <= self.ymax):
            return self._obs(), -200.0, True, {}

        self.trajectory.append(self.pos.copy())

        # ==================================================
        #              ACOUSTIC ENERGY HARVESTING
        # ==================================================

        r = np.linalg.norm(self.nodes[sel_w] - self.pos)
        r_m = 100 * r

        f_kHz = 70
        beacon_Pelec = 5.0
        eta_tx = 0.7
        DI_tx = 10
        DI_rx = 10

        TL = transmission_loss(f_kHz, r_m)
        SL = 170.8 + 10*np.log10(beacon_Pelec) + 10*np.log10(eta_tx) + DI_tx
        RL = received_level(SL, TL, DI_rx)

        P_h = harvested_power(RL)
        E_h = P_h * 25
        self.E_nodes[sel_w] += E_h

        # ==================================================
        #              INFORMATION TRANSMISSION
        # ==================================================

        r = np.linalg.norm(self.nodes[sel_d] - self.pos)
        r_m = 100 * r

        f_kHz = 50
        BW = 1000
        rate = 12_000
        NL_spec = 40

        TL = transmission_loss(f_kHz, r_m)
        NL_band = NL_spec + 10*np.log10(BW)

        gamma = 2**(rate / BW) - 1
        SNR_dB = 10*np.log10(gamma)

        SL_req = required_source_level(SNR_dB, TL, NL_band)
        P_tx = electrical_power_from_SL(SL_req)
        E_req = P_tx * 25

        if self.E_nodes[sel_d] >= E_req:
            self.E_nodes[sel_d] -= E_req
            self.AoI[sel_d] = 0
            self.occ[sel_d] += 1
        else:
            reward -= 2.0

        # -------- AoI Update --------
        self.AoI = np.minimum(self.AoI + 1, self.AoI_max)

        # -------- Fairness --------
        sum_occ = np.sum(self.occ)
        sum_sq = np.sum(self.occ**2)
        Jain = (sum_occ**2) / (self.N * sum_sq) if sum_sq > 0 else 0.0
        reward -= self.lambda_fair * (1 - Jain)

        # -------- Other rewards --------
        reward -= 0.05 * (abs(dtheta) + abs(dv))
        reward -= np.mean(self.AoI)
        reward -= 0.1 * np.linalg.norm(self.pos - self.goal)

        # -------- Goal --------
        if np.linalg.norm(self.pos - self.goal) <= self.goal_radius:
            reward += 300.0
            return self._obs(), reward, True, {}

        done = self.step_count >= self.max_steps or self.E <= 0
        return self._obs(), reward, done, {}

    # ======================================================

    def _obs(self):
        return np.hstack([
            self.pos,
            self.theta,
            self.v,
            self.AoI,
            self.E_nodes
        ]).astype(np.float32)

    # ======================================================

    def plot_trajectory(self):
        traj = np.array(self.trajectory)
        plt.figure(figsize=(6,6))
        plt.plot(traj[:,0], traj[:,1], '-b', label='AUV')
        plt.scatter(self.nodes[:,0], self.nodes[:,1], c='r', marker='s')
        plt.scatter(*self.start_pos, c='g', label='Start')
        plt.scatter(*self.goal, c='k', marker='*', s=150, label='Goal')
        plt.grid()
        plt.legend()
        plt.show()
