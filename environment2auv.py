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
#                    ENVIRONMENT (2 AUVs)
# ======================================================

class AUV2DEnv2AUV(gym.Env):
    """
    Two-AUV environment with Gaussian collision-risk penalty.
    Each AUV freezes permanently after reaching the goal.
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
            [1,1],[8,5],[5,9],[3,7],[2,4],
            [9,2],[5,2],[7,8],[8,2],[7,3],
            [7,1],[3,1],[9,7]
        ])
        self.N = len(self.nodes)

        # -------- AUVs --------
        self.M = 2
        self.start_pos = np.array([[0.5,0.5],[0.5,1.5]])
        self.pos = self.start_pos.copy()
        self.theta = np.zeros(self.M)
        self.v = np.ones(self.M)
        self.reached = np.zeros(self.M, dtype=bool)

        # -------- Motion --------
        self.dt = 0.25
        self.w_theta = np.deg2rad(25)
        self.w_v = 0.4
        self.v_max = 4.0

        # -------- Control --------
        self.K_theta = 11
        self.K_v = 7

        # -------- AoI & Fairness --------
        self.AoI = np.ones(self.N)
        self.AoI_max = 50
        self.occ = np.zeros(self.N)
        self.lambda_fair = 2.0

        # -------- Service accumulation --------
        self.K_reset = 3
        self.service_count = np.zeros(self.N)

        # -------- Energy --------
        self.E_init = 2e6
        self.E = self.E_init * np.ones(self.M)
        self.E_nodes = 0.1 * np.ones(self.N)

        # -------- Collision penalty --------
        self.alpha_c = 4.0
        self.sigma_c = 20.0
        self.d_th = 100.0


        # -------- Episode --------
        self.max_steps = 72
        self.step_count = 0
        self.trajectory = [[],[]]

        # -------- Action space --------
        self.action_space = spaces.MultiDiscrete([
            self.K_theta, self.K_v, self.N, self.N,
            self.K_theta, self.K_v, self.N, self.N
        ])

        # -------- Observation --------
        obs_dim = 8 + 8*self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    # ======================================================

    def reset(self):
        self.pos = self.start_pos.copy()
        self.theta[:] = 0.0
        self.v[:] = 1.0
        self.reached[:] = False
        self.E[:] = self.E_init

        self.AoI[:] = 1
        self.occ[:] = 0
        self.service_count[:] = 0
        self.E_nodes[:] = 0.1

        self.step_count = 0
        self.trajectory = [[self.pos[0].copy()], [self.pos[1].copy()]]
        return self._obs()

    # ======================================================

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        action = np.array(action).reshape(2,4)

        for i in range(self.M):

            # ===== FREEZE AFTER GOAL =====
            if self.reached[i]:
                self.v[i] = 0.0
                self.trajectory[i].append(self.pos[i].copy())
                continue

            dtheta_idx, dv_idx, sel_w, sel_d = action[i]
            dtheta = 2*dtheta_idx/(self.K_theta-1) - 1
            dv = 2*dv_idx/(self.K_v-1) - 1

            theta_new = self.theta[i] + dtheta*self.w_theta
            v_new = np.clip(self.v[i] + dv*self.w_v, 0, self.v_max)

            pos_new = self.pos[i] + v_new*self.dt*np.array([
                np.cos(theta_new), np.sin(theta_new)
            ])

            if not (self.xmin <= pos_new[0] <= self.xmax and
                    self.ymin <= pos_new[1] <= self.ymax):
                reward -= 20.0
                pos_new = self.pos[i]
                v_new = self.v[i]
                theta_new = self.theta[i]

            prev = self.pos[i].copy()
            self.pos[i] = pos_new
            self.theta[i] = theta_new
            self.v[i] = v_new
            self.trajectory[i].append(self.pos[i].copy())

            dist = np.linalg.norm(self.pos[i]-prev)
            if dist < 1e-2:
                reward -= 1.0
            self.E[i] -= propulsion_energy(self.v[i], dist)

            reward += np.linalg.norm(prev-self.goal) - np.linalg.norm(self.pos[i]-self.goal)

            # ===== Harvest =====
            r = np.linalg.norm(self.nodes[sel_w]-self.pos[i])
            TL = transmission_loss(70, 100*r)
            SL = 170.8 + 10*np.log10(5.0) + 10*np.log10(0.7) + 10
            RL = received_level(SL, TL, 10)
            self.E_nodes[sel_w] += harvested_power(RL)*25

            # ===== Data =====
            r = np.linalg.norm(self.nodes[sel_d]-self.pos[i])
            TL = transmission_loss(50, 100*r)
            gamma = 2**(12000/1000)-1
            SL_req = required_source_level(10*np.log10(gamma), TL, 40+10*np.log10(1000))
            E_req = electrical_power_from_SL(SL_req)*25

            margin = self.E_nodes[sel_d]-E_req
            reward += 2*np.tanh(margin/(E_req+1e-12))

            if self.E_nodes[sel_d] >= E_req:
                self.E_nodes[sel_d] -= E_req
                self.service_count[sel_d] += 1
                self.occ[sel_d] += 1
                if self.service_count[sel_d] >= self.K_reset:
                    self.AoI[sel_d] = 1
                    self.service_count[sel_d] = 0
            else:
                reward -= 0.02

            if np.linalg.norm(self.pos[i]-self.goal) <= self.goal_radius:
                self.reached[i] = True
                self.v[i] = 0.0

        # ===== Collision penalty =====
        d12 = 100 * np.linalg.norm(self.pos[0] - self.pos[1])   # meters

# --------- thresholded "unsafe distance" ----------
        violation = max(0.0, self.d_th - d12)   # meters inside unsafe zone

        # Smooth risk in [0, 1): 0 when safe, ~1 when deep inside unsafe zone
        risk = 1.0 - np.exp(-(violation**2) / (2.0 * self.sigma_c**2))

        # Penalty in [-alpha_c, 0]
        penalty = -self.alpha_c * risk
        reward += penalty

        #print(f"d12={d12:.2f} m | violation={violation:.2f} m | risk={risk:.3f} | collision_penalty={penalty:.3f}")


        # ===== AoI / fairness =====
        self.AoI = np.minimum(self.AoI+1, self.AoI_max)
        if np.sum(self.occ) > 3:
            J = (np.sum(self.occ)** 2)/(self.N*np.sum(self.occ**2))
            reward -= self.lambda_fair*(1-J)
        reward -= 0.1*np.mean(self.AoI)

        done = (np.all(self.reached) or
                self.step_count >= self.max_steps or
                np.any(self.E <= 0))

        return self._obs(), reward, done, {
            "d_auv": d12,
            "collision": int(d12 < self.d_th),
            "reached": self.reached.copy()
        }

    # ======================================================

    def _obs(self):
        rel1 = self.nodes - self.pos[0]
        rel2 = self.nodes - self.pos[1]
        return np.hstack([
            self.pos[0], [self.theta[0], self.v[0]],
            self.pos[1], [self.theta[1], self.v[1]],
            self.AoI,
            self.E_nodes,
            rel1.flatten(),
            rel2.flatten(),
            np.linalg.norm(rel1,axis=1),
            np.linalg.norm(rel2,axis=1)
        ]).astype(np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    def plot_trajectory(self):
        traj1 = np.array(self.trajectory[0])
        traj2 = np.array(self.trajectory[1])

        plt.figure(figsize=(6, 6))

        if len(traj1) > 0:
            plt.plot(traj1[:, 0], traj1[:, 1], '-b', label='AUV 1')
            plt.scatter(traj1[0, 0], traj1[0, 1], c='b', marker='o')

        if len(traj2) > 0:
            plt.plot(traj2[:, 0], traj2[:, 1], '-g', label='AUV 2')
            plt.scatter(traj2[0, 0], traj2[0, 1], c='g', marker='o')

        plt.scatter(self.nodes[:, 0], self.nodes[:, 1],
                    c='r', marker='s', label='Nodes')

        plt.scatter(self.goal[0], self.goal[1],
                    c='k', marker='*', s=150, label='Goal')

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.grid()
        plt.legend()
        plt.title("2-AUV Greedy Trajectories")
        plt.show()