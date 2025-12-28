import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# ====================== Acoustic & Energy Helpers ======================

def thorp_absorption(f_kHz):
    f2 = f_kHz**2
    return (0.11*f2/(f2+1) + 44*f2/(f2+4100) + 2.75e-4*f2 + 0.003) * 1e-3

def transmission_loss(f_kHz, r, k=1.5):
    return 10*k*np.log10(max(r,1.0)) + thorp_absorption(f_kHz)*r

def electrical_power_from_SL(SL_dB):
    return 10**((SL_dB - 170.8)/10)

def propulsion_energy(V, d):
    rho, Cd, S, eta, H = 1000, 0.006, 3, 0.7, 40
    return (((rho*Cd*S*V**3)/(2*eta)) + H) * (d/max(V,1e-6))


# ====================== Environment ======================

class AUV2DEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # -------- Area --------
        self.xmin, self.xmax = 0.0, 10.0
        self.ymin, self.ymax = 0.0, 10.0

        # -------- Goal --------
        self.goal = np.array([9.5, 9.5])
        self.goal_radius = 0.3

        # -------- Sensor Nodes (2D) --------
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
        self.w_theta = np.deg2rad(25)   # max heading change per step
        self.w_v = 0.4                  # max speed change per step
        self.v_max = 4.0

        # -------- AoI & Fairness --------
        self.AoI = np.ones(self.N)
        self.AoI_max = 50
        self.occ = np.zeros(self.N)     # successful updates per node
        self.lambda_fair = 2.0           # fairness weight (tune 0.5–5)

        # -------- Energy --------
        self.E_init = 2e6
        self.E = self.E_init
        self.E_nodes = np.zeros(self.N)

        # -------- Episode --------
        self.max_steps = 200
        self.step_count = 0

        # -------- Logs --------
        self.trajectory = []

        # -------- Action space (DICT) --------
        self.action_space = spaces.Dict({
            "motion": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),  # [Δθ~, Δv~]
            "sel_wet": spaces.Discrete(self.N),
            "sel_data": spaces.Discrete(self.N),
        })

        # -------- Observation --------
        # [x, y, theta, v, AoI(N), E_nodes(N)]
        obs_dim = 4 + self.N + self.N
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

        # ---- Decode Dict action ----
        dtheta, dv = action["motion"]
        sel_w = int(action["sel_wet"])
        sel_d = int(action["sel_data"])

        # ---- Motion (kinematic model) ----
        self.theta += dtheta * self.w_theta
        self.v = np.clip(self.v + dv*self.w_v, 0.0, self.v_max)

        prev = self.pos.copy()
        self.pos[0] += self.v * np.cos(self.theta) * self.dt
        self.pos[1] += self.v * np.sin(self.theta) * self.dt

        dist = np.linalg.norm(self.pos - prev)
        self.E -= propulsion_energy(self.v, dist)

        # ---- Out of bounds → terminate ----
        if not (self.xmin <= self.pos[0] <= self.xmax and
                self.ymin <= self.pos[1] <= self.ymax):
            return self._obs(), -200.0, True, {}

        self.trajectory.append(self.pos.copy())

        # ---- Harvest energy (deterministic) ----
        r = np.linalg.norm(self.nodes[sel_w] - self.pos)
        TL = transmission_loss(70, 100*r)
        P_h = electrical_power_from_SL(170 - TL)
        self.E_nodes[sel_w] += P_h * 25

        # ---- Data transmission ----
        r = np.linalg.norm(self.nodes[sel_d] - self.pos)
        TL = transmission_loss(50, 100*r)
        E_req = electrical_power_from_SL(180 - TL) * 25

        if self.E_nodes[sel_d] > E_req:
            self.E_nodes[sel_d] -= E_req
            self.AoI[sel_d] = 0
        else:
            reward -= 2.0

        self.AoI = np.minimum(self.AoI + 1, self.AoI_max)

        # ---- Fairness (Jain index) ----
        sum_occ = np.sum(self.occ)
        sum_sq  = np.sum(self.occ ** 2)

        if sum_sq > 0:
            Jain = (sum_occ ** 2) / (self.N * sum_sq)
        else:
            Jain = 0.0

        reward -= self.lambda_fair * (1.0 - Jain)

        # ---- Reward components ----
        reward -= 0.05 * (abs(dtheta) + abs(dv))       # smoothness
        reward -= np.mean(self.AoI)                    # freshness
        reward -= 0.1 * np.linalg.norm(self.pos - self.goal)
        print(f"Harvest: node {sel_w}, r={r:.2f}, P_h={P_h:.3e}, E_nodes[{sel_w}]={self.E_nodes[sel_w]:.3e}")
        print(f"Tx: node {sel_d}, r={r:.2f}, E_req={E_req:.3e}, E_nodes[{sel_d}]={self.E_nodes[sel_d]:.3e}")

        # ---- Goal reached ----
        if np.linalg.norm(self.pos - self.goal) <= self.goal_radius:
            reward += 300.0
            return self._obs(), reward, True, {}

        done = self.step_count >= self.max_steps or self.E <= 0.0
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
        plt.figure(figsize=(6, 6))
        plt.plot(traj[:,0], traj[:,1], '-b', label='AUV trajectory')
        plt.scatter(self.nodes[:,0], self.nodes[:,1],
                    c='r', marker='s', label='Nodes')
        plt.scatter(*self.start_pos, c='g', label='Start')
        plt.scatter(*self.goal, c='k', marker='*', s=150, label='Goal')
        plt.legend()
        plt.grid()
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('AUV 2D Trajectory (Fairness-Aware)')
        plt.show()
