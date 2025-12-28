import gym
from gym import spaces
import numpy as np
import pygame


def thorp_absorption(f_kHz: float) -> float:
    """Thorp absorption coefficient α [dB / m] for frequency f [kHz]."""
    f2 = f_kHz**2
    return (0.11 * f2/(f2 + 1)
          + 44   * f2/(f2 + 4100)
          + 2.75e-4 * f2 + 0.003) * 1e-3

def transmission_loss(f_kHz: float, r_m: float, k: float = 1.5) -> float:
    """One-way TL  =  k·10·log10(r)  +  α·r   [dB]."""
    return 10*k*np.log10(np.maximum(r_m, 1.0)) + thorp_absorption(f_kHz)*r_m

def required_source_level(SNR_dB: float, TL_dB: float, NL_band_dB: float) -> float:
    """Passive-sonar equation (omni Tx/Rx)."""
    return SNR_dB + TL_dB + NL_band_dB

def electrical_power_from_SL(SL_dB: float, eta_tx: float = 0.7, DI_tx: float = 10) -> float:
    """Electrical power that yields source-level SL_dB [W]."""
    return 10**((SL_dB - 170.8 - 10*np.log10(eta_tx) - DI_tx)/10)

def received_level(SL_dB: float, TL_dB: float, DI_rx: float = 10) -> float:
    """Acoustic received level at the hydrophone [dB re 1 µPa]."""
    return SL_dB - TL_dB + DI_rx

def harvested_power(RL_dB: float, *, RVS_dB: float = -150, Rp: float = 125,
                    n_hydro: int = 4, harv_eff: float = 0.7) -> float:
    """Electrical power harvested at the rectifier [W]."""
    p_Pa   = 10**(RL_dB/20)                     # acoustic pressure
    V_ind  = p_Pa * 10**(RVS_dB/20)             # induced voltage
    P_av   = n_hydro * (V_ind**2) / (4*Rp)      # available power
    return harv_eff * P_av
def calculate_energy_consumption(H, eta,Cd,S,rho,V,d):


    # Total energy consumption equation
    E_tot = (((rho * Cd * S * V**3) / (2 * eta)) + H)*(d/V)

    return E_tot


def sample_rician(K_lin, size=()):
    """
    Sample complex Rician fading with linear K-factor K_lin.
    Returns h ~ Rician( K_lin/(K_lin+1) deterministic + 1/(K_lin+1) diffuse ).
    """
    # LOS amplitude component
    nu    = np.sqrt(K_lin/(K_lin+1))
    # Diffuse component std‐dev (per real/imag)
    sigma = np.sqrt(1/(2*(K_lin+1)))
    # real + imag
    x = nu + sigma * np.random.randn(*size)
    y =         sigma * np.random.randn(*size)
    return x + 1j*y


class MultiAUVEnvironment(gym.Env):
    def __init__(self):
        super(MultiAUVEnvironment, self).__init__()
        self.window_size = 500 
        self.render_mode = "human"  
        self.metadata = {"render_fps": 30} 
        '''pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock() 
        '''
        
        # Max battery for AUVs 
        self.auv_max_energy = 2158000

        
        self.auv_positions = [np.array([5, 5, 2]), np.array([8, 8, 2])]
        self.auv_energy    = [self.auv_max_energy, 
                              self.auv_max_energy]
        

        self.charging_stations = [
            np.array([1, 1, 1]),
            np.array([1, 10, 4]),
            np.array([10, 1, 4]),
            np.array([10, 10, 4])
        ]
        self.charging_colors = [(0, 255, 255), (0, 200, 200), (0, 150, 150), (0, 100, 100)]

        self.sensor_node_positions = [
            np.array([1, 1, 1]),
            np.array([8, 8, 2]),
            np.array([9, 1, 3]),
            np.array([1, 6, 4]),
            np.array([8, 5, 3]),
            np.array([6, 2, 2]),
            np.array([4, 4, 3]),
           np.array([3, 7, 2]),
            np.array([5, 8, 4]),
            np.array([6, 9, 3])
        ]   
        '''
       
          
            ,
        ,'''
       
        self.num_devices = len(self.sensor_node_positions)
        self.AoI_all_nodes=[1]*self.num_devices 
        self.occurence=[0]*self.num_devices
        self.energy_stored = [0] * self.num_devices 
        self.energy_harvested=0

        self.max_iterations=100
        self.AoI_max=self.max_iterations/2
        self.reward_per_step=[]
        self.total_steps=0
        self.collision_penalty = 15
        self.num_collision=0
        self.threshhold=3*21571

        self.action_space = spaces.MultiDiscrete([
                    6, self.num_devices, self.num_devices,  # AUV 1
                    6, self.num_devices, self.num_devices   # AUV 2
                ])
        self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2 * 3 + self.num_devices * 2,)
                )        

    def step(self, action):
        total_reward = 0
        self.total_steps += 1

        # Unpack the two AUV actions
        direction1, wet1, data1 = action[:3]
        direction2, wet2, data2 = action[3:]
        done_flags = []

        for i, (direction, sel_wet, sel_data) in enumerate([
            (direction1, wet1, data1),
            (direction2, wet2, data2)
        ]):
            reward = 0
            prev_pos = self.auv_positions[i].copy()

            # —— Move (or penalize invalid move) ——
            possible = self.get_possible_directions(i)
            if direction in possible:
                delta = np.array([
                    1 if direction == 0 else -1 if direction == 1 else 0,
                    1 if direction == 2 else -1 if direction == 3 else 0,
                    1 if direction == 4 else -1 if direction == 5 else 0
                ])
                new_pos = prev_pos + delta
                self.auv_positions[i] = new_pos
            else:
                reward -= 10
                new_pos = prev_pos


            # —— Energy consumption & check for “done” ——
            E_move = calculate_energy_consumption(
                H=40, eta=0.7, Cd=0.006, S=3, rho=1000, V=4, d=100
            )
            self.auv_energy[i] -= E_move

            dists = [np.linalg.norm(self.auv_positions[i] - st)
                     for st in self.charging_stations]
            min_d = min(dists)
            E_to_charge = calculate_energy_consumption(
                H=40, eta=0.7, Cd=0.006, S=3, rho=1000, V=4, d=100 * min_d
            )
            done_i = (self.auv_energy[i] < E_to_charge + self.threshhold)
            done_flags.append(done_i)

            # —— If this AUV is done, teleport it to the nearest *unused* station ——
            if done_i:
                taken = {
                    tuple(self.auv_positions[j])
                    for j, dj in enumerate(done_flags) if dj and j != i
                }
                for idx in np.argsort(dists):
                    station = self.charging_stations[idx]
                    if tuple(station) not in taken:
                        self.auv_positions[i] = station.copy()
                        break

            # —— Harvesting —— 
            h = sample_rician(K_lin=10)
            node_wet_pos = self.sensor_node_positions[sel_wet]
            r_old = np.linalg.norm(node_wet_pos - prev_pos)
            r_new = np.linalg.norm(node_wet_pos - new_pos)
            r_avg = 0.5 * (r_old + r_new)
            #print(f"AUV {i} Harvest: r_old={r_old:.2f}, r_new={r_new:.2f}, r_avg={r_avg:.2f}")

            E_har = self.compute_harvested_energy(100*r_avg, h)
            self.energy_stored[sel_wet] += E_har
            self.energy_harvested    += E_har

            # ── Transmission & DEBUG ──
            node_data_pos = self.sensor_node_positions[sel_data]
            r_old = np.linalg.norm(node_data_pos - prev_pos)
            r_new = np.linalg.norm(node_data_pos - new_pos)
            r_avg = 0.5 * (r_old + r_new)
            #print(f"AUV {i} Transmit: r_old={r_old:.2f}, r_new={r_new:.2f}, r_avg={r_avg:.2f}")

            E_req = self.energy_required_for_trans(100*r_avg, h)

            if self.energy_stored[sel_data] > E_req:
                self.energy_stored[sel_data] -= E_req
                self.occurence[sel_data] += 1
                _ = self.update_Age(sel_data)
                if self.occurence[sel_data] > 27:
                    reward -= 10
            else:
                reward -= 3

            total_reward += reward

        # —— Collision penalty —— 
        if np.all(np.abs(self.auv_positions[0] - self.auv_positions[1]) <= 1):
            total_reward -= self.collision_penalty
            self.num_collision += 1

        # —— AoI & fairness penalty —— 
        AoI = self.update_all_Age()
        sum_sq = sum(x**2 for x in self.occurence)
        sum_val = sum(self.occurence)
        Jain = (sum_val**2) / (sum_sq * self.num_devices) if sum_sq else 0
        total_reward -= (1 - Jain) * (np.sum(AoI) / self.num_devices)
        self.reward_per_step.append(np.sum(AoI) / self.num_devices)
        # —— Episode termination —— 
        self.max_iterations -= 1
        env_done = all(done_flags) or self.max_iterations <= 0

        state = self._get_observation()
        return state, total_reward, env_done, {}


    def reset(self):
        self.auv_positions = [np.array([5, 5, 2]), np.array([8, 8, 2])]
        self.max_iterations = 100
        self.auv_max_energy = 2158000
        self.auv_energy = [self.auv_max_energy, 
                              self.auv_max_energy]
        self.total_steps = 0
        self.AoI_all_nodes = [1] * self.num_devices
        self.energy_stored = [0] * self.num_devices
        self.energy_harvested = 0
        self.occurence = [0] * self.num_devices
        self.num_collision=0
        return np.hstack((self.auv_positions[0], self.auv_positions[1], self.AoI_all_nodes, self.energy_stored))
    
    def _get_observation(self):
        return np.hstack((self.auv_positions[0], self.auv_positions[1], self.AoI_all_nodes, self.energy_stored))

    
    def seed(self, seed=None):
        pass

    def compute_harvested_energy(self, r_m: float, h: complex ) -> float:
        f_kHz        = 70                       # beacon centre-frequency
        TL_dB        = transmission_loss(f_kHz, r_m)
        beacon_Pelec = 5                      # W   (electrical power budget)
        eta_tx       = 0.7
        DI_tx        = 10                       # directivity gain [dB]
        DI_rx        = 10                       # hydrophone array gain [dB]

        # Beacon source level
        SL_beacon = 170.8 + 10*np.log10(beacon_Pelec) \
                            + 10*np.log10(eta_tx) + DI_tx

        RL_dB = received_level(SL_beacon, TL_dB, DI_rx)
        P_har = harvested_power(RL_dB) * (np.abs(h)**2)   # fading gain
        duration=25
        return P_har * duration 
    

    def energy_required_for_trans(self, r_m: float, h: complex ) -> float:
        f_kHz          = 50
        BW_Hz          = 1_000          # channel bandwidth
        data_rate_bps  = 12_000              # payload bits per second
        TL_dB          = transmission_loss(f_kHz, r_m)
        NL_spec_dB     = 40             # spectral noise density [dB re 1µPa/√Hz]
        NL_band_dB     = NL_spec_dB + 10*np.log10(BW_Hz)
 
        gamma_lin = 2**(data_rate_bps / BW_Hz) - 1        # required SNR (linear)
        SNR_req_dB = 10*np.log10(gamma_lin)

        SL_req_dB = required_source_level(SNR_req_dB, TL_dB, NL_band_dB)
        P_tx_W    = electrical_power_from_SL(SL_req_dB) * (np.abs(h)**2)
        duration=25
        return P_tx_W * duration         

    def update_Age(self,node_selected_index):
        self.AoI_all_nodes[node_selected_index]=0
        return self.AoI_all_nodes
    
    def update_all_Age(self):
        for i in range(len(self.AoI_all_nodes)):
                self.AoI_all_nodes[i] = min(self.AoI_max, self.AoI_all_nodes[i] + 1)
        return self.AoI_all_nodes
    
    def get_cumulative_rewards(self):
        return self.cumulative_rewards

    def get_possible_directions(self, auv_index):
        auv_position = self.auv_positions[auv_index]
        possible_mvt = np.ones(6)
        
        if auv_position[0] == 1: possible_mvt[1] = 0  # Left boundary
        if auv_position[0] == 10: possible_mvt[0] = 0  # Right boundary
        if auv_position[1] == 1: possible_mvt[3] = 0  # Bottom boundary
        if auv_position[1] == 10: possible_mvt[2] = 0  # Top boundary
        if auv_position[2] == 1: possible_mvt[5] = 0  # Lowest depth
        if auv_position[2] == 4: possible_mvt[4] = 0  # Maximum depth
        
        return np.where(possible_mvt == 1)[0]

    
    def render(self):
        
       
        self.screen.fill((255, 255, 255))
        
        cell_size = self.window_size // 10
        for i in range(11):
            pygame.draw.line(self.screen, (100, 100, 100), (i * cell_size, 0), (i * cell_size, self.window_size), 1)
            pygame.draw.line(self.screen, (100, 100, 100), (0, i * cell_size), (self.window_size, i * cell_size), 1)
        
        # Draw AUVs
        for i, pos in enumerate(self.auv_positions):
            color = (0, 0, 255) if i == 0 else (0, 255, 0)  # Different colors for each AUV
            pygame.draw.circle(self.screen, color, (int(pos[0] * cell_size), int(pos[1] * cell_size)), 5)
        
        # Draw sensor nodes
        for sensor_node_pos in self.sensor_node_positions:
            node_x = int(sensor_node_pos[0] * cell_size)
            node_y = int(sensor_node_pos[1] * cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(node_x - 5, node_y - 5, 10, 10))


        for i, station in enumerate(self.charging_stations):
            station_x = int(station[0] * cell_size)
            station_y = int(station[1] * cell_size)
            pygame.draw.circle(self.screen, self.charging_colors[i], 
                             (station_x, station_y), 10)
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.time.delay(200)

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _is_at_charging_station(self, position):
        return any(np.array_equal(position, station) for station in self.charging_stations)
