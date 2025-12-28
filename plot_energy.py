import numpy as np
import matplotlib.pyplot as plt

# ======================================================
#        FUNCTIONS (COPIED EXACTLY FROM ENV)
# ======================================================

def thorp_absorption(f_kHz: float) -> float:
    f2 = f_kHz**2
    return (0.11 * f2/(f2 + 1)
          + 44   * f2/(f2 + 4100)
          + 2.75e-4 * f2 + 0.003) * 1e-3

def transmission_loss(f_kHz: float, r_m: float, k: float = 1.5) -> float:
    return 10*k*np.log10(np.maximum(r_m, 1.0)) + thorp_absorption(f_kHz)*r_m

def required_source_level(SNR_dB: float, TL_dB: float, NL_band_dB: float) -> float:
    return SNR_dB + TL_dB + NL_band_dB

def electrical_power_from_SL(SL_dB: float, eta_tx: float = 0.7, DI_tx: float = 10) -> float:
    return 10**((SL_dB - 170.8 - 10*np.log10(eta_tx) - DI_tx)/10)

def received_level(SL_dB: float, TL_dB: float, DI_rx: float = 10) -> float:
    return SL_dB - TL_dB + DI_rx

def harvested_power(RL_dB: float, *, RVS_dB: float = -150, Rp: float = 125,
                    n_hydro: int = 4, harv_eff: float = 0.7) -> float:
    p_Pa  = 10**(RL_dB/20)
    V_ind = p_Pa * 10**(RVS_dB/20)
    P_av  = n_hydro * (V_ind**2) / (4*Rp)
    return harv_eff * P_av

# ======================================================
#        PARAMETERS (MATCH ENV)
# ======================================================

# Harvesting
f_wet = 70
beacon_Pelec = 5.0     # W
eta_tx = 0.7
DI_tx = 10
DI_rx = 10
duration = 25          # s

# Transmission
f_data = 50
BW = 1_000             # Hz
rate = 12_000          # bps
NL_spec = 40           # dB re 1µPa/√Hz

# Fading (mean case)
h2 = 1.0               # |h|^2

# ======================================================
#        ENERGY COMPUTATION
# ======================================================

def harvested_energy(r_m):
    TL = transmission_loss(f_wet, r_m)
    SL = 170.8 + 10*np.log10(beacon_Pelec) + 10*np.log10(eta_tx) + DI_tx
    RL = received_level(SL, TL, DI_rx)
    P_h = harvested_power(RL) * h2
    return P_h * duration

def required_energy(r_m):
    TL = transmission_loss(f_data, r_m)
    NL_band = NL_spec + 10*np.log10(BW)

    gamma = 2**(rate / BW) - 1
    SNR_dB = 10*np.log10(gamma)

    SL_req = required_source_level(SNR_dB, TL, NL_band)
    P_tx = electrical_power_from_SL(SL_req) * h2
    return P_tx * duration

# ======================================================
#        DISTANCE SWEEP (MATCH SIM)
# ======================================================

# Your env uses r_avg * 100
r_sim = np.linspace(1, 8, 200)        # grid units
r_m   = 100 * r_sim                   # meters

E_h = np.array([harvested_energy(r) for r in r_m])
E_r = np.array([required_energy(r) for r in r_m])

# ======================================================
#        PLOTS
# ======================================================

plt.figure(figsize=(7,5))
plt.semilogy(r_sim, E_h, label=r"$E_{\mathrm{harv}}$")
plt.semilogy(r_sim, E_r, label=r"$E_{\mathrm{req}}$")
plt.xlabel("Distance r (grid units)")
plt.ylabel("Energy per step (J)")
plt.title("Energy Harvesting vs Transmission (2-AUV Model)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.semilogy(r_sim, E_h / E_r)
plt.axhline(1.0, color="r", linestyle="--")
plt.xlabel("Distance r (grid units)")
plt.ylabel(r"$E_{\mathrm{harv}} / E_{\mathrm{req}}$")
plt.title("Energy Feasibility Ratio")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
