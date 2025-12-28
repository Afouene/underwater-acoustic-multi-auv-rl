import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────
#  ↓↓↓  Link-budget helpers  ↓↓↓
# ───────────────────────────────────────────────────────────────
def thorp_absorption(f_kHz):
    """Thorp absorption coefficient [dB/m] at frequency f [kHz]."""
    f2 = f_kHz**2
    return (0.11 * f2/(f2 + 1)
          + 44   * f2/(f2 + 4100)
          + 2.75e-4 * f2 + 0.003) * 1e-3

def transmission_loss(f_kHz, r_m, k=1.5):
    """One-way TL = k·10·log10(r) + α·r  [dB]."""
    return 10*k*np.log10(np.maximum(r_m, 1.0)) + thorp_absorption(f_kHz)*r_m



def required_source_level(SNR_dB, TL_dB, NL_band_dB):
    """Passive sonar equation (omni tx/rx)."""
    return SNR_dB + TL_dB + NL_band_dB

def electrical_power_from_SL(SL_dB, eta=0.7, DI=20):
    """Convert source level to electrical power [W]."""
    return 10**((SL_dB - 170.8 - 10*np.log10(eta) - DI)/10)

def received_level(SL_dB, TL_dB, DI_rx=20):
    return SL_dB - TL_dB + DI_rx

def harvested_power(RL_dB, RVS_dB=-150, Rp=125,
                    n_hydro=4, harv_eff=0.7):
    """Electrical power harvested [W] at the rectifier."""
    p_Pa = 10**(RL_dB/20)
    V_ind = p_Pa * 10**(RVS_dB/20)
    P_av = n_hydro * (V_ind**2) / (4*Rp)
    return harv_eff * P_av

def link_budget(range_m,
                f_kHz=50,
                BW_Hz=1_000,
                data_rate_bps=10*3_000,
                NL_spec_dB=40,
                eta_tx=0.7,
                DI_tx=20,
                DI_rx=20,
                beacon_Pelec_W=300,
                **harv_kwargs):
    """Return (P_tx_W, P_har_W)."""
    TL_dB = transmission_loss(f_kHz, range_m)
    NL_band_dB = NL_spec_dB + 10*np.log10(BW_Hz)
    print(f"TL: {TL_dB:.2f} dB, NL_band: {NL_band_dB:.2f} dB")
    gamma_lin = 2**(data_rate_bps / BW_Hz) - 1          #   γ = 2^{S/B} − 1
    SNR_dB    = 10*np.log10(gamma_lin)
    SL_req = required_source_level(SNR_dB, TL_dB, NL_band_dB)
    P_tx_W = electrical_power_from_SL(SL_req, eta_tx, DI_tx)

    SL_beacon = 170.8 + 10*np.log10(beacon_Pelec_W) \
                + 10*np.log10(eta_tx) + DI_tx
    RL_dB = received_level(SL_beacon, TL_dB, DI_rx)
    P_har_W = harvested_power(RL_dB, **harv_kwargs)
    return P_tx_W, P_har_W

# ───────────────────────────────────────────────────────────────
#  ↓↓↓  Plot harvested vs. required power vs. range  ↓↓↓
# ───────────────────────────────────────────────────────────────
# ----- user-editable parameters -----
rng_min, rng_max, pts = 100, 5000, 200      # range sweep [m]
databits_per_sec      = 10*3_000               # node→AUV throughput
beacon_power_W        = 300               # AUV beacon electrical power
DI_tx_node            = 20                  # node Tx array gain [dB]
DI_rx_auv             = 20                  # AUV Rx array gain [dB]
# ------------------------------------

distances = np.linspace(rng_min, rng_max, pts)
P_tx, P_har = [], []

for d in distances:
    tx, har = link_budget(d,
                          data_rate_bps=databits_per_sec,
                          beacon_Pelec_W=beacon_power_W,
                          DI_tx=DI_tx_node,
                          DI_rx=DI_rx_auv)
    P_tx.append(tx)
    P_har.append(har)

P_tx = np.array(P_tx)
P_har = np.array(P_har)

# crossing point where harvested ≥ required
cross_idx = np.where(P_har <= P_tx)[0]
cross_d = distances[cross_idx[0]] if cross_idx.size else None

# ---------- plotting ----------
plt.figure(figsize=(8,5))
plt.plot(distances, P_tx, label='Required TX power (node → AUV)')
plt.plot(distances, P_har, label='Harvested power (from beacon)')
if cross_d is not None:
    plt.axvline(cross_d, ls='--', label=f'P_har = P_tx ≈ {cross_d:.0f} m')
plt.yscale('log')
plt.grid(True, which='both', ls=':')
plt.xlabel('Range  [m]')
plt.ylabel('Power  [W]  (log scale)')
plt.title('Harvested vs Required Transmit Power vs Range')
plt.legend()
plt.tight_layout()
plt.show()
