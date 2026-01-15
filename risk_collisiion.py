import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters (MUST match env)
# -------------------------------
sigma_c = 300.0      # same as env.sigma_c
alpha_c = 4.0      # same as env.alpha_c (only affects penalty, not risk)

# distance range (same units as d_auv)
d = np.linspace(0, 6, 300)
d=100*d
# Gaussian collision risk (NOT a true probability)
risk = np.exp(-(d**2) / (2 * sigma_c**2))

# -------------------------------
# Plot
# -------------------------------
plt.figure()
plt.plot(d, risk)
plt.xlabel("Inter-AUV distance d")
plt.ylabel("Collision risk (Gaussian kernel)")
plt.grid()
plt.title("Gaussian Distance-Dependent Collision Risk")
plt.show()
