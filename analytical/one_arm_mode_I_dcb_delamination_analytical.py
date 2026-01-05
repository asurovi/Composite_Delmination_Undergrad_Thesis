# Author: Agnila Ghosh Surovi


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# 1. INPUT: geometry, material, fracture toughness
# ---------------------------------------------------------
E1  = 1.227e11   # Pa, longitudinal modulus E_1
E2  = 1.01e10    # Pa, transverse modulus E_2
G12 = 5.5e9      # Pa, in-plane shear modulus G_12

GIc = 969.0      # J/m^2, Mode I fracture toughness

B   = 0.02       # m, specimen width
h   = 0.0015     # m, full-thickness of one arm (total thickness = 2h)

a0    = 0.03     # m, initial crack length
a_max = 0.06519   # m, final crack length
n_pts = 200      # number of points for propagation branch

# ---------------------------------------------------------
# 2. Crack-tip rotation correction (Williams / Pereira)
# ---------------------------------------------------------
Gamma   = 1.18 * np.sqrt(E1 * E2 / G12)
Delta_c = h * np.sqrt((E1 / (11.0 * G12)) * (3.0 - 2.0 * (Gamma / (1.0 + Gamma))**2))

print(f"Crack-tip correction Delta_c = {Delta_c*1e3:.3f} mm")

# ---------------------------------------------------------
# 3. Elastic branch (no crack growth): a = a0
# ---------------------------------------------------------
a0_bar = a0 + Delta_c

C0 = 4.0 * a0_bar**3 / (E1 * B * h**3)

Pc0 = np.sqrt(GIc * E1 * B**2 * h**3 / (12.0 * a0_bar**2))
delta0 = C0 * Pc0

n_elastic = 80
P_elastic = np.linspace(0.0, Pc0, n_elastic)
delta_elastic_m  = C0 * P_elastic
delta_elastic_mm = delta_elastic_m * 1e3

# ---------------------------------------------------------
# 4. Propagation branch (crack-tip corrected)
# ---------------------------------------------------------
a = np.linspace(a0, a_max, n_pts)
a_bar = a + Delta_c

P_prop = np.sqrt(GIc * E1 * B**2 * h**3 / (12.0 * a_bar**2))
delta_prop_m  = 4.0 * P_prop * a_bar**3 / (E1 * B * h**3)
delta_prop_mm = delta_prop_m * 1e3

# ---------------------------------------------------------
# 5. PLOT: load–displacement curve
# ---------------------------------------------------------
plt.figure()
plt.plot(delta_elastic_mm, P_elastic, '--', label="Elastic analytical (no growth)")
plt.plot(delta_prop_mm,   P_prop,    '-',  label="Corrected analytical (crack-tip)")

plt.xlabel("Displacement [mm]")
plt.ylabel("Load [N]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. EXPORT CSV
# ---------------------------------------------------------
branch_elastic = np.array(["elastic"] * n_elastic)
branch_prop    = np.array(["corrected"] * n_pts)

df_elastic = pd.DataFrame({
    "branch": branch_elastic,
    "a_m":    np.full(n_elastic, a0),
    "a_bar_m":np.full(n_elastic, a0_bar),
    "P_N":    P_elastic,
    "delta_m":delta_elastic_m,
    "delta_mm":delta_elastic_mm
})

df_prop = pd.DataFrame({
    "branch": branch_prop,
    "a_m":    a,
    "a_bar_m":a_bar,
    "P_N":    P_prop,
    "delta_m":delta_prop_m,
    "delta_mm":delta_prop_mm
})

df_all = pd.concat([df_elastic, df_prop], ignore_index=True)
df_all.to_csv("analytical_DCB_P_delta_elastic_and_corrected.csv", index=False)

print("Saved analytical data to analytical_DCB_P_delta_elastic_and_corrected.csv")

