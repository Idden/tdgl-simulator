# |error| vs h on log-log axes for rho_c and rho_e, both codes, with slope-1 and slope-2 guides.
# Reference = finest pyTDGL run (same convention as result.py).
# Reads mine_results.txt and pytdgl_results.txt (columns: h N T rho_c rho_e steady).
# Saves convergence_rect.pdf

import numpy as np
import matplotlib.pyplot as plt

M = np.loadtxt("mine_results.txt")
P = np.loadtxt("pytdgl_results.txt")

# reference: pyTDGL row with the smallest h
k_ref = 0
for k in range(len(P)):
    if P[k, 0] < P[k_ref, 0]:
        k_ref = k
print(f"reference: pyTDGL h = {P[k_ref, 0]}  rho_c = {P[k_ref, 3]:.6f}  rho_e = {P[k_ref, 4]:.6f}")

# steady-state check (last two saved frames)
for k in range(len(M)):
    print(f"Fortran h = {M[k, 0]:<5}  steady = {M[k, 5]:.1e}")
for k in range(len(P)):
    print(f"pyTDGL  h = {P[k, 0]:<5}  steady = {P[k, 5]:.1e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
names = [r"$\rho_c$ (center)", r"$\rho_e$ (min over $|x|\leq 1$)"]

for c in range(2):
    col = 3 + c
    ref = P[k_ref, col]

    hM = M[:, 0]
    eM = np.abs(M[:, col] - ref)

    hP = []
    eP = []
    for k in range(len(P)):
        if k != k_ref:
            hP.append(P[k, 0])
            eP.append(abs(P[k, col] - ref))

    ax[c].loglog(hM, eM, "o-", label="Fortran")
    ax[c].loglog(hP, eP, "s-", label="pyTDGL")

    # guide lines through the smallest-h Fortran point
    h0 = hM[np.argmin(hM)]
    e0 = eM[np.argmin(hM)]
    hg = np.array([min(hM), max(hM)])
    if e0 > 0:
        ax[c].loglog(hg, e0 * (hg / h0)**1, "k:", lw=1, label=r"$\propto h$")
        ax[c].loglog(hg, e0 * (hg / h0)**2, "k--", lw=1, label=r"$\propto h^2$")

    ax[c].set_xlabel(r"h ($\xi$)")
    ax[c].set_ylabel("|error| vs finest pyTDGL")
    ax[c].set_title(names[c])
    ax[c].legend()

plt.tight_layout()
plt.savefig("convergence_rect.pdf")
print("saved convergence_rect.pdf")
