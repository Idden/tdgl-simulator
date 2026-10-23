# Reads mine_results.txt and pytdgl_results.txt and compares the two codes.
# Reference = finest pyTDGL run (stands in for the exact continuum answer).
# Columns in both files: h N T rho_c rho_e steady

import numpy as np
import matplotlib.pyplot as plt

M = np.loadtxt("mine_results.txt")
P = np.loadtxt("pytdgl_results.txt")

# reference: pyTDGL row with the smallest h
k_ref = 0
for k in range(len(P)):
    if P[k, 0] < P[k_ref, 0]:
        k_ref = k
rc_ref = P[k_ref, 3]
re_ref = P[k_ref, 4]
print(f"reference: pyTDGL h = {P[k_ref, 0]}  rho_c = {rc_ref:.6f}  rho_e = {re_ref:.6f}\n")

# pyTDGL rows without the reference
P_use = []
for k in range(len(P)):
    if k != k_ref:
        P_use.append(P[k])
P_use = np.array(P_use)

# table
print(f"{'code':<8}{'h':<7}{'N':<9}{'T (s)':<11}{'rho_c':<11}{'err_c':<11}{'rho_e':<11}{'err_e':<11}")
for k in range(len(M)):
    print(f"{'Fortran':<8}{M[k,0]:<7}{int(M[k,1]):<9}{M[k,2]:<11.2f}{M[k,3]:<11.6f}"
          f"{abs(M[k,3]-rc_ref):<11.2e}{M[k,4]:<11.6f}{abs(M[k,4]-re_ref):<11.2e}")
for k in range(len(P_use)):
    print(f"{'pyTDGL':<8}{P_use[k,0]:<7}{int(P_use[k,1]):<9}{P_use[k,2]:<11.2f}{P_use[k,3]:<11.6f}"
          f"{abs(P_use[k,3]-rc_ref):<11.2e}{P_use[k,4]:<11.6f}{abs(P_use[k,4]-re_ref):<11.2e}")

# Figure 1: error vs wall time (top), rho vs N (bottom)
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

for c in range(2):
    col  = 3 + c                      # 3 = rho_c, 4 = rho_e
    ref  = rc_ref if c == 0 else re_ref
    name = r"$\rho_c$" if c == 0 else r"$\rho_e$"

    errM = np.abs(M[:, col] - ref)
    errP = np.abs(P_use[:, col] - ref)

    ax[0, c].loglog(M[:, 2], errM, "o-", label="Fortran")
    ax[0, c].loglog(P_use[:, 2], errP, "s-", label="pyTDGL")
    ax[0, c].set_xlabel("wall time (s)")
    ax[0, c].set_ylabel("|error| in " + name)
    ax[0, c].set_title("error vs time: " + name)
    ax[0, c].legend()

    ax[1, c].semilogx(M[:, 1], M[:, col], "o-", label="Fortran")
    ax[1, c].semilogx(P[:, 1], P[:, col], "s-", label="pyTDGL")
    ax[1, c].axhline(ref, color="k", ls="--", lw=1, label="reference")
    ax[1, c].set_xlabel("N (sites)")
    ax[1, c].set_ylabel(name)
    ax[1, c].set_title(name + " vs N")
    ax[1, c].legend()

plt.tight_layout()
plt.savefig("compare_accuracy.pdf")

# Figure 2: wall time vs N
fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(M[:, 1], M[:, 2], "o-", label="Fortran")
ax.loglog(P[:, 1], P[:, 2], "s-", label="pyTDGL")
ax.set_xlabel("N (sites)")
ax.set_ylabel("wall time (s)")
ax.set_title("cost vs N")
ax.legend()
plt.tight_layout()
plt.savefig("compare_time.pdf")

print("\nsaved compare_accuracy.pdf, compare_time.pdf")