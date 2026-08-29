# Current-field phase diagram: the critical current J_c against applied field.
# Below a curve no vortex enters, above it one does.
#
# One curve per table in `files`, and each table is a SEPARATE sweep run --
# rerun sweep_curr_hp with the parameter changed, save the result under its
# own name, then add it here. The dict key is the legend label.

import numpy as np
import matplotlib.pyplot as plt

# Read one sweep table into (H values, I grid, vortex flag grid).
# Columns are: iH, iI, H_ext, I_curr, vortex, t_c.
def load_grid(fname):
    data = np.loadtxt(fname)
    iH = data[:, 0].astype(int)
    iI = data[:, 1].astype(int)
    H = data[:, 2]
    I = data[:, 3]
    vortex = data[:, 4].astype(int)

    nH, nI = iH.max(), iI.max()
    H_vals = np.full(nH, np.nan)
    I_grid = np.full((nH, nI), np.nan)
    v_grid = np.full((nH, nI), np.nan)

    H_vals[iH - 1] = H
    I_grid[iH - 1, iI - 1] = I
    v_grid[iH - 1, iI - 1] = vortex
    return H_vals, I_grid, v_grid

# J_c at each field is the smallest current that let a vortex in. Rows where
# no current triggered one stay NaN and simply are not drawn.
def critical_current(I_grid, v_grid):
    nH = I_grid.shape[0]
    Ic = np.full(nH, np.nan)
    for h in range(nH):
        hit = np.where(v_grid[h] == 1)[0]
        if hit.size:
            Ic[h] = I_grid[h, hit.min()]
    return Ic

files = {
    "1.0": "data/grid10.txt",
    "1.5": "data/grid15.txt",
    "2.0": "data/grid20.txt",
}

plt.figure(figsize=(16, 9.5))

# Guide lines for the empirical I_c(H) window the sweep searches within.
# Same constants the Fortran uses to narrow its current range.
# xtest = np.linspace(0, 0.27, 100)
# plt.plot(xtest, -0.37037 * xtest + 0.122, '--', color='0.6')
# plt.plot(xtest, -0.37037 * xtest + 0.145, '--', color='0.6')
# plt.plot(np.linspace(0.27, 0.4, 100), np.full(100, 0.035), '--', color='0.6')

for label, fname in files.items():
    H_vals, I_grid, v_grid = load_grid(fname)
    Ic = critical_current(I_grid, v_grid)
    ok = ~np.isnan(Ic)
    plt.plot(H_vals[ok], Ic[ok], lw=2, label=label)

plt.legend(fontsize=35)
# Label the two regions of the phase diagram. Both bubbles are placed by hand.
plt.annotate(
    "No Vortices",
    xy=(0.08, 0.05),    # point being labelled
    xytext=(0.1, 0.04), # where the bubble sits
    ha='center', va='center',
    fontsize=45,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        lw=1.5
    )
)
plt.annotate(
    "Vortices",
    xy=(0.08, 0.05),
    xytext=(0.2, 0.12),
    ha='center', va='center',
    fontsize=45,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        lw=1.5
    )
)
plt.xlabel("$H_{ext} \\, / \\, H_{c_2}$", fontsize=35)
plt.ylabel("$J_c \\, / \\, J_0$", fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.title("Current-Field Phase Diagram", fontsize=45)
plt.xlim(0, 0.4)
plt.ylim(-0.001, 0.15)
plt.grid(alpha=0.3)
plt.savefig("Jc_vs_H.png", dpi=200)
plt.show()