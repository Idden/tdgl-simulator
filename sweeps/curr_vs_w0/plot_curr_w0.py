# Critical current against constriction width. Reads every grid_task*.txt the
# sweep produced, one W value per task, and plots J_c(w0).
#
# The raw curve comes out as a staircase, and that is a real effect rather
# than noise: the blockade mask is |i - ci| > gap_half with gap_half =
# w0*dimy/2, and i is an integer, so the number of open rows only changes when
# gap_half crosses a half-integer. Every w0 inside one of those windows gives
# the identical geometry, so the identical I_c. Each step is collapsed to the
# centre of its window, which is the true effective channel width -- unbiased
# even for the first and last steps, which the sweep range chops off.

import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# Grid parameters -- must match the Fortran run
DIMY = 60          # number of rows in the simulation grid
SHOW_STAIRS = True # draw the raw (staircase) curve underneath

def geom_key(w0, dimy=DIMY):
    """k = number of open rows on ONE side of the channel centre."""
    g = 0.5 * dimy * np.asarray(w0, dtype=float)
    k = np.floor(g + 0.5)
    return np.clip(k, 0, dimy // 2).astype(int)

# Load data. Pass a glob on the command line to point at a different run.
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data/grid_task*.txt"))

W = []
Ic = []

for fname in files:
    data = np.loadtxt(fname)
    data = np.atleast_2d(data)
    if data.size == 0:
        continue

    iI = data[:, 1].astype(int)
    w0 = data[:, 2]
    I = data[:, 3]
    vortex = data[:, 4].astype(int)

    # I_c is the smallest current that let a vortex in
    order = np.argsort(iI)
    I, vortex = I[order], vortex[order]
    hit = np.where(vortex == 1)[0]
    if hit.size:
        W.append(w0[0])
        Ic.append(I[hit.min()])

W = np.array(W)
Ic = np.array(Ic)
order = np.argsort(W)
W, Ic = W[order], Ic[order]

# Collapse each stair step to its midpoint
k = geom_key(W)

w_mid, Ic_mid, n_in_step = [], [], []
for kk in np.unique(k):
    m = k == kk
    step_Ic = Ic[m]
    if np.ptp(step_Ic) > 0:
        # Same geometry should give the same I_c. If it did not, say so
        # loudly and fall back to the median.
        print(f"  note: k={kk} (w={2*kk/DIMY:.4f}) has {len(np.unique(step_Ic))} "
              f"distinct I_c values, spread {np.ptp(step_Ic):.3e} -- using median")
    w_mid.append(2.0 * kk / DIMY)
    Ic_mid.append(np.median(step_Ic))
    n_in_step.append(m.sum())

w_mid = np.array(w_mid)
Ic_mid = np.array(Ic_mid)

print(f"{len(W)} sweep points collapse to {len(w_mid)} distinct geometries")
print(f"{'w0_run_centre':>14} {'n_open_rows':>12} {'n_runs':>7} {'I_c':>12}")
for wm, im, nn in zip(w_mid, Ic_mid, n_in_step):
    print(f"{wm:14.4f} {int(round(wm*DIMY)):12d} {nn:7d} {im:12.6f}")

# I_c curve
plt.figure(figsize=(16, 9.5))

if SHOW_STAIRS:
    plt.step(W, Ic, where="post", color="0.75", lw=1.5,
             label="raw sweep (grid-discretizeted)")

plt.plot(w_mid, Ic_mid, "-o", color="tab:blue", lw=2.5, ms=8,
         label="step midpoints (effective width)")

plt.xlabel("$w_0 \\, / \\, w_{lead}$", fontsize=35)
plt.ylabel("$J_c \\, / \\, J_0$", fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.title("Current vs. Constriction Width", fontsize=45)
plt.xlim(0.2, 0.9)
plt.ylim(0.07, 0.15)
plt.grid(alpha=0.3)
plt.legend(fontsize=30)
plt.tight_layout()
plt.savefig("Jc_vs_width.png", dpi=200)
plt.show()