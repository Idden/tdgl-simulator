import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# load data
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data_fine/grid_task*.txt"))

W = []
Ic = []

for fname in files:
    data = np.loadtxt(fname)
    data = np.atleast_2d(data)

    iI = data[:, 1].astype(int)
    w0 = data[:, 2]
    I = data[:, 3]
    vortex = data[:, 4].astype(int)

    # I_c: smallest I with vortex==1
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

# I_c curve
plt.figure(figsize=(16, 9.5))
plt.plot(W, Ic)
plt.xlabel("$w_0 \\, / \\, w_{lead}$", fontsize=35)
plt.ylabel("$J_c \\, / \\, J_0$", fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.title("Current vs. Constriction Width", fontsize=45)
plt.xlim(0.2, 0.9)
plt.ylim(0.07, 0.15)
plt.grid(alpha=0.3)
plt.savefig("Jc_vs_width.pdf", dpi=200)
plt.show()