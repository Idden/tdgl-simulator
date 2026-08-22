import numpy as np
import matplotlib.pyplot as plt

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

# guide lines
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
plt.annotate(
    "No Vortices",               # Text inside the bubble
    xy=(0.08, 0.05),        # Coordinate of the dot
    xytext=(0.1, 0.04),      # Coordinate of the bubble
    ha='center', va='center',
    fontsize=45,
    
    # 1. This creates the "Text Bubble"
    bbox=dict(
        boxstyle="round,pad=0.5",   # Rounded corners with padding
        facecolor="white",          # Bubble background color
        edgecolor="black",          # Bubble border color
        lw=1.5
    ),
    
    # 2. This creates the line and the "Dot"
    # arrowprops=dict(
    #     arrowstyle="-|>",           # Line style (replaces standard arrow)
    #     connectionstyle="arc3",     # Straight line connection
    #     color="black",              # Line and dot color
    #     lw=1.5,                     # Line width
    # )
)
plt.annotate(
    "Vortices",               # Text inside the bubble
    xy=(0.08, 0.05),        # Coordinate of the dot
    xytext=(0.2, 0.12),      # Coordinate of the bubble
    ha='center', va='center',
    fontsize=45,
    
    # 1. This creates the "Text Bubble"
    bbox=dict(
        boxstyle="round,pad=0.5",   # Rounded corners with padding
        facecolor="white",          # Bubble background color
        edgecolor="black",          # Bubble border color
        lw=1.5
    ),
    
    # 2. This creates the line and the "Dot"
    # arrowprops=dict(
    #     arrowstyle="-|>",           # Line style (replaces standard arrow)
    #     connectionstyle="arc3",     # Straight line connection
    #     color="black",              # Line and dot color
    #     lw=1.5,                     # Line width
    # )
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