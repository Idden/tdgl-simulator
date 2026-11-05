# Final |psi|^2 for every h: Fortran (top row) and pyTDGL (bottom row), same color scale.
# Reads data_h=<h>/psi.bin and pytdgl_h=<h>.npz. Saves fields_rect.pdf

import numpy as np
import matplotlib.pyplot as plt
from rect_geom import L, outline_xy

hs = [2.0, 1.0, 0.5, 0.25]

# load everything first so all panels share one color scale
F = []      # Fortran rho[ix, iy], NaN on blockade
P = []      # pyTDGL npz
lo = 1e9
hi = 0.0
for h in hs:
    n = int(L / h) + 1
    psi = np.fromfile(f"data_h={h}/psi.bin", dtype=np.complex128).reshape(-1, n, n)
    rho = np.abs(psi[-1])**2
    rho[rho == 0.0] = np.nan              # blockade sites
    F.append(rho)

    d = np.load(f"pytdgl_h={h}.npz")
    P.append(d)

    lo = min(lo, np.nanmin(rho), d["rho"].min())
    hi = max(hi, np.nanmax(rho), d["rho"].max())

px, py = outline_xy()
fig, ax = plt.subplots(2, len(hs), figsize=(4.5*len(hs), 10))

for k in range(len(hs)):
    h = hs[k]

    # Fortran: rho[ix, iy] -> transpose to [iy, ix] for imshow; pixel centers on the nodes
    e = L/2 + h/2
    im = ax[0, k].imshow(F[k].T, origin="lower", extent=[-e, e, -e, e],
                         vmin=lo, vmax=hi, cmap="viridis", interpolation="nearest")
    ax[0, k].set_title(f"Fortran  h = {h}")

    d = P[k]
    ax[1, k].tripcolor(d["x"], d["y"], d["tri"], d["rho"], shading="gouraud",
                       vmin=lo, vmax=hi, cmap="viridis")
    ax[1, k].set_title(f"pyTDGL  h = {h}")

    for r in range(2):
        ax[r, k].plot(px, py, "w-", lw=0.6)
        ax[r, k].set_xlim(-L/2, L/2)
        ax[r, k].set_ylim(-L/2, L/2)
        ax[r, k].set_aspect("equal")
        ax[r, k].set_xlabel(r"x ($\xi$)")
        ax[r, k].set_ylabel(r"y ($\xi$)")

fig.colorbar(im, ax=ax, shrink=0.8, label=r"$|\psi|^2$")
plt.savefig("fields_rect.pdf")
print("saved fields_rect.pdf")
