# Line cuts of the final |psi|^2 for every h, Fortran (solid) vs pyTDGL (dashed):
#   panel 1: across the channel, along y at x = 0
#   panel 2: along the center line, along x at y = 0
#   panel 3: just inside the notch faces, along x at y = y_corner (passes by the notch corners)
# Fortran is linearly interpolated between the two grid lines around the cut,
# pyTDGL is linearly interpolated on its triangles.
# Reads data_h=<h>/psi.bin and pytdgl_h=<h>.npz. Saves cuts_rect.pdf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from rect_geom import L, B

hs = [2.0, 1.0, 0.5, 0.25]
y_corner = B - 0.5          # cut height for panel 3
nq = 600                    # points along each pyTDGL cut


def fortran_line(rho, h, x0, along):
    # rho[ix, iy] on nodes at -L/2 + k*h.
    # along = "y": values along y at x = x0.   along = "x": values along x at y = x0.
    n = rho.shape[0]
    s = -L/2 + h * np.arange(n)
    k = int(np.floor((x0 + L/2) / h))
    k = min(max(k, 0), n - 2)
    w = (x0 - s[k]) / h
    if along == "y":
        a = rho[k, :]; b = rho[k+1, :]
    else:
        a = rho[:, k]; b = rho[:, k+1]
    if w < 1e-12:
        return s, a.copy()
    if w > 1.0 - 1e-12:
        return s, b.copy()
    return s, (1.0 - w) * a + w * b      # NaN if either side is blockade


def pytdgl_line(d, x0, along):
    tri = mtri.Triangulation(d["x"], d["y"], d["tri"])
    f = mtri.LinearTriInterpolator(tri, d["rho"])
    s = np.linspace(-L/2, L/2, nq)
    if along == "y":
        v = f(np.full(nq, x0), s)
    else:
        v = f(s, np.full(nq, x0))
    return s, np.ma.filled(v, np.nan)    # NaN outside the film


cuts = [("y", 0.0,      r"along y at x = 0"),
        ("x", 0.0,      r"along x at y = 0"),
        ("x", y_corner, rf"along x at y = {y_corner}")]

colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(hs)))
fig, ax = plt.subplots(1, len(cuts), figsize=(6*len(cuts), 4.8))

for k in range(len(hs)):
    h = hs[k]
    n = int(L / h) + 1
    psi = np.fromfile(f"data_h={h}/psi.bin", dtype=np.complex128).reshape(-1, n, n)
    rho = np.abs(psi[-1])**2
    rho[rho == 0.0] = np.nan
    d = np.load(f"pytdgl_h={h}.npz")

    for c in range(len(cuts)):
        along, x0, name = cuts[c]
        s, v = fortran_line(rho, h, x0, along)
        ax[c].plot(s, v, "o-", ms=2.5, lw=1.0, color=colors[k], label=f"Fortran h={h}")
        s, v = pytdgl_line(d, x0, along)
        ax[c].plot(s, v, "--", lw=1.0, color=colors[k], label=f"pyTDGL h={h}")

for c in range(len(cuts)):
    along, x0, name = cuts[c]
    ax[c].set_title(name)
    ax[c].set_xlabel(r"y ($\xi$)" if along == "y" else r"x ($\xi$)")
    ax[c].set_ylabel(r"$|\psi|^2$")
    ax[c].set_xlim(-L/2, L/2)
ax[0].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("cuts_rect.pdf")
print("saved cuts_rect.pdf")
