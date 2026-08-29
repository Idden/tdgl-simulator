# Phase quantization check, and the vortex counter. Walking the gauge
# invariant phase around one grid plaquette must come back to a multiple of
# 2*pi; that integer is the winding number, and a nonzero one means a vortex
# core sits in that plaquette. Prints a count for every frame that has any.

import numpy as np

# Match Fortran parameters
dim = 60
xymax = 30.0
h = xymax / (dim - 1)
H_ext = 0.0353
center = (dim + 1)/2
# Fold a phase difference into (-pi, pi] so a wrap from -pi to pi does not
# read as a huge gradient
wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi

# Blockade geometry, same lorentzian constriction the solver uses
ci = cj = (dim + 1)/2
lead_half, R = 0.45*dim, 0.20*dim
w0_half = 0.57*lead_half
ii = np.arange(1, dim+1); jj = np.arange(1, dim+1)
J, I = np.meshgrid(jj, ii, indexing="ij")
half_w = lead_half - (lead_half - w0_half)/(1 + ((J - cj)/R)**2)
blk = np.abs(I - ci) > half_w

# A plaquette touching the blockade has meaningless phase, so mark any
# plaquette with a blockade cell on any of its four corners
bad = blk[:-1,:-1] | blk[1:,:-1] | blk[:-1,1:] | blk[1:,1:]

# Load data. psi.bin carries one extra leading frame (the initial
# condition), so drop it to line up with the saved time-loop output.
psi = np.fromfile("psi.bin", dtype=np.complex128).reshape(-1, dim, dim)
psi = psi[1:]
# theta = np.angle(psi[200])

d3 = 3 * dim

vec = np.fromfile("vec_pot.bin", dtype=np.float64).reshape(-1, 2, d3, d3)

for f in range(psi.shape[0]):
    theta = np.angle(psi[f])
    Ax_full, Ay_full = vec[f, 0], vec[f, 1]
    # A lives on the 3x grid, so cut out the SC block from the middle
    Ax_in = Ax_full[dim:2*dim, dim:2*dim]
    Ay_in = Ay_full[dim:2*dim, dim:2*dim]
    # Applied field as a vector potential in the Landau gauge, which the
    # solver holds separately from the induced A
    j = np.arange(1, dim+1)[:, None]
    a_y = (H_ext * (j - center) * h) * h

    # Gauge invariant phase differences: d(theta) - A along each link
    d_i = wrap(theta[:, 1:] - theta[:, :-1] - (a_y + Ay_in[:, :-1]) * h) # 0(i+1, j) - 0(i, j) - A_i
    d_j = wrap(theta[1:, :] - theta[:-1, :] - Ax_in[:-1, :] * h) # 0(i, j+1) - 0(i, j)

    # Line integral around each plaquette (counter clockwise)
    circ = d_i[1:, :] + d_j[:, :-1] - d_i[:-1, :] - d_j[:, 1:]
    vort = np.round(circ / (2*np.pi)).astype(int)

    vort[bad] = 0 # ignore anything sitting in the blockade
    if np.abs(vort).sum() > 0:
        print(f"frame {f} (t={f*200}): {np.abs(vort).sum()} vortices, net {vort.sum()}")

# j = np.arange(1, dim+1)[:, None]
# a_y = (H_ext * (j - center) * h) * h

# # take gradient
# d_i = wrap(theta[:, 1:] - theta[:, :-1] - a_y) # 0(i+1, j) - 0(i, j) - A_i
# d_j = wrap(theta[1:, :] - theta[:-1, :]) # 0(i, j+1) - 0(i, j)

# # line integral (counter clockwise)
# circ = d_i[1:, :] + d_j[:, :-1] - d_i[:-1, :] - d_j[:, 1:]
# vort = np.round(circ / (2*np.pi)).astype(int)

# # take into account phase change from blockade
# ci = cj = (dim + 1)/2
# lead_half, R = 0.45*dim, 0.20*dim
# w0_half = 0.57*lead_half
# ii = np.arange(1, dim+1); jj = np.arange(1, dim+1)
# J, I = np.meshgrid(jj, ii, indexing="ij")
# half_w = lead_half - (lead_half - w0_half)/(1 + ((J - cj)/R)**2)
# blk = np.abs(I - ci) > half_w

# # disinclude phase winding from blockade sites
# bad = blk[:-1,:-1] | blk[1:,:-1] | blk[:-1,1:] | blk[1:,1:]
# vort[bad] = 0

# print("net winding:", int(vort.sum())) 
# print("num vortices:", int(np.abs(vort).sum()))