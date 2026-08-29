# Fluxoid quantization check. For a closed loop inside the SC the fluxoid
#
#     (flux of B through the loop) + (line integral of Js/|psi|^2 around it)
#
# has to come out as an integer multiple of Phi0. Counting that integer counts
# the vortices inside the loop, and getting a clean integer is the check that
# the solver is behaving. Loops of several sizes are tried so the answer can be
# seen to be independent of where the loop is drawn.

import numpy as np

# Match Fortran parameters
dim = 60
xymax = 30.0
h = xymax / (dim - 1)
H_ext = 0.0353
pad = dim
dimA = 3 * dim
Phi0 = 2 * np.pi # flux quantum in the code's units
stride = 250
H_ramp = 5000

# Load data
A_frames = np.fromfile("vec_pot60.bin", dtype=np.float64).reshape(-1, 2 * dimA * dimA)
psi = np.fromfile("psi60.bin", dtype=np.complex128).reshape(-1, dim * dim)[1:]
Js = np.fromfile("Js60.bin", dtype=np.float64).reshape(-1, 2 * dim * dim)

# Helper functions
# H_now rebuilds the applied field at frame f, since the solver ramps it in
# over H_ramp steps rather than switching it on at once.
def H_now(f):
    return H_ext * min((f + 1) * stride / H_ramp, 1.0)

# Trapezoid rule along one edge of the loop.
def path(v):
    return (0.5 * (v[:-1] + v[1:])).sum()

# Fluxoid for the square loop inset m cells from the edge of the SC block
def fluxoid(f, m):
    Ax = A_frames[f, :dimA * dimA].reshape(dimA, dimA, order="F")
    Ay = A_frames[f, dimA * dimA:].reshape(dimA, dimA, order="F")
    p2 = np.abs(psi[f].reshape(dim, dim, order="F"))**2

    # Js/|psi|^2 is the superfluid velocity; it blows up where psi -> 0, and
    # those cells sit in the blockade and never end up inside a loop anyway
    with np.errstate(invalid="ignore"):
        Jx = Js[f, :dim * dim].reshape(dim, dim, order="F") / p2
        Jy = Js[f, dim * dim:].reshape(dim, dim, order="F") / p2

    i0 = j0 = m
    i1 = j1 = dim - 1 - m

    # Flux of B through the loop, from the circulation of A around each
    # plaquette plus the uniform applied field
    circ = h * (Ax[:-1, :-1] + Ay[:-1, 1:] - Ax[1:, :-1] - Ay[:-1, :-1]) + H_now(f) * h**2
    flux = circ[pad + i0:pad + i1, pad + j0:pad + j1].sum()

    # Line integral of the superfluid velocity around the loop, counter
    # clockwise: bottom edge + right edge - top edge - left edge
    curr = h * (path(Jx[i0, j0:j1 + 1]) + path(Jy[i0:i1 + 1, j1])
                - path(Jx[i1, j0:j1 + 1]) - path(Jy[i0:i1 + 1, j0]))
    
    return flux / Phi0, curr / Phi0

# Main loop
# m determines the size of the loop in coherence lengths from the superconducting
# boundary. Larger m means smaller loop and smaller m means larger loop.
# A clean run prints the same n for every m -- the fluxoid should not care
# how big the loop is, only how many vortices it encloses.
frame = 149
print(f"frame {frame}")
for m in [15, 17, 19, 21, 23]:
    fl, cu = fluxoid(frame, m)
    tot = fl + cu
    print(f"m={m:2d}  flux={fl:7.3f}  current_term={cu:7.3f}  fluxoid={tot:7.3f}  n={round(tot)}")