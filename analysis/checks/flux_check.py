import numpy as np

# match Fortran parameters
dim = 60
xymax = 30.0
h = xymax / (dim - 1)
H_ext = 0.0353
pad = dim
dimA = 3 * dim
Phi0 = 2 * np.pi
stride = 200
H_ramp = 5000

# load data
A_frames = np.fromfile("vec_pot60.bin", dtype=np.float64).reshape(-1, 2 * dimA * dimA)
psi = np.fromfile("psi60.bin", dtype=np.complex128).reshape(-1, dim * dim)[1:]
Js = np.fromfile("Js60.bin", dtype=np.float64).reshape(-1, 2 * dim * dim)

# helper functions
def H_now(f):
    return H_ext * min((f + 1) * stride / H_ramp, 1.0)

def path(v):
    return (0.5 * (v[:-1] + v[1:])).sum()

# calculate the flux with Js
def fluxoid(f, m):
    Ax = A_frames[f, :dimA * dimA].reshape(dimA, dimA, order="F")
    Ay = A_frames[f, dimA * dimA:].reshape(dimA, dimA, order="F")
    p2 = np.abs(psi[f].reshape(dim, dim, order="F"))**2

    with np.errstate(invalid="ignore"):
        Jx = Js[f, :dim * dim].reshape(dim, dim, order="F") / p2
        Jy = Js[f, dim * dim:].reshape(dim, dim, order="F") / p2

    i0 = j0 = m
    i1 = j1 = dim - 1 - m

    # circulation of A and flux
    circ = h * (Ax[:-1, :-1] + Ay[:-1, 1:] - Ax[1:, :-1] - Ay[:-1, :-1]) + H_now(f) * h**2
    flux = circ[pad + i0:pad + i1, pad + j0:pad + j1].sum()

    # take into account Js
    curr = h * (path(Jx[i0, j0:j1 + 1]) + path(Jy[i0:i1 + 1, j1])
                - path(Jx[i1, j0:j1 + 1]) - path(Jy[i0:i1 + 1, j0]))
    
    return flux / Phi0, curr / Phi0

# main loop
frame = 149
print(f"frame {frame}")
for m in [15, 17, 19, 21, 23]:
    fl, cu = fluxoid(frame, m)
    tot = fl + cu
    print(f"m={m:2d}  flux={fl:7.3f}  current_term={cu:7.3f}  fluxoid={tot:7.3f}  n={round(tot)}")