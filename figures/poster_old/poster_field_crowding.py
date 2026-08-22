import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# (keep in sync with the Fortran run, same as your other poster scripts)
dimx = 120
dimy = 60
ymax = 30.0
n_t = 30000
tmax = 300.0
stride = 200

include_external_field = True
H_ext = 0.0353

# which snapshot to show. Pick a frame with vortices sitting in the channel so
# both B_z spots and the current crowding are visible.
FRAME = 100

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_field_crowding_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.pdf"

# ------------------------------------------------------------------ derived
n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

# big vector-potential grid (same convention as gTDGL_plotting.py / curlA.py)
padx = (3 * dimy - dimx) // 2
pady = dimy
dimA = 3 * dimy

Js_filename = "data/Js.bin"
A_filename = "data/vec_pot.bin"

# ------------------------------------------------------------------ loaders
def load_current(filename):
    # supercurrent Jsx, Jsy on the dimx*dimy grid (from gTDGL_plotting.py)
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * 2 * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Jx and Jy for each saved frame."
        )
    flat = flat.reshape(n_snapshots, 2, dimx, dimy)
    Jsx = flat[:, 0, :, :].transpose(0, 2, 1)   # [frame, y, x]
    Jsy = flat[:, 1, :, :].transpose(0, 2, 1)
    return Jsx, Jsy


def load_vecpot(filename):
    # Ax, Ay on the big dimA*dimA grid (from gTDGL_plotting.py)
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * 2 * dimA * dimA
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Ax and Ay on the {dimA}x{dimA} grid for each saved frame."
        )
    flat = flat.reshape(n_snapshots, 2, dimA, dimA)
    Ax = flat[:, 0, :, :].transpose(0, 2, 1)   # [frame, y, x] on big grid
    Ay = flat[:, 1, :, :].transpose(0, 2, 1)
    return Ax, Ay


def compute_curlA(Ax_big, Ay_big):
    # B_z = curl(A), cropped to the SC sub-block (same as gTDGL_plotting.py)
    n_fr = Ax_big.shape[0]
    Bz = np.empty((n_fr, dimy, dimx))
    for f in range(n_fr):
        dAy_dx = np.gradient(Ay_big[f], h, axis=1)
        dAx_dy = np.gradient(Ax_big[f], h, axis=0)
        Bz_big = dAy_dx - dAx_dy
        Bz[f] = Bz_big[pady:pady + dimy, padx:padx + dimx]
    if include_external_field:
        Bz = Bz + H_ext
    return Bz


print("Loading Js and vec_pot...")
Jsx_all, Jsy_all = load_current(Js_filename)
Ax_all, Ay_all = load_vecpot(A_filename)
Bz_all = compute_curlA(Ax_all, Ay_all)

if FRAME < 0 or FRAME >= Jsx_all.shape[0]:
    raise IndexError(f"FRAME={FRAME} out of range 0..{Jsx_all.shape[0]-1}.")

# current-density MAGNITUDE |J| = sqrt(Jx^2 + Jy^2) -- this is the crowding map
Jmag = np.sqrt(Jsx_all[FRAME] ** 2 + Jsy_all[FRAME] ** 2)
Bz = Bz_all[FRAME]
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# ------------------------------------------------------------------ colormaps
# B_z is signed (can be + or -), so a diverging map centered at 0 reads best.
# |J| is >= 0, so magma (your poster map) for the crowding panel.
from matplotlib.colors import TwoSlopeNorm

# symmetric color limit for B_z about 0
bz_absmax = np.nanpercentile(np.abs(Bz), 100.0)
if not (np.isfinite(bz_absmax) and bz_absmax > 0):
    bz_absmax = 1.0
bz_norm = TwoSlopeNorm(vmin=-bz_absmax, vcenter=0.0, vmax=bz_absmax)

jmag_cmap = plt.cm.magma
jmax = np.nanpercentile(Jmag, 100.0)   # 99th pct so one hot pixel doesn't wash it out
if not (np.isfinite(jmax) and jmax > 0):
    jmax = 1.0

# ------------------------------------------------------------------ figure (up/down)
# Same stacked format as your other posters: magma + shrink=0.7 colorbars, tight gap.
fig, (ax_bz, ax_j) = plt.subplots(
    2, 1, figsize=(6.2, 6.5), constrained_layout=True
)
fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.02)

# --- top: magnetic field B_z ---
im_bz = ax_bz.imshow(
    Bz, origin="lower", extent=extent,
    cmap="PuOr", norm=bz_norm, interpolation="nearest",
)
_bz_title = r"Magnetic Field $B_z = (\nabla\times A)_z + H_{ext}$" \
    if include_external_field else r"Induced Field $B_z = (\nabla\times A)_z$"
ax_bz.set_title(_bz_title, fontsize=14)
ax_bz.set_xlabel(r"x $(\xi)$", fontsize=12)
ax_bz.set_ylabel(r"y $(\xi)$", fontsize=12)
cb_bz = fig.colorbar(im_bz, ax=ax_bz, fraction=0.046, pad=0.02, shrink=0.7)
cb_bz.set_label(r"$B_z$ / $H_{c_2}$", fontsize=12)

# --- bottom: current crowding |J| ---
im_j = ax_j.imshow(
    Jmag, origin="lower", extent=extent,
    cmap=jmag_cmap, vmin=0.0, vmax=jmax, interpolation="nearest",
)
ax_j.set_title(r"Current Crowding $|J_s| = \sqrt{J_x^2 + J_y^2}$", fontsize=14)
ax_j.set_xlabel(r"x $(\xi)$", fontsize=12)
ax_j.set_ylabel(r"y $(\xi)$", fontsize=12)
cb_j = fig.colorbar(im_j, ax=ax_j, fraction=0.046, pad=0.02, shrink=0.7)
cb_j.set_label(r"$|J_s|$ / $J_0$", fontsize=12)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")