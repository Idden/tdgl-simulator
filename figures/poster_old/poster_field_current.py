import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ==================================================================== config
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

# ---- OVERLAY STYLE -------------------------------------------------------
# "quiver"     -> current shown as arrows (direction + relative magnitude)
# "streamline" -> current shown as smooth streamlines (flow paths / crowding)
OVERLAY = "quiver"          # <-- flip between "quiver" and "streamline"

# quiver tuning
QUIV_STRIDE = 2             # plot an arrow every Nth grid point (bigger = sparser)
QUIV_SCALE = None           # None -> matplotlib autoscale; set a number to fix arrow length
QUIV_COLOR = "black"
QUIV_WIDTH = 0.001

# streamline tuning
STREAM_DENSITY = 1.4        # higher -> more streamlines
STREAM_COLOR = "black"
STREAM_LINEWIDTH = 0.4
# color streamlines by |J| instead of a flat color? (uses a light colormap)
STREAM_COLOR_BY_MAG = False

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_field_and_current_{OVERLAY}_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.pdf"

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

# fields for this frame
Jx = Jsx_all[FRAME]                             # [y, x]
Jy = Jsy_all[FRAME]
Jmag = np.sqrt(Jx ** 2 + Jy ** 2)               # crowding map (not drawn here, used for scaling)
Bz = Bz_all[FRAME]
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# ------------------------------------------------------------------ colormap for B_z
# B_z is signed (can be + or -), so a diverging map centered at 0 reads best.
bz_absmax = np.nanpercentile(np.abs(Bz), 100.0)
if not (np.isfinite(bz_absmax) and bz_absmax > 0):
    bz_absmax = 1.0
bz_norm = TwoSlopeNorm(vmin=-bz_absmax, vcenter=0.0, vmax=bz_absmax)

# real-space coordinate grids for x and y (cell centers)
xs = np.linspace(0, xmax, dimx)
ys = np.linspace(0, ymax, dimy)
X, Y = np.meshgrid(xs, ys)                      # [y, x]

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)

# --- background: magnetic field B_z ---
im_bz = ax.imshow(
    Bz, origin="lower", extent=extent,
    cmap="PuOr", norm=bz_norm, interpolation="nearest",
)
cb_bz = fig.colorbar(im_bz, ax=ax, fraction=0.046, pad=0.02, shrink=0.85)
cb_bz.set_label(r"$B_z$ / $H_{c_2}$", fontsize=12)

# --- overlay: supercurrent vectors ---
if OVERLAY == "quiver":
    s = QUIV_STRIDE
    q = ax.quiver(
        X[::s, ::s], Y[::s, ::s],
        Jx[::s, ::s], Jy[::s, ::s],
        color=QUIV_COLOR, width=QUIV_WIDTH,
        scale=QUIV_SCALE, scale_units="xy" if QUIV_SCALE else None,
        pivot="mid", alpha=0.9,
    )
elif OVERLAY == "streamline":
    if STREAM_COLOR_BY_MAG:
        strm = ax.streamplot(
            xs, ys, Jx, Jy,
            density=STREAM_DENSITY, color=Jmag, cmap="Greys",
            linewidth=STREAM_LINEWIDTH, arrowsize=0.8,
        )
    else:
        ax.streamplot(
            xs, ys, Jx, Jy,
            density=STREAM_DENSITY, color=STREAM_COLOR,
            linewidth=STREAM_LINEWIDTH, arrowsize=0.8,
        )
else:
    raise ValueError(f"OVERLAY must be 'quiver' or 'streamline', got {OVERLAY!r}")

_bz_title = r"$B_z = (\nabla\times A)_z + H_{ext}$" \
    if include_external_field else r"$B_z = (\nabla\times A)_z$"
ax.set_title(f"Magnetic Field {_bz_title} with Supercurrent $J_s$", fontsize=14)
ax.set_xlabel(r"x $(\xi)$", fontsize=12)
ax.set_ylabel(r"y $(\xi)$", fontsize=12)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")