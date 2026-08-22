import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

# Fortran parameters
dimx = 60
dimy = 60
ymax = 30.0
n_t = 1000
tmax = 10.0
stride = 100

# one frame for all three panels
FRAME = 4

# external field (added to the induced B_z, same as your field scripts)
include_external_field = True
H_ext = 0.0353

# --- supercurrent quiver tuning (drawn on the B_z panel) ------------------
QUIV_STRIDE = 4          # arrow every Nth grid point (bigger = sparser). 150x150 is dense.
QUIV_SCALE = None        # None -> autoscale; set a number to fix arrow length
QUIV_COLOR = "black"
QUIV_WIDTH = 0.004
QUIV_ALPHA = 0.9

# --- global text sizing for POSTER use -----------------------------------
# Everything scales off these. Bump BASE up/down to make ALL text bigger/smaller.
BASE       = 22          # base font size (ticks, colorbar numbers)
TITLE_SIZE = 25          # panel titles (a)(b)(c)
LABEL_SIZE = 22          # x / y axis labels
CBAR_LABEL = 22          # colorbar labels
SUPTITLE   = 25          # the shared t = ... title

plt.rcParams.update({
    "font.size":        BASE,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  BASE,
    "ytick.labelsize":  BASE,
    "figure.titlesize": SUPTITLE,
})

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"geo2.png"

# data files (same folder convention as your other poster scripts)
psi_filename = "data/psi.bin"
A_filename   = "data/vec_pot.bin"
Js_filename  = "data/Js.bin"

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

# ------------------------------------------------------------------ loaders
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)
    expected_size = (n_snapshots + 1) * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )
    data = flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)
    return data


def load_current(filename):
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


# ------------------------------------------------------------------ load
print("Loading psi, vec_pot, Js...")
psi_all = load_psi(psi_filename)
psi = psi_all[1:]                       # drop initial condition (matches your scripts)
Jsx_all, Jsy_all = load_current(Js_filename)
Ax_all, Ay_all = load_vecpot(A_filename)
Bz_all = compute_curlA(Ax_all, Ay_all)

n_frames = min(psi.shape[0], Jsx_all.shape[0], Bz_all.shape[0])
if FRAME < 0 or FRAME >= n_frames:
    raise IndexError(f"FRAME={FRAME} out of range 0..{n_frames-1}.")

mag   = np.abs(psi[FRAME]) ** 2
phase = np.angle(psi[FRAME])
Jx    = Jsx_all[FRAME]
Jy    = Jsy_all[FRAME]
Bz    = Bz_all[FRAME]
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# ------------------------------------------------------------------ colormaps
mag_cmap = plt.cm.magma.copy()
mag_cmap.set_over("magenta")            # flag |psi|^2 > 1

phase_cmap = "twilight"                 # cyclic

bz_absmax = np.nanpercentile(np.abs(Bz), 100.0)
if not (np.isfinite(bz_absmax) and bz_absmax > 0):
    bz_absmax = 1.0
bz_norm = TwoSlopeNorm(vmin=-bz_absmax, vcenter=0.0, vmax=bz_absmax)

xs = np.linspace(0, xmax, dimx)
ys = np.linspace(0, ymax, dimy)
X, Y = np.meshgrid(xs, ys)              # [y, x]

# ------------------------------------------------------------------ figure
# Three SQUARE panels packed close together. constrained_layout OFF so we can
# drive the spacing manually with subplots_adjust (tight wspace = less whitespace).
fig, (ax_mag, ax_phase, ax_bz) = plt.subplots(1, 3, figsize=(14.0, 6.5))

TITLE_PAD = 13           # gap between panel title and the image (bigger = higher)
CBAR_FRAC = 0.038        # colorbar thickness (smaller = thinner bar)
CBAR_PAD  = 0.25         # gap between image and colorbar (bigger = more room for x-label)

def square(ax):
    ax.set_box_aspect(1)               # keep each panel square
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"x $(\xi)$", labelpad=8)
    ax.tick_params(labelsize=BASE, length=6, width=1.4)

# --- magnitude |psi|^2 ---
im_mag = ax_mag.imshow(
    mag, origin="lower", extent=extent,
    vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
)
ax_mag.set_title(r"Magnitude $|\psi|^2$", pad=TITLE_PAD)
ax_mag.set_ylabel(r"y $(\xi)$")
square(ax_mag)
cb_mag = fig.colorbar(im_mag, ax=ax_mag, extend="max",
                      orientation="horizontal", fraction=CBAR_FRAC, pad=CBAR_PAD)
cb_mag.set_label(r"$|\psi|^2$", fontsize=CBAR_LABEL)
cb_mag.ax.tick_params(labelsize=BASE)

# --- phase vartheta ---
im_phase = ax_phase.imshow(
    phase, origin="lower", extent=extent,
    cmap=phase_cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest",
)
ax_phase.set_title(r"Phase $\vartheta$", pad=TITLE_PAD)
square(ax_phase)
cb_ph = fig.colorbar(im_phase, ax=ax_phase, orientation="horizontal",
                     fraction=CBAR_FRAC, pad=CBAR_PAD,
                     ticks=[-np.pi, 0, np.pi])
cb_ph.set_label(r"$\vartheta$", fontsize=CBAR_LABEL)
cb_ph.ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
cb_ph.ax.tick_params(labelsize=BASE)

# --- B_z with supercurrent quiver ---
im_bz = ax_bz.imshow(
    Bz, origin="lower", extent=extent,
    cmap="PuOr", norm=bz_norm, interpolation="nearest",
)
s = QUIV_STRIDE
ax_bz.quiver(
    X[::s, ::s], Y[::s, ::s],
    Jx[::s, ::s], Jy[::s, ::s],
    color=QUIV_COLOR, width=QUIV_WIDTH,
    scale=QUIV_SCALE, scale_units="xy" if QUIV_SCALE else None,
    pivot="mid", alpha=QUIV_ALPHA,
)
_bz_title = r"$B_z + H_{ext}$" if include_external_field else r"$B_z$"
ax_bz.set_title(rf"Field {_bz_title} with $J_s$", pad=TITLE_PAD)
square(ax_bz)
cb_bz = fig.colorbar(im_bz, ax=ax_bz, orientation="horizontal",
                     fraction=CBAR_FRAC, pad=CBAR_PAD)
cb_bz.set_label(r"$(B_z + H_{ext})$ / $H_{c_2}$", fontsize=CBAR_LABEL)
cb_bz.ax.tick_params(labelsize=BASE)

# fig.suptitle(rf"$t = {t_here:.1f}\,\tau_{{GL}}$", fontsize=SUPTITLE, y=1.06)

# pack panels tightly: small wspace = little gap between the squares.
# top leaves room for BOTH the panel titles and the suptitle above them.
fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.10, wspace=0.0)

# ==================================================================
# dashed circles around the vortices
# ==================================================================
# CIRC_R      = 2.6              # circle radius in xi
# CIRC_COLOR  = "lime"           # visible over 'twilight'; try "cyan" / "white"
# CIRC_LW     = 2.2
# CIRC_DASH   = (0, (6, 4))      # 6 pts on, 4 pts off
# CIRCLE_AXES = [ax_phase]       # add ax_mag / ax_bz here to mark those too

# AUTO_FIND      = True          # False -> use MANUAL_CENTERS instead
# MANUAL_CENTERS = [(12.2, 17.0), (18.2, 17.0), (15.0, 11.0)]   # (x, y) in xi


# def find_vortices(phase, mag, h, min_mag=1e-6, merge_dist=1.5):
#     """Locate phase singularities by the winding number of each plaquette.

#     Around a closed loop enclosing a vortex, the phase must change by
#     2*pi*n. Summing the four wrapped corner-to-corner phase jumps of one
#     grid square gives exactly that n (0 almost everywhere, +/-1 at a core).
#     """
#     p = phase

#     def wrap(d):                      # fold any jump into (-pi, pi]
#         return np.angle(np.exp(1j * d))

#     d1 = wrap(p[:-1, 1:]  - p[:-1, :-1])
#     d2 = wrap(p[1:,  1:]  - p[:-1, 1:])
#     d3 = wrap(p[1:,  :-1] - p[1:,  1:])
#     d4 = wrap(p[:-1, :-1] - p[1:,  :-1])
#     wind = np.rint((d1 + d2 + d3 + d4) / (2.0 * np.pi)).astype(int)

#     # ignore plaquettes sitting in the blockade / vacuum
#     m4 = np.minimum.reduce([mag[:-1, :-1], mag[:-1, 1:],
#                             mag[1:,  :-1], mag[1:,  1:]])
#     wind[m4 <= min_mag] = 0

#     ii, jj = np.nonzero(wind)
#     pts = [((j + 0.5) * h, (i + 0.5) * h, int(wind[i, j]))
#            for i, j in zip(ii, jj)]

#     # merge neighbouring plaquettes that belong to the same core
#     merged = []
#     for x, y, n in pts:
#         for k, (mx, my, mn, c) in enumerate(merged):
#             if np.hypot(x - mx, y - my) < merge_dist and np.sign(n) == np.sign(mn):
#                 merged[k] = ((mx * c + x) / (c + 1),
#                              (my * c + y) / (c + 1), mn, c + 1)
#                 break
#         else:
#             merged.append((x, y, n, 1))
#     return [(x, y, n) for x, y, n, _ in merged]


# if AUTO_FIND:
#     centers = find_vortices(phase, mag, h)
#     print(f"Found {len(centers)} vortices:")
#     for x, y, n in centers:
#         print(f"   x = {x:6.2f} xi,  y = {y:6.2f} xi,  winding = {n:+d}")
#     centers = [(x, y) for x, y, n in centers]
# else:
#     centers = MANUAL_CENTERS

# for ax in CIRCLE_AXES:
#     for (cx, cy) in centers:
#         ax.add_patch(Circle(
#             (cx, cy), CIRC_R,
#             fill=False, edgecolor=CIRC_COLOR,
#             linestyle=CIRC_DASH, linewidth=CIRC_LW, zorder=6,
#         ))

fig.savefig(out_name, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_name}")