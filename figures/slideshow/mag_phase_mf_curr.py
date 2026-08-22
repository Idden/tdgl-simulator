import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm

# Fortran parameters
dimx = 150
dimy = 150
ymax = 30.0
n_t = 600000
tmax = 1000.0
stride = 3000

fps = 10
frame_skip = 1

include_external_field = True
H_ext = 0.0353
mask_blockade = False

OUTPUT_MODE = "combined"

# video name vars
sim_ext_mag = "00353"
sim_curr = "015"

psi_filename = "data/psi.bin"
A_filename   = "data/vec_pot.bin"
Js_filename  = "data/Js.bin"

# quiver tuning
QUIV_STRIDE = 4
QUIV_COLOR = "black"
QUIV_WIDTH = 0.004
QUIV_ALPHA = 0.9
QUIV_SCALE = None        # None -> pick one fixed scale from the data (see below)
QUIV_TARGET_LEN = 1.6    # only used when QUIV_SCALE is None:
                         # a "typical" arrow is this many grid-spacings long
QUIV_PCT = 99.0          # percentile of |J_s| that counts as "typical"

# text sizing
BASE       = 18          # base font size (ticks, colorbar numbers); 22 for poster stills
TITLE_SIZE = 20
LABEL_SIZE = 18
CBAR_LABEL = 18
SUPTITLE   = 20

plt.rcParams.update({
    "font.size":        BASE,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  BASE,
    "ytick.labelsize":  BASE,
    "figure.titlesize": SUPTITLE,
})

# derived
n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

xs = np.linspace(0, xmax, dimx)
ys = np.linspace(0, ymax, dimy)
X, Y = np.meshgrid(xs, ys)              # [y, x]

# big vector-potential grid (same convention as gTDGL_plotting.py / curlA.py)
padx = (3 * dimy - dimx) // 2
pady = dimy
dimA = 3 * dimy


# blockade mask
def make_blockade_mask(dimx, dimy):
    mask = np.zeros((dimy, dimx), dtype=bool)

    ci = (dimy + 1) / 2.0
    cj = (dimx + 1) / 2.0
    lead_half = 0.45 * dimy
    w0_half = 0.57 * lead_half
    R = 0.20 * dimy

    for i0 in range(dimy):
        for j0 in range(dimx):
            i = i0 + 1
            j = j0 + 1

            xx = j - cj
            half_w = lead_half - (lead_half - w0_half) / (1.0 + (xx / R)**2)

            if abs(i - ci) > half_w:
                mask[i0, j0] = True

    return mask


# loaders
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)
    expected_size = (n_snapshots + 1) * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )
    return flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)


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
    Ax = flat[:, 0, :, :].transpose(0, 2, 1)    # [frame, y, x] on big grid
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


def save_animation(animation, mp4_name, fps=15):
    print(f"Saving {mp4_name}...")
    try:
        animation.save(mp4_name, writer=FFMpegWriter(fps=fps))
    except Exception:
        gif_name = mp4_name.replace(".mp4", ".gif")
        print(f"ffmpeg not found, falling back to GIF: {gif_name}")
        animation.save(gif_name, writer=PillowWriter(fps=fps))
    print("Done.")


# load
print("Loading psi, vec_pot, Js...")
psi_all = load_psi(psi_filename)
psi = psi_all[1:]                       # drop initial condition
Jsx_all, Jsy_all = load_current(Js_filename)
Ax_all, Ay_all = load_vecpot(A_filename)
Bz_all = compute_curlA(Ax_all, Ay_all)

n_frames = min(psi.shape[0], Jsx_all.shape[0], Bz_all.shape[0])
psi = psi[:n_frames]
Jsx_all = Jsx_all[:n_frames]
Jsy_all = Jsy_all[:n_frames]
Bz_all = Bz_all[:n_frames]

if mask_blockade:
    blockade_mask = make_blockade_mask(dimx, dimy)
    psi = psi.copy()
    Jsx_all = Jsx_all.copy()
    Jsy_all = Jsy_all.copy()
    Bz_all = Bz_all.copy()
    psi[:, blockade_mask] = np.nan
    Jsx_all[:, blockade_mask] = np.nan
    Jsy_all[:, blockade_mask] = np.nan
    Bz_all[:, blockade_mask] = np.nan

frames = list(range(0, n_frames, frame_skip))
print(f"{n_frames} frames available, animating {len(frames)} of them.")

# colour limits
# Fixed for the whole movie, otherwise the colour scale flickers frame to frame.
mag_cmap = plt.cm.magma.copy()
mag_cmap.set_over("magenta")            # flag |psi|^2 > 1
phase_cmap = "twilight"                 # cyclic

sample_frames = np.linspace(0, n_frames - 1, min(200, n_frames), dtype=int)

bz_vals = np.abs(Bz_all[sample_frames]).ravel()
bz_vals = bz_vals[np.isfinite(bz_vals)]
bz_absmax = np.percentile(bz_vals, 100.0) if bz_vals.size else 1.0
if not (np.isfinite(bz_absmax) and bz_absmax > 0):
    bz_absmax = 1.0
bz_norm = TwoSlopeNorm(vmin=-bz_absmax, vcenter=0.0, vmax=bz_absmax)
print(f"B_z colour limit: +/- {bz_absmax:.6e}")

# one fixed quiver scale for the whole movie
if QUIV_SCALE is None:
    jmag = np.sqrt(Jsx_all[sample_frames]**2 + Jsy_all[sample_frames]**2).ravel()
    jmag = jmag[np.isfinite(jmag)]
    j_ref = np.percentile(jmag, QUIV_PCT) if jmag.size else 1.0
    if not (np.isfinite(j_ref) and j_ref > 0):
        j_ref = 1.0
    target_len = QUIV_TARGET_LEN * QUIV_STRIDE * h      # in units of xi
    quiv_scale = j_ref / target_len                     # data units per xi
else:
    quiv_scale = QUIV_SCALE
print(f"quiver scale: {quiv_scale:.6e}")

# panel helpers
TITLE_PAD = 13
CBAR_FRAC = 0.038
CBAR_PAD  = 0.25
s = QUIV_STRIDE

_bz_title = r"$B_z + H_{ext}$" if include_external_field else r"$B_z$"


def square(ax):
    ax.set_box_aspect(1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"x $(\xi)$", labelpad=8)
    ax.tick_params(labelsize=BASE, length=6, width=1.4)


def add_mag_panel(fig, ax, frame0):
    im = ax.imshow(
        np.abs(psi[frame0])**2, origin="lower", extent=extent,
        vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
    )
    ax.set_title(r"Magnitude $|\psi|^2$", pad=TITLE_PAD)
    ax.set_ylabel(r"y $(\xi)$")
    square(ax)
    cb = fig.colorbar(im, ax=ax, extend="max", orientation="horizontal",
                      fraction=CBAR_FRAC, pad=CBAR_PAD)
    cb.set_label(r"$|\psi|^2$", fontsize=CBAR_LABEL)
    cb.ax.tick_params(labelsize=BASE)
    return im


def add_phase_panel(fig, ax, frame0, ylabel=False):
    im = ax.imshow(
        np.angle(psi[frame0]), origin="lower", extent=extent,
        cmap=phase_cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    ax.set_title(r"Phase $\arg(\psi)$", pad=TITLE_PAD)
    if ylabel:
        ax.set_ylabel(r"y $(\xi)$")
    square(ax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      fraction=CBAR_FRAC, pad=CBAR_PAD,
                      ticks=[-np.pi, 0, np.pi])
    cb.set_label(r"$\arg(\psi)$", fontsize=CBAR_LABEL)
    cb.ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    cb.ax.tick_params(labelsize=BASE)
    return im


def add_field_panel(fig, ax, frame0, ylabel=False):
    im = ax.imshow(
        Bz_all[frame0], origin="lower", extent=extent,
        cmap="PuOr", norm=bz_norm, interpolation="nearest",
    )
    qv = ax.quiver(
        X[::s, ::s], Y[::s, ::s],
        Jsx_all[frame0][::s, ::s], Jsy_all[frame0][::s, ::s],
        color=QUIV_COLOR, width=QUIV_WIDTH,
        angles="xy", scale_units="xy", scale=quiv_scale,
        pivot="mid", alpha=QUIV_ALPHA,
    )
    ax.set_title(rf"Field {_bz_title} with $J_s$", pad=TITLE_PAD)
    if ylabel:
        ax.set_ylabel(r"y $(\xi)$")
    square(ax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      fraction=CBAR_FRAC, pad=CBAR_PAD)
    cb.set_label(r"$B_z$ / $H_{c_2}$", fontsize=CBAR_LABEL)
    cb.ax.tick_params(labelsize=BASE)
    return im, qv


def time_text(frame):
    return rf"$t = {(frame + 1) * dt_snap:.1f}\,\tau_{{GL}}$"


# combined movie
def make_combined():
    fig, (ax_mag, ax_phase, ax_bz) = plt.subplots(1, 3, figsize=(14.0, 7.2))

    im_mag = add_mag_panel(fig, ax_mag, frames[0])
    im_phase = add_phase_panel(fig, ax_phase, frames[0])
    im_bz, quiv = add_field_panel(fig, ax_bz, frames[0])

    title = fig.suptitle(time_text(frames[0]), fontsize=SUPTITLE, y=0.965)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.10, wspace=0.0)

    def update(frame):
        im_mag.set_data(np.abs(psi[frame])**2)
        im_phase.set_data(np.angle(psi[frame]))
        im_bz.set_data(Bz_all[frame])
        quiv.set_UVC(Jsx_all[frame][::s, ::s], Jsy_all[frame][::s, ::s])
        title.set_text(time_text(frame))
        return im_mag, im_phase, im_bz, quiv, title

    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    save_animation(ani, f"mag_phase_mf_curr_{sim_ext_mag}mag_{sim_curr}curr.mp4",
                   fps=fps)
    plt.close(fig)


# single-panel movies
def make_single(kind):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    if kind == "mag":
        im = add_mag_panel(fig, ax, frames[0])
        quiv = None
    elif kind == "phase":
        im = add_phase_panel(fig, ax, frames[0], ylabel=True)
        quiv = None
    elif kind == "field":
        im, quiv = add_field_panel(fig, ax, frames[0], ylabel=True)
    else:
        raise ValueError("kind must be 'mag', 'phase' or 'field'.")

    title = fig.suptitle(time_text(frames[0]), fontsize=SUPTITLE, y=0.97)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.86, bottom=0.10)

    def update(frame):
        if kind == "mag":
            im.set_data(np.abs(psi[frame])**2)
        elif kind == "phase":
            im.set_data(np.angle(psi[frame]))
        else:
            im.set_data(Bz_all[frame])
            quiv.set_UVC(Jsx_all[frame][::s, ::s], Jsy_all[frame][::s, ::s])
        title.set_text(time_text(frame))
        return (im, title) if quiv is None else (im, quiv, title)

    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    save_animation(ani, f"anim_{kind}_{sim_ext_mag}mag_{sim_curr}curr.mp4", fps=fps)
    plt.close(fig)


if OUTPUT_MODE in ("combined", "both"):
    make_combined()
if OUTPUT_MODE in ("separate", "both"):
    for kind in ("mag", "phase", "field"):
        make_single(kind)