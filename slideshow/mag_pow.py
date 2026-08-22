import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize, LogNorm

# ------------------------------------------------------------------ Fortran parameters
dimx = 150
dimy = 150
ymax = 30.0
n_t = 600000
tmax = 1000.0
stride = 3000          # psi dump stride
stride_power = 300     # power dump stride (set = stride if they match)

fps = 10
frame_skip = 1         # 1 = every master frame, 2 = every other, ...

ANIM_BASE = "psi"      # "psi" -> one movie frame per psi dump (coarser, shorter movie)
                       # "power" -> one movie frame per power dump (finer, longer movie)

mask_blockade = True   # blank the vacuum blockade in the power panel
mask_mag = False       # also blank it in the |psi|^2 panel
log_scale = True       # log colour scale for the dissipation panel

# output
sim_ext_mag = "00353"
sim_curr = "015"
video_name = f"mag_pow_{sim_ext_mag}mag_{sim_curr}curr.mp4"

# data files
psi_filename = "data/psi.bin"
power_filename = "data/power.bin"

# ------------------------------------------------------------------ text sizing
BASE       = 18        # ticks, colorbar numbers  (22 for poster stills)
TITLE_SIZE = 20        # panel titles
LABEL_SIZE = 18        # axis labels
CBAR_LABEL = 18        # colorbar labels
SUPTITLE   = 20        # the shared t = ... title

plt.rcParams.update({
    "font.size":        BASE,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  BASE,
    "ytick.labelsize":  BASE,
    "figure.titlesize": SUPTITLE,
})

# ------------------------------------------------------------------ derived
n_snap_psi = n_t // stride
n_snap_pow = n_t // stride_power
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt              # psi frame spacing
dt_snap_pow = stride_power * dt    # power frame spacing
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]


# ------------------------------------------------------------------ blockade mask
def make_blockade_mask(dimx, dimy):
    mask = np.zeros((dimy, dimx), dtype=bool)
    ci = (dimy + 1) / 2.0
    cj = (dimx + 1) / 2.0
    lead_half = 0.45 * dimy
    w0_half = 0.57 * lead_half
    R = 0.20 * dimy
    j0 = np.arange(dimx)[None, :] + 1
    i0 = np.arange(dimy)[:, None] + 1
    xx = j0 - cj
    half_w = lead_half - (lead_half - w0_half) / (1.0 + (xx / R) ** 2)
    mask[:] = np.abs(i0 - ci) > half_w
    return mask


blockade_mask = make_blockade_mask(dimx, dimy)


# ------------------------------------------------------------------ loaders
def _frames_in_file(filename, itemsize, expect_frames, label):
    """Return the actual frame count in the file, warning if it disagrees."""
    nbytes = os.path.getsize(filename)
    per_frame = dimx * dimy * itemsize
    if nbytes % per_frame:
        raise ValueError(
            f"{filename}: {nbytes} bytes is not a whole number of "
            f"{dimx}x{dimy} {label} frames."
        )
    got = nbytes // per_frame
    if got != expect_frames:
        print(
            f"NOTE: {filename} holds {got} frames, script expected "
            f"{expect_frames}. Using {got} (check stride / stride_power / n_t)."
        )
    return got


def load_psi(filename):
    """psi.bin: one extra leading frame (the initial condition)."""
    n_frames = _frames_in_file(filename, 16, n_snap_psi + 1, "complex")
    flat = np.fromfile(filename, dtype=np.complex128)
    return flat.reshape(n_frames, dimx, dimy).transpose(0, 2, 1)


def load_scalar(filename, expect_frames):
    """Real dimx*dimy field on its own stride, no leading frame."""
    n_frames = _frames_in_file(filename, 8, expect_frames, "real")
    flat = np.fromfile(filename, dtype=np.float64)
    return flat.reshape(n_frames, dimx, dimy).transpose(0, 2, 1)


def save_animation(animation, mp4_name, fps=15):
    print(f"Saving {mp4_name}...")
    try:
        animation.save(mp4_name, writer=FFMpegWriter(fps=fps))
    except Exception:
        gif_name = mp4_name.replace(".mp4", ".gif")
        print(f"ffmpeg not found, falling back to GIF: {gif_name}")
        animation.save(gif_name, writer=PillowWriter(fps=fps))
    print("Done.")


# ------------------------------------------------------------------ load
print("Loading psi and power...")
psi = load_psi(psi_filename)[1:]                    # drop initial condition
pow_all = load_scalar(power_filename, n_snap_pow)   # own, possibly finer stride

n_psi_frames = psi.shape[0]
n_pow_frames = pow_all.shape[0]
print(f"psi frames   : {n_psi_frames}  (dt_snap     = {dt_snap:.3f} tau_GL)")
print(f"power frames : {n_pow_frames}  (dt_snap_pow = {dt_snap_pow:.3f} tau_GL)")

# sign-fix guard: runs from before the Ex/Ey fix stored -|Jn|^2
finite = pow_all[np.isfinite(pow_all)]
if finite.size and np.sum(finite) < 0.0:
    print("WARNING: power.bin predominantly negative (pre Ex/Ey sign fix). Flipping.")
    pow_all = -pow_all

if mask_blockade:
    pow_plot = pow_all.copy()
    pow_plot[:, blockade_mask] = np.nan
else:
    pow_plot = pow_all

if mask_mag:
    psi_plot = psi.copy()
    psi_plot[:, blockade_mask] = np.nan
else:
    psi_plot = psi

# ------------------------------------------------------------------ time sync
# Master clock, then nearest-in-time partner frame for the other field.
if ANIM_BASE == "psi":
    master_n, master_dt = n_psi_frames, dt_snap
elif ANIM_BASE == "power":
    master_n, master_dt = n_pow_frames, dt_snap_pow
else:
    raise ValueError("ANIM_BASE must be 'psi' or 'power'.")

master_frames = list(range(0, master_n, frame_skip))
t_of = lambda k: (k + 1) * master_dt


def nearest(t, dt_field, n_field):
    """Index of the field frame whose time is closest to t."""
    k = int(round(t / dt_field)) - 1
    return int(np.clip(k, 0, n_field - 1))


psi_idx = [nearest(t_of(k), dt_snap, n_psi_frames) for k in master_frames]
pow_idx = [nearest(t_of(k), dt_snap_pow, n_pow_frames) for k in master_frames]

print(f"animating {len(master_frames)} frames on the '{ANIM_BASE}' clock, "
      f"t = {t_of(master_frames[0]):.2f} .. {t_of(master_frames[-1]):.2f} tau_GL")

# ------------------------------------------------------------------ colour limits
# Fixed for the whole movie so the scale does not flicker frame to frame.
mag_cmap = plt.cm.magma.copy()
mag_cmap.set_over("magenta")        # flag |psi|^2 > 1

clim_pct = 100.0
sample_frames = np.linspace(0, n_pow_frames - 1, min(200, n_pow_frames), dtype=int)
vals = pow_plot[sample_frames].ravel()
vals = vals[np.isfinite(vals)]

vmax = np.percentile(vals, clim_pct) if vals.size else 1.0
if not (np.isfinite(vmax) and vmax > 0.0):
    vmax = 1.0

if log_scale:
    pos = vals[vals > 0.0]
    vmin = np.percentile(pos, 1.0) if pos.size else 1e-12
    if not (np.isfinite(vmin) and 0.0 < vmin < vmax):
        vmin = max(vmax * 1e-8, np.finfo(float).tiny)
    pow_norm = LogNorm(vmin=vmin, vmax=vmax)
    print(f"power colour limits (log): {vmin:.6e} .. {vmax:.6e}")
else:
    pow_norm = Normalize(vmin=0.0, vmax=vmax)
    print(f"power colour limit: {vmax:.6e}")

# ================================================================== figure
fig, (ax_mag, ax_pow) = plt.subplots(1, 2, figsize=(13.0, 6.4))

TITLE_PAD = 13
CBAR_FRAC = 0.046      # colorbar thickness
CBAR_PAD = 0.035       # gap between panel and colorbar

f0_psi = psi_idx[0]
f0_pow = pow_idx[0]


def square(ax, show_ylabel=True):
    ax.set_box_aspect(1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"x $(\xi)$", labelpad=8)
    if show_ylabel:
        ax.set_ylabel(r"y $(\xi)$")
    ax.tick_params(labelsize=BASE, length=6, width=1.4)


# --- left: |psi|^2 ---
im_mag = ax_mag.imshow(
    np.abs(psi_plot[f0_psi]) ** 2, origin="lower", extent=extent,
    vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
)
ax_mag.set_title(r"Magnitude $|\psi|^2$", pad=TITLE_PAD)
square(ax_mag)
cb_mag = fig.colorbar(im_mag, ax=ax_mag, extend="max", orientation="vertical",
                      fraction=CBAR_FRAC, pad=CBAR_PAD)
cb_mag.set_label(r"$|\psi|^2$", fontsize=CBAR_LABEL)
cb_mag.ax.tick_params(labelsize=BASE)

# --- right: dissipation ---
im_pow = ax_pow.imshow(
    pow_plot[f0_pow], origin="lower", extent=extent,
    cmap="inferno", norm=pow_norm, interpolation="nearest",
)
ax_pow.set_title(r"Dissipation $p = |J_n|^2/\sigma_n$", pad=TITLE_PAD)
square(ax_pow, show_ylabel=False)
cb_pow = fig.colorbar(im_pow, ax=ax_pow, orientation="vertical",
                      fraction=CBAR_FRAC, pad=CBAR_PAD)
cb_pow.set_label(r"$p = J_n \cdot E$", fontsize=CBAR_LABEL)
cb_pow.ax.tick_params(labelsize=BASE)

title = fig.suptitle(rf"$t = {t_of(master_frames[0]):.1f}\,\tau_{{GL}}$",
                     fontsize=SUPTITLE, y=0.965)

fig.subplots_adjust(left=0.065, right=0.90, top=0.86, bottom=0.15, wspace=0.32)


def update_all(k):
    """k indexes into master_frames / psi_idx / pow_idx."""
    im_mag.set_data(np.abs(psi_plot[psi_idx[k]]) ** 2)
    im_pow.set_data(pow_plot[pow_idx[k]])
    title.set_text(rf"$t = {t_of(master_frames[k]):.1f}\,\tau_{{GL}}$")
    return im_mag, im_pow, title


ani = FuncAnimation(
    fig,
    update_all,
    frames=range(len(master_frames)),
    interval=80,
    blit=False,
)

save_animation(ani, video_name, fps=fps)
plt.close(fig)