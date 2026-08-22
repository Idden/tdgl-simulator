import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Fortran parameters
dimx = 150
dimy = 150
ymax = 30.0
n_t = 600000
tmax = 1000.0
stride = 3000
stride_power = 300

# --- annihilation timing --------------------------------------------------
# FRAME_ANNIH is a *psi* frame index (i.e. counted in units of `stride`).
FRAME_ANNIH = 61
FRAME_STEP = 1
FRAME_BEFORE = FRAME_ANNIH - FRAME_STEP
FRAME_AFTER = FRAME_ANNIH + FRAME_STEP

# --- P(t) cosmetics -------------------------------------------------------
P_YLIM = None      # None -> autoscale with headroom; or set e.g. (0.0, 1.2)
P_SMOOTH = 0       # 0 = raw. >1 = boxcar smoothing over this many power frames.

# --- global text sizing ---------------------------------------------------
BASE = 22          # ticks, colorbar numbers
TITLE_SIZE = 25    # panel titles
LABEL_SIZE = 22    # axis labels
CBAR_LABEL = 22    # colorbar labels

plt.rcParams.update({
    "font.size":        BASE,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  BASE,
    "ytick.labelsize":  BASE,
})

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_dissipation_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME_ANNIH}.pdf"

# data files
psi_filename = "data/psi.bin"
power_filename = "data/power.bin"

# ------------------------------------------------------------------ derived
n_snap_psi = n_t // stride              # 200
n_snap_pow = n_t // stride_power        # 2000
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt                   # psi frame spacing        (5.0 tau_GL)
dt_snap_pow = stride_power * dt         # power frame spacing      (0.5 tau_GL)
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

# ------------------------------------------------------------------ blockade mask
# excludes the vacuum blockade from the P(t) sum
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
    data = flat.reshape(n_frames, dimx, dimy).transpose(0, 2, 1)
    return data


def load_scalar(filename, expect_frames):
    """Real n=dimx*dimy field dumped once per its own stride, no leading frame."""
    n_frames = _frames_in_file(filename, 8, expect_frames, "real")
    flat = np.fromfile(filename, dtype=np.float64)
    data = flat.reshape(n_frames, dimx, dimy).transpose(0, 2, 1)
    return data


# ------------------------------------------------------------------ load
print("Loading psi and power...")
psi = load_psi(psi_filename)[1:]                       # drop initial condition
pow_all = load_scalar(power_filename, n_snap_pow)      # own, finer stride

n_psi_frames = psi.shape[0]
n_pow_frames = pow_all.shape[0]
print(f"psi frames   : {n_psi_frames}  (dt_snap     = {dt_snap:.3f} tau_GL)")
print(f"power frames : {n_pow_frames}  (dt_snap_pow = {dt_snap_pow:.3f} tau_GL)")

# sign-fix guard: old runs stored -|Jn|^2
finite = pow_all[np.isfinite(pow_all)]
if finite.size and np.sum(finite) < 0.0:
    print("WARNING: power.bin predominantly negative (pre Ex/Ey sign fix). Flipping.")
    pow_all = -pow_all

# frame bounds check -- these index psi only
for nm, fr in (("BEFORE", FRAME_BEFORE), ("ANNIH", FRAME_ANNIH), ("AFTER", FRAME_AFTER)):
    if fr < 0 or fr >= n_psi_frames:
        raise IndexError(f"{nm} psi frame {fr} out of range 0..{n_psi_frames-1}.")

mag_before = np.abs(psi[FRAME_BEFORE]) ** 2
mag_after = np.abs(psi[FRAME_AFTER]) ** 2

t_before = (FRAME_BEFORE + 1) * dt_snap
t_annih = (FRAME_ANNIH + 1) * dt_snap
t_after = (FRAME_AFTER + 1) * dt_snap
print(f"before f{FRAME_BEFORE} t={t_before:.1f} | annih f{FRAME_ANNIH} t={t_annih:.1f} "
      f"| after f{FRAME_AFTER} t={t_after:.1f}")

# ------------------------------------------------------------------ P(t) curve
# total dissipated power vs time: integrate p over SC only (blockade excluded),
# on the FINE power time grid.
P_tot = np.nansum(np.where(blockade_mask[None, :, :], np.nan, pow_all), axis=(1, 2)) * h ** 2
t_axis = (np.arange(n_pow_frames) + 1) * dt_snap_pow

if P_SMOOTH and P_SMOOTH > 1:
    k = int(P_SMOOTH)
    kern = np.ones(k) / k
    P_plot = np.convolve(P_tot, kern, mode="same")
else:
    P_plot = P_tot

t_end = n_pow_frames * dt_snap_pow

# ------------------------------------------------------------------ colormap
mag_cmap = plt.cm.magma.copy()
mag_cmap.set_over("magenta")

# ================================================================== figure
# Top row: two square snapshots (before | after) sharing one vertical colorbar
# on the far right. Bottom row: P(t) spanning the full width.
fig = plt.figure(figsize=(11.0, 9.5))
gs = GridSpec(2, 2, height_ratios=[1.45, 0.9], figure=fig)

ax_before = fig.add_subplot(gs[0, 0])
ax_after = fig.add_subplot(gs[0, 1])
ax_power = fig.add_subplot(gs[1, :])   # span both columns

TITLE_PAD = 13


def square_snap(ax, data, title, show_ylabel=True):
    im = ax.imshow(
        data, origin="lower", extent=extent,
        vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
    )
    ax.set_box_aspect(1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_title(title, pad=TITLE_PAD)
    ax.set_xlabel(r"x $(\xi)$", labelpad=8)
    if show_ylabel:
        ax.set_ylabel(r"y $(\xi)$")
    ax.tick_params(labelsize=BASE, length=6, width=1.4)
    return im


im_before = square_snap(ax_before, mag_before,
                        rf"Before ($t = {t_before:.0f}\,\tau_{{GL}}$)")

im_after = square_snap(ax_after, mag_after,
                       rf"After ($t = {t_after:.0f}\,\tau_{{GL}}$)",
                       show_ylabel=False)

# --- total dissipated power P(t), full width ---
ax_power.plot(t_axis, P_plot, lw=1.6, color="#3b0f70")
# bracket the annihilation: faint before/after markers + bold annihilation line
ax_power.axvline(t_before, color="#8c8c8c", lw=1.6, ls=":", alpha=0.8)
ax_power.axvline(t_after, color="#8c8c8c", lw=1.6, ls=":", alpha=0.8)
ax_power.axvline(t_annih, color="#b73779", lw=2.6, ls="--",
                 label=rf"$t = {t_annih:.0f}\,\tau_{{GL}}$")
ax_power.set_xlabel(r"$t\ (\tau_{GL})$")
ax_power.set_ylabel(r"$P(t) = \int J_n \cdot E\; dA$")
ax_power.set_xlim(0, t_end)
if P_YLIM is None:
    pmax = np.nanmax(P_plot)
    ax_power.set_ylim(0, 1.08 * pmax if np.isfinite(pmax) and pmax > 0 else 1.0)
else:
    ax_power.set_ylim(*P_YLIM)
ax_power.set_title("Total Dissipated Power", pad=TITLE_PAD)
ax_power.grid(alpha=0.3)
ax_power.legend(fontsize=20, frameon=False, loc="upper left")
ax_power.margins(x=0.01)

# lay out the main axes (left margin wide for the P(t) y-label,
# right margin leaves room for the vertical shared colorbar)
fig.subplots_adjust(left=0.1, right=0.87, top=0.94, bottom=0.1,
                    wspace=0.25, hspace=0.42)

# one shared vertical colorbar to the far right, spanning the snapshot row
p_after = ax_after.get_position()
cax = fig.add_axes([p_after.x1 + 0.015, p_after.y0, 0.018, p_after.height])
cb = fig.colorbar(im_before, cax=cax, extend="max", orientation="vertical")
cb.set_label(r"$|\psi|^2$", fontsize=CBAR_LABEL)
cb.ax.tick_params(labelsize=BASE)

t_e = 50.0
y_e = np.interp(t_e, t_axis, P_plot)
ax_power.annotate("Mag",
                  xy=(t_e, y_e),
                  xytext=(t_e, y_e + 0.2),
                  ha="center", va="center",
                  color="#b73779", fontsize=18,
                  arrowprops=dict(arrowstyle="->", color="#b73779", lw=2,
                                  connectionstyle="arc3,rad=0.25"))

t_e = 100.0
y_e = np.interp(t_e, t_axis, P_plot)
ax_power.annotate("Curr",
                  xy=(t_e, y_e),
                  xytext=(t_e, y_e + 0.25),
                  ha="center", va="center",
                  color="#b73779", fontsize=18,
                  arrowprops=dict(arrowstyle="->", color="#b73779", lw=2,
                                  connectionstyle="arc3,rad=0.25"))

t_e = 160.0
y_e = np.interp(t_e, t_axis, P_plot)
ax_power.annotate("1st Nucleation",
                  xy=(t_e, y_e),
                  xytext=(t_e, y_e + 0.35),
                  ha="center", va="center",
                  color="#b73779", fontsize=18,
                  arrowprops=dict(arrowstyle="->", color="#b73779", lw=2,
                                  connectionstyle="arc3,rad=0.25"))

CIRC_XY = (15.0, 10.8)   # (x, y) center of the annihilating pair
CIRC_R  = 4.5            # radius in xi

def mark_region(ax, xy=CIRC_XY, r=CIRC_R, color="lime"):
    ax.add_patch(Circle(
        xy, r,
        fill=False,
        edgecolor=color,
        linestyle=(0, (6, 4)),   # dash pattern: 6 on, 4 off
        linewidth=2.5,
        zorder=5,
    ))

im_before = square_snap(ax_before, mag_before,
                        rf"Before ($t = {t_before:.0f}\,\tau_{{GL}}$)")
mark_region(ax_before)

im_after = square_snap(ax_after, mag_after,
                       rf"After ($t = {t_after:.0f}\,\tau_{{GL}}$)",
                       show_ylabel=False)
mark_region(ax_after)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")