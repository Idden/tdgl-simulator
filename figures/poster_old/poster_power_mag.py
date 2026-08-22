import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# parameters
dimx = 150
dimy = 150
ymax = 30.0
n_t = 600000
tmax = 1000.0
stride = 3000

# Which snapshot is the annihilation moment. Set this to the frame where the
# two vortices merge (look at your movie, read off t, convert: FRAME = t/dt_snap - 1).
FRAME = 150

sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_power_mag_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.pdf"

# ------------------------------------------------------------------ derived
n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

psi_filename = "data/psi.bin"
power_filename = "data/power.bin"

# ------------------------------------------------------------------ blockade mask
# (from power.py -- used only to exclude the vacuum blockade from the P(t) sum)
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
            half_w = lead_half - (lead_half - w0_half) / (1.0 + (xx / R) ** 2)
            if abs(i - ci) > half_w:
                mask[i0, j0] = True
    return mask

blockade_mask = make_blockade_mask(dimx, dimy)

# ------------------------------------------------------------------ loaders
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)
    expected_size = (n_snapshots + 1) * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}."
        )
    data = flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)
    return data


def load_scalar(filename):
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}."
        )
    data = flat.reshape(n_snapshots, dimx, dimy).transpose(0, 2, 1)
    return data


print("Loading psi and power...")
psi = load_psi(psi_filename)[1:]        # drop initial condition
pow_all = load_scalar(power_filename)

# Same sign-fix guard as power.py: old runs stored -|Jn|^2.
finite = pow_all[np.isfinite(pow_all)]
if finite.size and np.sum(finite) < 0.0:
    print("WARNING: power.bin predominantly negative (pre Ex/Ey sign fix). Flipping.")
    pow_all = -pow_all

if FRAME < 0 or FRAME >= psi.shape[0]:
    raise IndexError(f"FRAME={FRAME} out of range 0..{psi.shape[0]-1}.")

mag = np.abs(psi[FRAME]) ** 2
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# ------------------------------------------------------------------ P(t) curve
# total dissipated power vs time (from power.py): integrate p over SC only
P_tot = np.nansum(np.where(blockade_mask, np.nan, pow_all), axis=(1, 2)) * h**2
t_axis = (np.arange(n_snapshots) + 1) * dt_snap

# ------------------------------------------------------------------ colormap
mag_cmap = plt.cm.magma
mag_cmap = mag_cmap.copy()
mag_cmap.set_over("magenta")

# ------------------------------------------------------------------ figure (up/down)
fig, (ax_mag, ax_pow) = plt.subplots(
    2, 1, figsize=(6.0, 6.4),
    gridspec_kw={"height_ratios": [1.35, 1.0]},
    constrained_layout=True,
)

# --- top: magnitude snapshot ---
im_mag = ax_mag.imshow(
    mag, origin="lower", extent=extent,
    vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
)
ax_mag.set_title(rf"Vortex Annihilation: $|\psi|^2$ at $t={t_here:.1f}\,\tau_{{GL}}$",
                 fontsize=14)
ax_mag.set_xlabel(r"x $(\xi)$", fontsize=12)
ax_mag.set_ylabel(r"y $(\xi)$", fontsize=12)
cb = fig.colorbar(im_mag, ax=ax_mag, extend="max",
                  fraction=0.046, pad=0.02, shrink=0.7)
cb.set_label(r"$|\psi|^2$", fontsize=12)

# --- bottom: P(t) with marker at the snapshot time ---
# Add an invisible colorbar of the SAME width to the P(t) panel so both panels
# line up and the curve is centered (no blank gap on the right).
cb_spacer = fig.colorbar(im_mag, ax=ax_pow, fraction=0.046, pad=0.02)
cb_spacer.ax.set_visible(False)

ax_pow.plot(t_axis, P_tot, lw=1.6, color="#3b0f70")   # dark magma-purple line
ax_pow.axvline(t_here, color="#b73779", lw=1.6, ls="--",
               label=f"snapshot  t = {t_here:.1f}")
ax_pow.set_xlabel(r"t $(\tau_{GL})$", fontsize=12)
ax_pow.set_ylabel(r"$P(t) = \int J_n \cdot E\; dA$", fontsize=12)
ax_pow.set_title("Total Dissipated Power", fontsize=14)
ax_pow.grid(alpha=0.3)
ax_pow.legend(fontsize=10, frameon=False)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")