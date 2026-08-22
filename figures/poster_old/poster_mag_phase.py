import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# (keep in sync with the Fortran run, same as gTDGL_plotting.py)
dimx = 150
dimy = 150
ymax = 30.0
n_t = 600000
tmax = 1000.0
stride = 3000

# which snapshot to show. 0 = first saved frame. Pick the frame that shows the
# vortices you want on the poster.
FRAME = 150

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_mag_phase_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.pdf"

# ------------------------------------------------------------------ derived
n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

psi_filename = "data/psi.bin"

# ------------------------------------------------------------------ load psi
# (identical to load_psi in gTDGL_plotting.py)
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)
    expected_size = (n_snapshots + 1) * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )
    # Fortran fast index is i (y): reshape [frame, x, y] then transpose -> [frame, y, x]
    data = flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)
    return data


print("Loading psi...")
psi_all = load_psi(psi_filename)
psi = psi_all[1:]  # drop the initial condition, matches the plotting script

if FRAME < 0 or FRAME >= psi.shape[0]:
    raise IndexError(
        f"FRAME={FRAME} out of range 0..{psi.shape[0]-1}. "
        f"(t would be {(FRAME+1)*dt_snap:.3f})"
    )

mag = np.abs(psi[FRAME]) ** 2
phase = np.angle(psi[FRAME])
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# ------------------------------------------------------------------ colormaps
# mako / twilight. mako ships with seaborn; fall back to a dark matplotlib map
# if seaborn is not installed so the script still runs.
mag_cmap = plt.cm.magma
mag_cmap = mag_cmap.copy()
mag_cmap.set_over("magenta")  # |psi|^2 > 1 flagged, same idea as your viridis version

phase_cmap = "twilight"  # cyclic, dark

# ------------------------------------------------------------------ figure
# Two rows, one column: magnitude on top, phase on bottom.
# figsize tuned so each wide (2:1) panel stacks nicely in a poster column.
fig, (ax_mag, ax_phase) = plt.subplots(
    2, 1, figsize=(6.0, 6.4), constrained_layout=True
)

im_mag = ax_mag.imshow(
    mag, origin="lower", extent=extent,
    vmin=0.0, vmax=1.0, cmap=mag_cmap, interpolation="nearest",
)
ax_mag.set_title(r"Magnitude $|\psi|^2$", fontsize=15)

im_phase = ax_phase.imshow(
    phase, origin="lower", extent=extent,
    cmap=phase_cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest",
)
ax_phase.set_title(r"Phase $\arg(\psi)$", fontsize=15)

for ax in (ax_mag, ax_phase):
    ax.set_xlabel(r"x $(\xi)$", fontsize=12)
    ax.set_ylabel(r"y $(\xi)$", fontsize=12)
    ax.tick_params(labelsize=10)

# own colorbar per panel. shrink < 1 makes the bar shorter.
cb_mag = fig.colorbar(im_mag, ax=ax_mag, extend="max",
                      fraction=0.046, pad=0.02, shrink=0.7)
cb_mag.set_label(r"$|\psi|^2$", fontsize=12)

cb_ph = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.02, shrink=0.7,
                     ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb_ph.set_label(r"$\arg(\psi)$", fontsize=12)
cb_ph.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")