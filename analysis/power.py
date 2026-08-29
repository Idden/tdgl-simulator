import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize, LogNorm

dimx = 60
dimy = 60
n = dimx*dimy
ymax = 30.0
n_t = 100000
tmax = 1000.0
stride = 250
fps = 10
frame_skip = 1

mask_blockade = False
log_scale = True

# video name vars
sim_ext_mag = "00353"
sim_curr = "015"

n_snapshots = n_t // stride

h = ymax / (dimy - 1)
xmax = h * (dimx - 1)
dt = tmax / n_t
dt_snap = stride * dt

x = np.linspace(0, xmax, dimx)
y = np.linspace(0, ymax, dimy)
X, Y = np.meshgrid(x, y)
extent = [0, xmax, 0, ymax]

power_filename = "data/power.bin"

video_name = f"power_{n_t}t_{dimx}dimx_{dimy}dimy_{sim_ext_mag}mag_{sim_curr}curr.mp4"
total_png_name = f"power_total_{n_t}t_{dimx}dimx_{dimy}dimy_{sim_ext_mag}mag_{sim_curr}curr.png"

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

blockade_mask = make_blockade_mask(dimx, dimy)

# load data
def load_scalar(filename):
    flat = np.fromfile(filename, dtype=np.float64)

    expected_size = n_snapshots * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename. "
            f"(Is unit 100 / power.bin uncommented and the sim rerun?)"
        )
    
    data = flat.reshape(n_snapshots, dimx, dimy).transpose(0, 2, 1)

    return data


print("Loading data...")

pow_all = load_scalar(power_filename)

# Dissipation p = Jn.E = |Jn|^2 / sigma_n must be >= 0.
# Data from before the Ex/Ey sign fix in tdgl.F90 comes out as -|Jn|^2:
# detect that and flip so old runs still plot correctly.
finite = pow_all[np.isfinite(pow_all)]
if finite.size and np.sum(finite) < 0.0:
    print("WARNING: power.bin is predominantly negative -- this looks like data "
          "from before the Ex/Ey sign fix in tdgl.F90. Flipping sign. "
          "Rerun the sim with the fixed code to get rid of this warning.")
    pow_all = -pow_all

print(f"power shape: {pow_all.shape}")

if mask_blockade:
    pow_plot_data = pow_all.copy()
    pow_plot_data[:, blockade_mask] = np.nan
else:
    pow_plot_data = pow_all

# helper functions
def save_animation(animation, mp4_name, fps=15):
    print(f"Saving {mp4_name}...")

    try:
        animation.save(mp4_name, writer=FFMpegWriter(fps=fps))
    except Exception:
        gif_name = mp4_name.replace(".mp4", ".gif")
        print(f"ffmpeg not found, falling back to GIF: {gif_name}")
        animation.save(gif_name, writer=PillowWriter(fps=fps))

    print("Done.")

# total dissipated power vs time
P_tot = np.nansum(np.where(blockade_mask, np.nan, pow_all), axis=(1, 2)) * h**2
t_axis = (np.arange(n_snapshots) + 1) * dt_snap

inside = pow_all[:, ~blockade_mask]
print("inside SC:  min =", np.nanmin(inside),
      "max =", np.nanmax(inside),
      "mean =", np.nanmean(inside))
print("P_tot range:", np.nanmin(P_tot) if 'P_tot' in dir() else "n/a",
      "to", np.nanmax(P_tot) if 'P_tot' in dir() else "n/a")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_axis, P_tot, lw=1.2)
ax.set_xlabel(r"t $(\tau_{GL})$")
ax.set_ylabel(r"$P(t) = \int J_n \cdot E \; dA$")
ax.set_title("Total Dissipated Power")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(total_png_name, dpi=150)
plt.close(fig)
print(f"Saved {total_png_name}")

# animation
print("Computing color limits...")
clim_pct = 100.0

sample_frames = np.linspace(0, n_snapshots - 1, min(200, n_snapshots), dtype=int)
frames = list(range(0, n_snapshots, frame_skip))

vals = pow_plot_data[sample_frames].ravel()
vals = vals[np.isfinite(vals)]
vmax = np.percentile(vals, clim_pct) if vals.size else 1.0
if not (np.isfinite(vmax) and vmax > 0.0):
    vmax = 1.0

if log_scale:
    pos = vals[vals > 0.0]
    vmin = np.percentile(pos, 1.0) if pos.size else 1e-12
    pow_norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    pow_norm = Normalize(vmin=0.0, vmax=vmax)

print(f"power color limit: {vmax:.6e}")
print("Done.")

frame0 = 0
pow0 = pow_plot_data[frame0]

fig, ax = plt.subplots(figsize=(9, 6))

im_pow = ax.imshow(
    pow0,
    origin="lower",
    extent=extent,
    cmap="inferno",
    norm=pow_norm,
    interpolation="nearest",
)

ax.set_title(r"Dissipation $p = J_n \cdot E = |J_n|^2 / \sigma_n$")
ax.set_xlabel(r"x $(\xi)$")
ax.set_ylabel(r"y $(\xi)$")

fig.colorbar(im_pow, ax=ax)

title = fig.suptitle("t = 0.000")


def update_all(frame):
    im_pow.set_data(pow_plot_data[frame])
    title.set_text(f"t = {(frame + 1) * dt_snap:.3f}")
    return im_pow, title


ani = FuncAnimation(
    fig,
    update_all,
    frames=frames,
    interval=80,
    blit=False,
)

plt.tight_layout()
save_animation(ani, video_name, fps=fps)
plt.close(fig)