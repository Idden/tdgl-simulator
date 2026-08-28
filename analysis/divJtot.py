import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm

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

div_Jtot_filename = "data/div_Jtot.bin"

video_name = f"divJtot_{n_t}t_{dimx}dimx_{dimy}dimy_{sim_ext_mag}mag_{sim_curr}curr.mp4"

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
            f"(Is unit 90 / div_Jtot.bin uncommented and the sim rerun?)"
        )
    
    data = flat.reshape(n_snapshots, dimx, dimy).transpose(0, 2, 1)

    return data


print("Loading data...")

div_Jtot_all = load_scalar(div_Jtot_filename)

print(f"div_Jtot shape: {div_Jtot_all.shape}")

if mask_blockade:
    div_plot_data = div_Jtot_all.copy()
    div_plot_data[:, blockade_mask] = np.nan
else:
    div_plot_data = div_Jtot_all

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

# color limits
print("Computing color limits...")
clim_pct = 100.0

sample_frames = np.linspace(0, n_snapshots - 1, min(200, n_snapshots), dtype=int)
frames = list(range(0, n_snapshots, frame_skip))

def robust_abs_limit(data, frame_idx, pct):
    vals = np.abs(np.asarray(data)[frame_idx]).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    lim = np.percentile(vals, pct)
    return lim if (np.isfinite(lim) and lim > 0.0) else 1.0

div_abs_max = robust_abs_limit(div_plot_data, sample_frames, clim_pct)
div_norm = TwoSlopeNorm(vmin=-div_abs_max, vcenter=0.0, vmax=div_abs_max)

print(f"div(Jtot) color limit: +/- {div_abs_max:.6e}")
print("Done.")

# animation
frame0 = 0
div0 = div_plot_data[frame0]

fig, ax = plt.subplots(figsize=(9, 6))

im_div = ax.imshow(
    div0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=div_norm,
    interpolation="nearest",
)

ax.set_title(r"Fortran $\nabla \cdot J_{tot}$")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.colorbar(im_div, ax=ax)

title = fig.suptitle("t = 0.000")


def update_all(frame):
    im_div.set_data(div_plot_data[frame])
    title.set_text(f"t = {(frame + 1) * dt_snap:.3f}")
    return im_div, title


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