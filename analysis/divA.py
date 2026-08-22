import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import os

# Parameters must keep in sync with Fortran
dimx = 90
dimy = 60
n = dimx*dimy
ymax = 30.0
n_t = 30000
tmax = 300.0
stride = 200
fps = 10
frame_skip = 1

# "bin"
# "compute"
divA_source = "bin"

# video name vars
sim_ext_mag = "00353"
sim_curr = "015"
H_ext = 0.0353

n_snapshots = n_t // stride

h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)

padx = (3*dimy - dimx) // 2
pady = dimy
dimA = 3*dimy

A_filename = "data/vec_pot.bin"
divA_filename = "data/div_A.bin"

video_name = f"divA_0gam_2.1kap_{n_t}t_{dimx}dimx_{dimy}dimy_{sim_ext_mag}mag_{sim_curr}curr.mp4"

# pad cells above and below
padx_phys = padx * h
padx_right_phys = (3*dimy - padx - dimx) * h
pady_phys = pady * h
extent = [-padx_phys, xmax + padx_right_phys, -pady_phys, ymax + pady_phys]

# rectangle marking the SC sub-block, for overlay
sc_rect_xy = (0.0, 0.0)
sc_rect_w = xmax
sc_rect_h = ymax

def load_divA_bin(filename):
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * dimA * dimA
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} reals, expected {expected_size} "
            f"(= n_snapshots * (3*dimy)^2). Check dimx, dimy, n_t, stride."
        )
    data = flat.reshape(n_snapshots, dimA, dimA).transpose(0, 2, 1)  # [frame, y, x]
    return data


def load_vecpot(filename):
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * 2 * dimA * dimA
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} reals, expected {expected_size}. "
            f"Expected Ax and Ay on the {dimA}x{dimA} grid for each frame."
        )
    flat = flat.reshape(n_snapshots, 2, dimA, dimA)
    Ax = flat[:, 0, :, :].transpose(0, 2, 1)
    Ay = flat[:, 1, :, :].transpose(0, 2, 1)
    return Ax, Ay


def compute_divA_from_A(Ax_big, Ay_big):
    n_fr = Ax_big.shape[0]
    divA = np.full((n_fr, dimA, dimA), np.nan)
    for f in range(n_fr):
        Ax = Ax_big[f]
        Ay = Ay_big[f]
        dAx_dx = (Ax[1:-1, 1:-1] - Ax[1:-1, 0:-2]) / h
        dAy_dy = (Ay[1:-1, 1:-1] - Ay[0:-2, 1:-1]) / h
        divA[f, 1:-1, 1:-1] = dAx_dx + dAy_dy
    return divA


print("Loading data...")

if divA_source == "bin" and os.path.exists(divA_filename):
    divA_all = load_divA_bin(divA_filename)
    print(f"div(A) loaded from {divA_filename} (Fortran).")

    divA_all[:, 0, :] = np.nan
    divA_all[:, -1, :] = np.nan
    divA_all[:, :, 0] = np.nan
    divA_all[:, :, -1] = np.nan
else:
    if divA_source == "bin":
        print(f"[warn] {divA_filename} not found; computing div(A) from "
              f"{A_filename} instead.")
    Ax_all, Ay_all = load_vecpot(A_filename)
    divA_all = compute_divA_from_A(Ax_all, Ay_all)

print(f"divA shape: {divA_all.shape}  (full {dimA}x{dimA} domain)")

divA_plot_data = divA_all

# save helper
def save_animation(animation, mp4_name, fps=15):
    print(f"Saving animation: {mp4_name}...")
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


divA_abs_max = robust_abs_limit(divA_plot_data, sample_frames, clim_pct)
divA_norm = TwoSlopeNorm(vmin=-divA_abs_max, vcenter=0.0, vmax=divA_abs_max)
print(f"divA color limit: +/- {divA_abs_max:.6e}")
print("Done.")

# animation
frame0 = 0
divA0 = divA_plot_data[frame0]

fig, ax = plt.subplots(figsize=(7.5, 6.5))

im_divA = ax.imshow(
    divA0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=divA_norm,
    interpolation="nearest",
)

# outline the SC inside the vacuum domain
sc_rect = plt.Rectangle(
    sc_rect_xy, sc_rect_w, sc_rect_h,
    fill=False, edgecolor="k", lw=1.2, ls="--",
)
ax.add_patch(sc_rect)
ax.text(0.0, ymax + 0.4 * pady_phys, "SC block", fontsize=9)

ax.set_title(r"$\nabla \cdot A$ over full vector-potential domain")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im_divA, ax=ax)

title = fig.suptitle("t = 0.000")


def update_all(frame):
    im_divA.set_data(divA_plot_data[frame])
    title.set_text(f"t = {(frame + 1) * dt_snap:.3f}")
    return im_divA, title


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