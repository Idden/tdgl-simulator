import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm

# Parameters must keep in sync with Fortran
dimx = 90
dimy = 60
n = dimx*dimy
ymax = 30
n_t = 30000
tmax = 300.0
stride = 200
fps = 10
frame_skip = 1
include_external_field = True

# "staggered"
# "centered"
curl_stencil = "staggered"

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

label = "totalBz" if include_external_field else "inducedBz"
video_name = (
    f"curlA_full_{label}_1gam_2.1kap_{n_t}t_{dimx}dimx_{dimy}dimy_"
    f"{sim_ext_mag}mag_{sim_curr}curr.mp4"
)

padx_phys = padx * h
padx_right_phys = (3*dimy - padx - dimx) * h
pady_phys = pady * h
extent = [-padx_phys, xmax + padx_right_phys, -pady_phys, ymax + pady_phys]

# rectangle marking the SC sub-block, for overlay
sc_rect_xy = (0.0, 0.0)
sc_rect_w = xmax
sc_rect_h = ymax

# Load data
def load_vecpot(filename):
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * 2 * dimA * dimA
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} reals, expected {expected_size}. "
            f"Expected Ax and Ay on the {dimA}x{dimA} grid for each frame."
        )
    flat = flat.reshape(n_snapshots, 2, dimA, dimA)
    Ax = flat[:, 0, :, :].transpose(0, 2, 1) # [frame, y, x] on big grid
    Ay = flat[:, 1, :, :].transpose(0, 2, 1)
    return Ax, Ay


def compute_curlA_full(Ax_big, Ay_big):
    n_fr = Ax_big.shape[0]
    Bz = np.full((n_fr, dimA, dimA), np.nan)

    for f in range(n_fr):
        Ax = Ax_big[f]
        Ay = Ay_big[f]
        if curl_stencil == "staggered":
            dAy_dx = (Ay[1:-1, 1:-1] - Ay[1:-1, 0:-2]) / h
            dAx_dy = (Ax[1:-1, 1:-1] - Ax[0:-2, 1:-1]) / h
            Bz[f, 1:-1, 1:-1] = dAy_dx - dAx_dy
        elif curl_stencil == "centered":
            dAy_dx = np.gradient(Ay, h, axis=1)
            dAx_dy = np.gradient(Ax, h, axis=0)
            Bz[f] = dAy_dx - dAx_dy
        else:
            raise ValueError("curl_stencil must be 'staggered' or 'centered'.")

    if include_external_field:
        Bz = Bz + H_ext

    return Bz


print("Loading data...")
Ax_all, Ay_all = load_vecpot(A_filename)
curlA_all = compute_curlA_full(Ax_all, Ay_all)

# curlA_all[:, 0, :] = np.nan
# curlA_all[:, -1, :] = np.nan
# curlA_all[:, :, 0] = np.nan
# curlA_all[:, :, -1] = np.nan

print(f"curlA shape: {curlA_all.shape}  (full {dimA}x{dimA} domain)")

curlA_plot_data = curlA_all

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

# Color limits
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

curlA_abs_max = robust_abs_limit(curlA_plot_data, sample_frames, clim_pct)
if include_external_field:
    # total field is offset by H_ext; keep 0 centered but widen to include it
    vmax = max(curlA_abs_max, abs(H_ext) + curlA_abs_max)
    curlA_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
else:
    curlA_norm = TwoSlopeNorm(vmin=-curlA_abs_max, vcenter=0.0, vmax=curlA_abs_max)
print(f"curlA color limit: +/- {curlA_abs_max:.6e}")
print("Done.")

# animation
frame0 = 0
curlA0 = curlA_plot_data[frame0]

fig, ax = plt.subplots(figsize=(7.5, 6.5))

im_curlA = ax.imshow(
    curlA0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=curlA_norm,
    interpolation="nearest",
)

# outline the SC inside the vacuum domain
sc_rect = plt.Rectangle(
    sc_rect_xy, sc_rect_w, sc_rect_h,
    fill=False, edgecolor="k", lw=1.2, ls="--",
)
ax.add_patch(sc_rect)
ax.text(0.0, ymax + 0.4 * pady_phys, "SC block", fontsize=9)

_title = r"Total $B_z = (\nabla\times a)_z + H_{ext}$" if include_external_field \
    else r"Induced $B_z = (\nabla\times a)_z$ over full domain"
ax.set_title(_title)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im_curlA, ax=ax)

title = fig.suptitle("t = 0.000")


def update_all(frame):
    im_curlA.set_data(curlA_plot_data[frame])
    title.set_text(f"t = {(frame + 1) * dt_snap:.3f}")
    return im_curlA, title


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