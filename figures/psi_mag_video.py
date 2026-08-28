import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Parameters must keep in sync with Fortran
dimx = 60
dimy = 60
n = dimx * dimy
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
dt = tmax / n_t
dt_snap = stride * dt

xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

psi_filename = "data/psi.bin"

video_name = f"mag_{sim_ext_mag}mag_{sim_curr}curr.mp4"


# Mask blockade (only used if mask_blockade = True)
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

            x = j - cj
            half_w = lead_half - (lead_half - w0_half) / (1.0 + (x / R) ** 2)

            if abs(i - ci) > half_w:
                mask[i0, j0] = True

    return mask


blockade_mask = make_blockade_mask(dimx, dimy)


# Load psi (identical convention to gTDGL_plotting.py)
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)

    expected_size = (n_snapshots + 1) * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )

    # Fortran stores idx = i + (j-1)*dimy, so i (y) is the fast index:
    # reshape gives [frame, x, y], transpose flips to [frame, y, x]
    data = flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)

    return data


def save_animation(animation, mp4_name, fps=15):
    print(f"Saving {mp4_name}...")

    try:
        animation.save(mp4_name, writer=FFMpegWriter(fps=fps))
    except Exception:
        gif_name = mp4_name.replace(".mp4", ".gif")
        print(f"ffmpeg not found, falling back to GIF: {gif_name}")
        animation.save(gif_name, writer=PillowWriter(fps=fps))

    print("Done.")


print("Loading data...")
psi_all = load_psi(psi_filename)

# psi_all[0] is the initial condition; drop it to align with the saved time-loop frames
psi = psi_all[1:]
print(f"psi shape: {psi.shape}")

if mask_blockade:
    psi_plot = psi.copy()
    psi_plot[:, blockade_mask] = np.nan
else:
    psi_plot = psi

psi_abs_max = 1.0

frames = list(range(0, n_snapshots, frame_skip))

# ---------------------------------------------------------------- figure
frame0 = 0
mag0 = np.abs(psi_plot[frame0]) ** 2

fig, ax_mag = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)

mag_cmap = plt.cm.viridis.copy()
mag_cmap.set_over("magenta")   # |psi|^2 > 1 shows as magenta

im_mag = ax_mag.imshow(
    mag0,
    origin="lower",
    extent=extent,
    vmin=0.0,
    vmax=psi_abs_max,
    cmap=mag_cmap,
    interpolation="nearest",
)

ax_mag.set_aspect("auto")
ax_mag.set_title(r"Magnitude $|\psi|^2$")
ax_mag.set_xlabel(r"x ($\xi$)")
ax_mag.set_ylabel(r"y ($\xi$)")

cb = fig.colorbar(im_mag, ax=ax_mag, extend="max")
cb.set_label(r"$|\psi|^2$")

title = fig.suptitle("t = 0.000")


def update_mag(frame):
    mag = np.abs(psi_plot[frame]) ** 2
    im_mag.set_data(mag)
    title.set_text(f"t = {(frame + 1) * dt_snap:.3f}")
    return im_mag, title


ani = FuncAnimation(
    fig,
    update_mag,
    frames=frames,
    interval=80,
    blit=False,
)

save_animation(ani, video_name, fps=fps)
plt.close(fig)