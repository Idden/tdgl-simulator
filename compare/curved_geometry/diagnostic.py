# Main diagnostic animation. Reads every field dump the solver writes and
# plays them side by side: |psi|^2, phase, curl(A), the J vectors, div(Js)
# and phi. This is the script to run first when checking a new run.

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm

# Parameters must keep in sync with Fortran
dimx = 121
dimy = 121
n = dimx*dimy
ymax = 30.0
n_t = 20000
tmax = 200.0
stride = 200
fps = 10
frame_skip = 1

include_external_field = True
mask_blockade = False

# Video name pieces
sim_ext_mag = "0"
sim_curr = "01"
H_ext = 0.0

n_snapshots = n_t // stride

h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt

xmax = h * (dimx - 1)
x = np.linspace(0, xmax, dimx)
y = np.linspace(0, ymax, dimy)
X, Y = np.meshgrid(x, y)
extent = [0, xmax, 0, ymax]

padx = (3*dimy - dimx) // 2
pady = dimy
dimA = 3*dimy

psi_filename = "data/psi.bin"
Js_filename = "data/Js.bin"
div_filename = "data/div_Js.bin"
phi_filename = "data/phi.bin"
A_filename = "data/vec_pot.bin"
Jn_filename = "data/Jn.bin"
Jtot_filename = "data/Jtot.bin"

video_name = f"rect_blockade_{sim_ext_mag}mag_{sim_curr}curr.mp4"

# Vector display options
current_vector_mode = "scaled" # "raw" | "scaled" | "normalized"
current_vector_scale = 7.0
eps = 1e-12
qskip = 2

# Mask blockade and compute mag flux
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
            half_w = lead_half - (lead_half - w0_half) / (1.0 + (x / R)**2)

            if abs(i - ci) > half_w:
                mask[i0, j0] = True

    return mask

blockade_mask = make_blockade_mask(dimx, dimy)

# Load data
def load_psi(filename):
    flat = np.fromfile(filename, dtype=np.complex128)

    expected_size = (n_snapshots + 1) * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} complex values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )

    # Fortran stores idx = i + (j-1)*dimy, so i (y) is the fast index.
    # Reshape gives [frame, x, y], the transpose flips it to [frame, y, x].
    data = flat.reshape(n_snapshots + 1, dimx, dimy).transpose(0, 2, 1)

    return data


def load_scalar(filename):
    flat = np.fromfile(filename, dtype=np.float64)

    expected_size = n_snapshots * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Check dimx, dimy, n_t, stride, or filename."
        )

    data = flat.reshape(n_snapshots, dimx, dimy).transpose(0, 2, 1)

    return data


def load_current(filename):
    flat = np.fromfile(filename, dtype=np.float64)

    expected_size = n_snapshots * 2 * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Jx and Jy for each saved frame."
        )

    # For each saved frame Fortran writes: write(20) Jx, Jy
    flat = flat.reshape(n_snapshots, 2, dimx, dimy)

    Jsx = flat[:, 0, :, :].transpose(0, 2, 1)
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

    Ax = flat[:, 0, :, :].transpose(0, 2, 1) # [frame, y, x] on big grid
    Ay = flat[:, 1, :, :].transpose(0, 2, 1)

    return Ax, Ay


def load_normal_current(filename):
    flat = np.fromfile(filename, dtype=np.float64)

    expected_size = n_snapshots * 2 * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Jnx and Jny for each saved frame."
        )

    flat = flat.reshape(n_snapshots, 2, dimx, dimy)

    Jnx = flat[:, 0, :, :].transpose(0, 2, 1)
    Jny = flat[:, 1, :, :].transpose(0, 2, 1)

    return Jnx, Jny


def load_total_current(filename):
    flat = np.fromfile(filename, dtype=np.float64)

    expected_size = n_snapshots * 2 * dimx * dimy

    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Jnx and Jny for each saved frame."
        )

    flat = flat.reshape(n_snapshots, 2, dimx, dimy)

    Jtotx = flat[:, 0, :, :].transpose(0, 2, 1)
    Jtoty = flat[:, 1, :, :].transpose(0, 2, 1)

    return Jtotx, Jtoty


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


print("Loading data...")

psi_all = load_psi(psi_filename)
Jsx_all, Jsy_all = load_current(Js_filename)
div_all = load_scalar(div_filename)
phi_all = load_scalar(phi_filename)
Ax_all, Ay_all = load_vecpot(A_filename)
Jnx_all, Jny_all = load_normal_current(Jn_filename)
Jtotx_all, Jtoty_all = load_total_current(Jtot_filename)
curlA_all = compute_curlA(Ax_all, Ay_all)

# Align psi with the J/div/phi frames.
# psi_all[0] is the initial condition, everything else starts at the first
# saved time-loop output, so drop it before pairing frames up.
psi = psi_all[1:]

print(f"psi shape:   {psi.shape}")
print(f"Js shape:    {Jsx_all.shape}")
print(f"Jn shape:    {Jnx_all.shape}")
print(f"Jtot shape:  {Jtotx_all.shape}")
print(f"div shape:   {div_all.shape}")
print(f"phi shape:   {phi_all.shape}")
print(f"curlA shape: {curlA_all.shape}")

if mask_blockade:
    psi_plot = psi.copy()
    Jx_plot_data = Jsx_all.copy()
    Jy_plot_data = Jsy_all.copy()
    Jnx_plot_data = Jnx_all.copy()
    Jny_plot_data = Jny_all.copy()
    Jtotx_plot_data = Jtotx_all.copy()
    Jtoty_plot_data = Jtoty_all.copy()
    div_plot_data = div_all.copy()
    phi_plot_data = phi_all.copy()
    curlA_plot_data = curlA_all.copy()

    psi_plot[:, blockade_mask] = np.nan
    Jx_plot_data[:, blockade_mask] = np.nan
    Jy_plot_data[:, blockade_mask] = np.nan
    Jnx_plot_data[:, blockade_mask] = np.nan
    Jny_plot_data[:, blockade_mask] = np.nan
    Jtotx_plot_data[:, blockade_mask] = np.nan
    Jtoty_plot_data[:, blockade_mask] = np.nan
    div_plot_data[:, blockade_mask] = np.nan
    phi_plot_data[:, blockade_mask] = np.nan
    curlA_plot_data[:, blockade_mask] = np.nan
else:
    psi_plot = psi
    Jx_plot_data = Jsx_all
    Jy_plot_data = Jsy_all
    Jnx_plot_data = Jnx_all
    Jny_plot_data = Jny_all
    Jtotx_plot_data = Jtotx_all
    Jtoty_plot_data = Jtoty_all
    div_plot_data = div_all
    phi_plot_data = phi_all
    curlA_plot_data = curlA_all

# Helper functions
def prepare_current_vectors(jx, jy, mode="scaled", scale=1.0, eps=1e-12):
    if mode == "raw":
        return jx.copy(), jy.copy()

    if mode == "scaled":
        return scale * jx, scale * jy

    if mode == "normalized":
        jmag = np.sqrt(jx**2 + jy**2)
        jx_norm = jx / (jmag + eps)
        jy_norm = jy / (jmag + eps)
        jx_norm[jmag < eps] = 0.0
        jy_norm[jmag < eps] = 0.0
        return jx_norm, jy_norm

    raise ValueError("current_vector_mode must be 'raw', 'scaled', or 'normalized'.")


def save_animation(animation, mp4_name, fps=15):
    print(f"Saving {mp4_name}...")

    try:
        animation.save(mp4_name, writer=FFMpegWriter(fps=fps))
    except Exception:
        gif_name = mp4_name.replace(".mp4", ".gif")
        print(f"ffmpeg not found, falling back to GIF: {gif_name}")
        animation.save(gif_name, writer=PillowWriter(fps=fps))

    print("Done.")


# Color limits, shared by every panel
print("Computing color limits...")

clim_pct = 100.0      # div, curl(A)
phi_clim_pct = 100.0  # phi

sample_frames = np.linspace(0, n_snapshots - 1, min(200, n_snapshots), dtype=int)
frames = list(range(0, n_snapshots, frame_skip))

psi_abs_max = 1.0
if not np.isfinite(psi_abs_max) or psi_abs_max == 0.0:
    psi_abs_max = 1.0


def robust_abs_limit(data, frame_idx, pct):
    vals = np.abs(np.asarray(data)[frame_idx]).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    lim = np.percentile(vals, pct)
    return lim if (np.isfinite(lim) and lim > 0.0) else 1.0


div_abs_max = robust_abs_limit(div_plot_data, sample_frames, clim_pct)
phi_abs_max = robust_abs_limit(phi_plot_data, sample_frames, phi_clim_pct)
curlA_abs_max = robust_abs_limit(curlA_plot_data, sample_frames, clim_pct)

div_norm = TwoSlopeNorm(vmin=-div_abs_max, vcenter=0.0, vmax=div_abs_max)
phi_norm = TwoSlopeNorm(vmin=-phi_abs_max, vcenter=0.0, vmax=phi_abs_max)
curlA_norm = TwoSlopeNorm(vmin=-curlA_abs_max, vcenter=0.0, vmax=curlA_abs_max)

print(f"|psi| max:        {psi_abs_max:.6e}")
print(f"div color limit:  +/- {div_abs_max:.6e}")
print(f"phi color limit:  +/- {phi_abs_max:.6e}")
print(f"curlA color lim:  +/- {curlA_abs_max:.6e}")
print("Done.")

# Animation: |psi|, phase, curl(A), J vectors, divJs, phi
frame0 = 0

mag0 = np.abs(psi_plot[frame0])**2
phase0 = np.angle(psi_plot[frame0])

jx0 = Jx_plot_data[frame0]
jy0 = Jy_plot_data[frame0]

div0 = div_plot_data[frame0]
phi0 = phi_plot_data[frame0]
curlA0 = curlA_plot_data[frame0]

jnx0 = Jnx_plot_data[frame0]
jny0 = Jny_plot_data[frame0]
jtotx0 = Jtotx_plot_data[frame0]
jtoty0 = Jtoty_plot_data[frame0]

jx_vec0, jy_vec0 = prepare_current_vectors(
    jx0,
    jy0,
    mode=current_vector_mode,
    scale=current_vector_scale,
    eps=eps,
)

jnx_vec0, jny_vec0 = prepare_current_vectors(
    jnx0,
    jny0,
    mode=current_vector_mode,
    scale=current_vector_scale,
    eps=eps,
)

jtotx_vec0, jtoty_vec0 = prepare_current_vectors(
    jtotx0,
    jtoty0,
    mode=current_vector_mode,
    scale=current_vector_scale,
    eps=eps,
)

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

ax_mag = axes[0, 0]
ax_phase = axes[0, 1]
ax_curl = axes[0, 2]
ax_div = axes[0, 3]
ax_vec = axes[1, 0]
ax_vec_n = axes[1, 1]
ax_vec_tot = axes[1, 2]
ax_phi = axes[1, 3]

mag_cmap = plt.cm.viridis.copy()
mag_cmap.set_over("magenta") # |psi|^2 > 1 shows as magenta
im_mag = ax_mag.imshow(
    mag0,
    origin="lower",
    extent=extent,
    vmin=0.0,
    vmax=psi_abs_max,
    cmap=mag_cmap,
    interpolation="nearest",
)

im_phase = ax_phase.imshow(
    phase0,
    origin="lower",
    extent=extent,
    cmap="hsv",
    vmin=-np.pi,
    vmax=np.pi,
    interpolation="nearest",
)

im_curl = ax_curl.imshow(
    curlA0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=curlA_norm,
    interpolation="nearest",
)

quiv = ax_vec.quiver(
    X[::qskip, ::qskip],
    Y[::qskip, ::qskip],
    jx_vec0[::qskip, ::qskip],
    jy_vec0[::qskip, ::qskip],
    angles="xy",
    scale_units="xy",
    scale=1.0,
)

quiv_n = ax_vec_n.quiver(
    X[::qskip, ::qskip],
    Y[::qskip, ::qskip],
    jnx_vec0[::qskip, ::qskip],
    jny_vec0[::qskip, ::qskip],
    angles="xy",
    scale_units="xy",
    scale=1.0,
)

quiv_tot = ax_vec_tot.quiver(
    X[::qskip, ::qskip],
    Y[::qskip, ::qskip],
    jtotx_vec0[::qskip, ::qskip],
    jtoty_vec0[::qskip, ::qskip],
    angles="xy",
    scale_units="xy",
    scale=1.0,
)

im_div = ax_div.imshow(
    div0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=div_norm,
    interpolation="nearest",
)

im_phi = ax_phi.imshow(
    phi0,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=phi_norm,
    interpolation="nearest",
)

ax_mag.set_title(r"Magnitude $|\psi|^2$")
ax_phase.set_title(r"Phase $\arg(\psi)$")
_curl_label = r"Total $B_z = (\nabla\times A)_z + H_{ext}$" if include_external_field \
    else r"Induced $B_z = (\nabla\times A)_z$"
ax_curl.set_title(_curl_label)
ax_vec.set_title(r"Fortran supercurrent vector field $J_s$")
ax_vec_n.set_title(r"Fortran normal current vector field $J_n$")
ax_vec_tot.set_title(r"Fortran total current vector field $J_{tot}$")
ax_div.set_title(r"Fortran $\nabla \cdot J_s$")
ax_phi.set_title(r"Electric potential $\varphi$")

for ax in axes.flat:
    ax.set_xlabel(r"x ($\xi$)")
    ax.set_ylabel(r"y ($\xi$)")

fig.colorbar(im_mag, ax=ax_mag, extend="max")
fig.colorbar(im_phase, ax=ax_phase)
fig.colorbar(im_curl, ax=ax_curl)
fig.colorbar(im_div, ax=ax_div)
fig.colorbar(im_phi, ax=ax_phi)

title = fig.suptitle("t = 0.000")


def update_all(frame):
    mag = np.abs(psi_plot[frame])**2
    phase = np.angle(psi_plot[frame])

    jx = Jx_plot_data[frame]
    jy = Jy_plot_data[frame]

    jnx = Jnx_plot_data[frame]
    jny = Jny_plot_data[frame]

    jtotx = Jtotx_plot_data[frame]
    jtoty = Jtoty_plot_data[frame]

    divJ = div_plot_data[frame]
    phi = phi_plot_data[frame]
    curlA = curlA_plot_data[frame]

    jx_vec, jy_vec = prepare_current_vectors(
        jx,
        jy,
        mode=current_vector_mode,
        scale=current_vector_scale,
        eps=eps,
    )

    jnx_vec, jny_vec = prepare_current_vectors(
        jnx,
        jny,
        mode=current_vector_mode,
        scale=current_vector_scale,
        eps=eps,
    )

    jtotx_vec, jtoty_vec = prepare_current_vectors(
        jtotx,
        jtoty,
        mode=current_vector_mode,
        scale=current_vector_scale,
        eps=eps,
    )

    im_mag.set_data(mag)
    im_phase.set_data(phase)
    im_curl.set_data(curlA)
    quiv.set_UVC(jx_vec[::qskip, ::qskip], jy_vec[::qskip, ::qskip])
    quiv_n.set_UVC(jnx_vec[::qskip, ::qskip], jny_vec[::qskip, ::qskip])
    quiv_tot.set_UVC(jtotx_vec[::qskip, ::qskip], jtoty_vec[::qskip, ::qskip])
    im_div.set_data(divJ)
    im_phi.set_data(phi)

    title.set_text(
        f"t = {(frame + 1) * dt_snap:.3f}    current vectors: {current_vector_mode}"
    )

    return im_mag, im_phase, im_curl, quiv, quiv_n, quiv_tot, im_div, im_phi, title


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