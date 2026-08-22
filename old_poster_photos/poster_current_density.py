import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# keep in sync with the Fortran run
dimx = 200
dimy = 200
ymax = 30.0
n_t = 50000
tmax = 500.0
stride = 250
mask_blockade = False

FRAME = 90
EDGE_OFFSET = 3

# output
sim_ext_mag = "00353"
sim_curr = "015"
out_name = f"poster_current_density_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.pdf"

n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

Js_filename = "data_curr_dens/Js.bin"

def load_current(filename):
    # supercurrent Jsx, Jsy on the dimx*dimy grid (from gTDGL_plotting.py)
    flat = np.fromfile(filename, dtype=np.float64)
    expected_size = n_snapshots * 2 * dimx * dimy
    if flat.size != expected_size:
        raise ValueError(
            f"{filename} has {flat.size} real values, expected {expected_size}. "
            f"Expected Jx and Jy for each saved frame."
        )
    flat = flat.reshape(n_snapshots, 2, dimx, dimy)
    Jsx = flat[:, 0, :, :].transpose(0, 2, 1)   # [frame, y, x]
    Jsy = flat[:, 1, :, :].transpose(0, 2, 1)
    return Jsx, Jsy


def make_blockade_mask(dimx, dimy):
    mask = np.zeros((dimy, dimx), dtype=bool)   # rows = y (i), cols = x (j)

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


def top_sc_row_per_column(blockade_mask, edge_offset=0):
    dy, dx = blockade_mask.shape
    top_rows = np.full(dx, -1, dtype=int)
    is_sc = ~blockade_mask
    for j in range(dx):
        sc_rows = np.nonzero(is_sc[:, j])[0]
        if sc_rows.size == 0:
            continue
        top = sc_rows.max()                 # topmost SC cell in this column
        top = top - edge_offset             # step into the SC if requested
        top = min(max(top, 0), dy - 1)
        # guard: if the offset pushed us into a blockade cell, clamp back to SC
        if blockade_mask[top, j]:
            below = sc_rows[sc_rows <= top]
            top = below.max() if below.size else sc_rows.max()
        top_rows[j] = top
    return top_rows


print("Loading Js...")
Jsx_all, Jsy_all = load_current(Js_filename)

if FRAME < 0 or FRAME >= Jsx_all.shape[0]:
    raise IndexError(f"FRAME={FRAME} out of range 0..{Jsx_all.shape[0]-1}.")

Jx = Jsx_all[FRAME]                             # [y, x]
Jy = Jsy_all[FRAME]
Jmag = np.sqrt(Jx ** 2 + Jy ** 2)               # supercurrent magnitude map
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

# real-space coordinate grids (cell centers)
xs = np.linspace(0, xmax, dimx)
ys = np.linspace(0, ymax, dimy)

# trace the top SC edge and sample |J| along it
blockade_mask = make_blockade_mask(dimx, dimy)
top_rows = top_sc_row_per_column(blockade_mask, EDGE_OFFSET)

valid = top_rows >= 0
Jmag_edge = np.full(dimx, np.nan)
y_edge = np.full(dimx, np.nan)
cols = np.arange(dimx)
Jmag_edge[valid] = Jmag[top_rows[valid], cols[valid]]
y_edge[valid] = ys[top_rows[valid]]

# for masking the |J| map so the blockade doesn't drown out the SC crowding
if mask_blockade:
    Jmag_masked = np.where(blockade_mask, np.nan, Jmag)
else:
    Jmag_masked = Jmag

x_peak = xs[np.nanargmax(Jmag_edge)]
print(f"Top-SC edge |J|: range [{np.nanmin(Jmag_edge):.4g}, "
      f"{np.nanmax(Jmag_edge):.4g}]  peak at x = {x_peak:.2f}")
print(f"Edge y varies from {np.nanmin(y_edge):.2f} to {np.nanmax(y_edge):.2f} xi "
      f"(curves down at the constriction)")

# Top: supercurrent magnitude map.  Bottom: 1D |J| along the top SC edge.
fig, (ax_map, ax_line) = plt.subplots(
    2, 1, figsize=(8.0, 6.4), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True,
)

# top panel: |J| magnitude map
jmax = np.nanpercentile(Jmag_masked, 100.0)
if not (np.isfinite(jmax) and jmax > 0):
    jmax = 1.0
im_j = ax_map.imshow(
    Jmag_masked, origin="lower", extent=extent,
    cmap="magma", vmin=0.0, vmax=jmax, interpolation="nearest",
)
cb_j = fig.colorbar(im_j, ax=ax_map, fraction=0.046, pad=0.02, shrink=0.9)
cb_j.set_label(r"$|J_s|$ / $J_0$", fontsize=12)
# draw the traced top-SC edge on the map so you can see where the 1D cut is
ax_map.plot(xs[valid], y_edge[valid], color="cyan", lw=1.4, alpha=0.9,
            label="Top SC Boundary")
ax_map.set_title(r"Supercurrent Magnitude $|J_s| = \sqrt{J_x^2 + J_y^2}$",
                 fontsize=13)
ax_map.set_ylabel(r"y $(\xi)$", fontsize=12)
ax_map.set_ylim(0, ymax)
ax_map.legend(loc="upper right", fontsize=9, framealpha=0.8)

# bottom panel: 1D |J| along the top SC edge
ax_line.plot(xs, Jmag_edge, color="crimson", lw=1.8)
ax_line.fill_between(xs, 0.0, Jmag_edge, color="crimson", alpha=0.15)
ax_line.set_ylabel(r"Current Density $|J_s|$", fontsize=12)
ax_line.set_xlabel(r"x $(\xi)$", fontsize=12)
ax_line.set_title(r"$|J_s|$ Along the Top SC Boundary", fontsize=13)
ax_line.set_ylim(bottom=0.0)
ax_line.set_xlim(0, xmax)
ax_line.grid(True, alpha=0.3)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")