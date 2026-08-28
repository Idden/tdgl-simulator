import matplotlib
matplotlib.use("Agg")

import warnings
import numpy as np
import matplotlib.pyplot as plt

# Fortran parameters
dimx = 60
dimy = 60
ymax = 30.0
n_t = 100000
tmax = 1000.0
stride = 250
mask_blockade = False

FRAME = 96
EDGE_OFFSET = 1

# --- binning of the 1D edge trace (right panel only) ----------------------
BIN = 3                  # average this many adjacent x columns per point

# --- global text sizing ---------------------------------------------------
BASE       = 22          # ticks, colorbar numbers
TITLE_SIZE = 25          # panel titles
LABEL_SIZE = 22          # axis labels
CBAR_LABEL = 22          # colorbar labels

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
out_name = f"poster_current_density_{sim_ext_mag}mag_{sim_curr}curr_f{FRAME}.png"

n_snapshots = n_t // stride
h = ymax / (dimy - 1)
dt = tmax / n_t
dt_snap = stride * dt
xmax = h * (dimx - 1)
extent = [0, xmax, 0, ymax]

Js_filename = "data_curr_crowd/Js.bin"


def load_current(filename):
    # supercurrent Jsx, Jsy on the dimx*dimy grid
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
        if blockade_mask[top, j]:
            below = sc_rows[sc_rows <= top]
            top = below.max() if below.size else sc_rows.max()
        top_rows[j] = top
    return top_rows


def bin_edge_trace(x, y, nbin):
    """Chop x and y into non-overlapping groups of nbin adjacent columns.

    Returns (x_c, y_mean, y_min, y_max), one entry per group.
    NaNs inside a group are ignored; a group that is entirely NaN
    comes back as NaN and simply is not drawn.
    """
    n_groups = x.size // nbin
    n_used = n_groups * nbin            # drop the leftover columns at the end
    xg = x[:n_used].reshape(n_groups, nbin)
    yg = y[:n_used].reshape(n_groups, nbin)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN groups
        y_mean = np.nanmean(yg, axis=1)
        y_min = np.nanmin(yg, axis=1)
        y_max = np.nanmax(yg, axis=1)
    x_c = xg.mean(axis=1)               # group centre in x

    return x_c, y_mean, y_min, y_max


print("Loading Js...")
Jsx_all, Jsy_all = load_current(Js_filename)

if FRAME < 0 or FRAME >= Jsx_all.shape[0]:
    raise IndexError(f"FRAME={FRAME} out of range 0..{Jsx_all.shape[0]-1}.")

Jx = Jsx_all[FRAME]                             # [y, x]
Jy = Jsy_all[FRAME]
Jmag = np.sqrt(Jx ** 2 + Jy ** 2)               # supercurrent magnitude map
t_here = (FRAME + 1) * dt_snap
print(f"Frame {FRAME}  ->  t = {t_here:.3f}")

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

if mask_blockade:
    Jmag_masked = np.where(blockade_mask, np.nan, Jmag)
else:
    Jmag_masked = Jmag

x_peak = xs[np.nanargmax(Jmag_edge)]
print(f"Top-SC edge |J|: range [{np.nanmin(Jmag_edge):.4g}, "
      f"{np.nanmax(Jmag_edge):.4g}]  peak at x = {x_peak:.2f}")

# average the edge trace in non-overlapping groups of BIN adjacent columns
x_bin, J_bin_mean, J_bin_min, J_bin_max = bin_edge_trace(xs, Jmag_edge, BIN)
n_dropped = dimx - (dimx // BIN) * BIN
print(f"Binned edge trace: {x_bin.size} points of {BIN} columns each"
      + (f" ({n_dropped} trailing column(s) dropped)" if n_dropped else ""))
if np.any(np.isfinite(J_bin_mean)):
    print(f"Binned peak <|J|> = {np.nanmax(J_bin_mean):.4g} "
          f"at x = {x_bin[np.nanargmax(J_bin_mean)]:.2f}")

# ================================================================== figure
# Two panels side by side, equal width:
#   left  = supercurrent magnitude map (square, 150x150 grid)
#   right = 1D |J| along the top SC boundary
fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(15.0, 6.6))

TITLE_PAD = 13
CBAR_FRAC = 0.046
CBAR_PAD  = 0.02

# --- left: |J| magnitude map ---
jmax = np.nanpercentile(Jmag_masked, 100.0)
if not (np.isfinite(jmax) and jmax > 0):
    jmax = 1.0
im_j = ax_map.imshow(
    Jmag_masked, origin="lower", extent=extent,
    cmap="magma", vmin=0.0, vmax=jmax, interpolation="nearest",
)
ax_map.plot(xs[valid], y_edge[valid], color="lime", lw=4.0, alpha=0.9,
            label="Top SC Boundary")
ax_map.set_title(r"Supercurrent Magnitude $|J_s|$", pad=TITLE_PAD)
ax_map.set_xlabel(r"x $(\xi)$", labelpad=8)
ax_map.set_ylabel(r"y $(\xi)$")
ax_map.set_box_aspect(1)
ax_map.set_xlim(0, xmax)
ax_map.set_ylim(0, ymax)
ax_map.tick_params(labelsize=BASE, length=6, width=1.4)
ax_map.legend(loc="upper right", fontsize=BASE * 0.75, framealpha=0.85)
cb_j = fig.colorbar(im_j, ax=ax_map, fraction=CBAR_FRAC, pad=CBAR_PAD)
cb_j.set_label(r"$|J_s|$ / $J_0$", fontsize=CBAR_LABEL)
cb_j.ax.tick_params(labelsize=BASE)

# --- right: 1D |J| along the top SC boundary, averaged in groups of BIN ---
yerr = np.vstack([J_bin_mean - J_bin_min, J_bin_max - J_bin_mean])
ax_line.errorbar(
    x_bin, J_bin_mean, yerr=yerr,
    color="lime", lw=2.6,                    # connecting line through means
    marker="o", ms=5.0, mfc="lime", mec="lime",
    ecolor="#0b6b0b", elinewidth=1.6,        # darker green bars for contrast
    capsize=4.0, capthick=1.6,
    zorder=3,
)
ax_line.set_title(r"$|J_s|$ Along the Top SC Boundary", pad=TITLE_PAD)
ax_line.set_xlabel(r"x $(\xi)$", labelpad=8)
ax_line.set_ylabel(r"Current Density $|J_s|$")
ax_line.set_box_aspect(1)
ax_line.set_xlim(0, xmax)
ax_line.set_ylim(bottom=0)
ax_line.grid(True, alpha=0.3)
ax_line.tick_params(labelsize=BASE, length=6, width=1.4)
ax_line.axhline(y=0.508, color='crimson', linestyle='--', label=r"$J_d$")
ax_line.legend(fontsize=20, loc="upper right")

fig.subplots_adjust(left=0.02, right=1.0, top=0.9, bottom=0.14, wspace=0.32)

fig.savefig(out_name, dpi=200)
plt.close(fig)
print(f"Saved {out_name}")