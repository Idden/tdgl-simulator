import numpy as np
import matplotlib.pyplot as plt
import h5py
import tdgl
from tdgl.geometry import box

dts = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
N = 30 * 30
u, eps, psi0 = 5.79, 0.25, 1.0

def psi_exact(t):
    return (1/eps + (1/psi0**2 - 1/eps) * np.exp(-2*eps*t/u)) ** -0.5

t = np.arange(41)  # record k is at t = k (record 0 = initial state)
E = []
for dt in dts:
    psi = np.fromfile(f"data_dt={dt}/psi.bin", dtype=np.complex128).reshape(-1, N)
    assert psi.shape[0] == 41, f"dt={dt}: expected 41 records, got {psi.shape[0]}"
    spread = np.max(np.abs(psi).max(axis=1) - np.abs(psi).min(axis=1))
    imag = np.abs(psi.imag).max()
    err = np.max(np.abs(np.abs(psi[1:, 0]) - psi_exact(t[1:])))
    E.append(err)
    print(f"dt={dt:<9}  E={err:.4e}   spread={spread:.1e}   max|Im|={imag:.1e}")

E = np.array(E)
print("observed order p =", np.round(np.log2(E[:-1] / E[1:]), 4))

# -------------------------------------------------------------------------------------------------------------

# 10x10 square film, no terminals, gamma = 0 (same model as the Fortran test)
layer = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=0)
film = tdgl.Polygon("film", points=box(10, points=21))
device = tdgl.Device("square", layer=layer, film=film, length_units="um")
device.make_mesh(max_edge_length=2, smooth=100)

E_py = []
for dt in dts:
    options = tdgl.SolverOptions(
        solve_time=40,
        adaptive=False,
        dt_init=dt,
        dt_max=dt,
        save_every=round(1/dt),
        output_file=f"pytdgl_dt={dt}.h5",
    )
    solution = tdgl.solve(device, options, disorder_epsilon=eps)

    # read the 41 saved states (t = 0, 1, ..., 40) straight from the file
    with h5py.File(solution.path, "r") as f:
        t = np.array([f["data"][str(k)].attrs["time"] for k in range(41)])
        psi = np.array([f["data"][str(k)]["psi"][:] for k in range(41)])

    py = np.abs(psi[:, 0])
    err_py = np.max(np.abs(py[1:] - psi_exact(t[1:])))
    E_py.append(err_py)
    spread = np.max(np.abs(psi).max(axis=1) - np.abs(psi).min(axis=1))

    mine = np.fromfile(f"data_dt={dt}/psi.bin", dtype=np.complex128).reshape(-1, 900)
    vs_mine = np.max(np.abs(py - np.abs(mine[:, 0])))

    print(f"dt={dt:<9}  E={err_py:.4e}   spread={spread:.1e}   "
          f"max|t-k|={np.max(np.abs(t - np.arange(41))):.1e}   max|pyTDGL-mine|={vs_mine:.1e}")

plt.loglog(dts, E, "o-", label="mine")
plt.loglog(dts, E_py, "s--", label="pyTDGL")
plt.legend()
plt.minorticks_off()
plt.xticks(dts, [str(d) for d in dts], rotation=30)
plt.xlabel("dt")
plt.ylabel("max|psi-exact|")
plt.title(r"Time-integration error ($\gamma=0,\ \epsilon=0.25$)")
plt.grid(True, which="both", alpha=0.3)
plt.savefig("dt_error.pdf", dpi=150)
plt.show()