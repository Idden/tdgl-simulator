import numpy as np
import matplotlib.pyplot as plt
import h5py
import tdgl
from tdgl.geometry import box
from scipy.integrate import solve_ivp

g = 10                     # gamma
folder = "data_dt"     # Fortran output folders: data_g10_dt=<dt>/psi.bin
dts = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
N = 30 * 30
u, eps = 5.79, 0.25
t = np.arange(41)          # saved times t = 0, 1, ..., 40

# Reference: high-precision solve of  u*sqrt(1 + g^2 psi^2) dpsi/dt = (eps - psi^2) psi
ref = solve_ivp(lambda s, y: (eps - y**2) * y / (u * np.sqrt(1 + g**2 * y**2)),
                [0, 40], [1.0], method="DOP853", rtol=1e-13, atol=1e-15, t_eval=t).y[0]

# ---------------- Fortran ----------------
E = []
for dt in dts:
    psi = np.fromfile(f"{folder}={dt}/psi.bin", dtype=np.complex128).reshape(-1, N)
    assert psi.shape[0] == 41, f"dt={dt}: expected 41 records, got {psi.shape[0]}"
    err = np.max(np.abs(np.abs(psi[1:, 0]) - ref[1:]))
    E.append(err)
    print(f"mine    dt={dt:<9}  E={err:.4e}")
E = np.array(E)
print("mine    p =", np.round(np.log2(E[:-1] / E[1:]), 4))

# ---------------- pyTDGL ----------------
layer = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=g)
film = tdgl.Polygon("film", points=box(10, points=21))
device = tdgl.Device("square", layer=layer, film=film, length_units="um")
device.make_mesh(max_edge_length=2, smooth=100)

E_py = []
for dt in dts:
    options = tdgl.SolverOptions(solve_time=40, adaptive=False, dt_init=dt, dt_max=dt,
                                 save_every=round(1/dt), output_file=f"pytdgl_g{g}_dt={dt}.h5")
    try:
        solution = tdgl.solve(device, options, disorder_epsilon=eps)
    except RuntimeError:
        print(f"pyTDGL  dt={dt:<9}  UNSTABLE (blew up)")
        E_py.append(np.nan)
        continue
    with h5py.File(solution.path, "r") as f:
        tk = np.array([f["data"][str(k)].attrs["time"] for k in range(41)])
        psi = np.array([f["data"][str(k)]["psi"][:] for k in range(41)])
    py = np.abs(psi[:, 0])
    err_py = np.max(np.abs(py[1:] - ref[1:]))
    E_py.append(err_py)
    mine = np.fromfile(f"{folder}={dt}/psi.bin", dtype=np.complex128).reshape(-1, N)
    print(f"pyTDGL  dt={dt:<9}  E={err_py:.4e}   max|t-k|={np.max(np.abs(tk - t)):.1e}   "
          f"max|pyTDGL-mine|={np.max(np.abs(py - np.abs(mine[:, 0]))):.1e}")
E_py = np.array(E_py)
print("pyTDGL  p =", np.round(np.log2(E_py[:-1] / E_py[1:]), 4))

# ---------------- plot ----------------
plt.loglog(dts, E, "o-", label="mine")
plt.loglog(dts, E_py, "s--", label="pyTDGL")
plt.legend()
plt.minorticks_off()
plt.xticks(dts, [str(d) for d in dts], rotation=30)
plt.xlabel("dt")
plt.ylabel("max|psi - reference|")
plt.title(rf"Time-integration error ($\gamma={g},\ \epsilon=0.25$)")
plt.grid(True, which="both", alpha=0.3)
plt.savefig(f"dt_error_g{g}.pdf", dpi=150)