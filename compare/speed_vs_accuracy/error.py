import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
import matplotlib.pyplot as plt
import tdgl
from tdgl.geometry import box

hs   = [2.0, 1.0, 0.5, 0.25]
L    = 30.0
J    = 0.25
tmax = 200.0

E_mine = [0.0076753447787885065, 0.0016383456723781942, 0.00039585675489117733, 9.81497749963367e-05]
T_mine = [0.6010, 1.5725, 5.2900, 21.5398]   # Fortran, 1 thread, dt = 0.05

# ---- exact answer: J = q(1 - q^2), small root ----
lo, hi = 0.0, 1.0 / np.sqrt(3.0)
for k in range(100):
    q = 0.5 * (lo + hi)
    if q * (1 - q*q) < J:
        lo = q
    else:
        hi = q
rho_exact = 1 - q*q

# ---- pyTDGL strip: 30x30 film centered at 0, terminals on left/right edges ----
layer  = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=0)
film   = tdgl.Polygon("film", points=box(L))
source = tdgl.Polygon("source", points=box(L/100, 1.1*L)).translate(dx=-L/2)
drain  = source.scale(xfact=-1).set_name("drain")
device = tdgl.Device("strip", layer=layer, film=film,
                     terminals=[source, drain], length_units="um")
I = J * device.K0.to("uA/um").magnitude * L / 4    # pyTDGL: J_dimless = 4 I / (K0 W)

E_py, N_py, T_py, dt_py = [], [], [], []
for h in hs:
    device.make_mesh(max_edge_length=h, smooth=100)

    dt = 0.05
    while True:
        steps = int(round(tmax / dt))
        opts = tdgl.SolverOptions(solve_time=tmax, adaptive=False, dt_init=dt, dt_max=dt,
                                  save_every=steps // 10, current_units="uA",
                                  output_file=f"pytdgl_strip_h={h}.h5")
        try:
            t0  = time.perf_counter()
            sol = tdgl.solve(device, opts, terminal_currents=dict(source=I, drain=-I))
            T   = time.perf_counter() - t0
            break
        except RuntimeError:
            dt = dt / 2
            print("unstable, halving dt to", dt)

    rho = np.abs(sol.tdgl_data.psi)**2            # final state
    sol.load_tdgl_data(solve_step=-2)             # previous saved state
    steady = np.max(np.abs(rho - np.abs(sol.tdgl_data.psi)**2))

    x = device.mesh.sites[:, 0]                   # units of xi, centered at 0
    err = 0.0
    for k in range(len(x)):
        if abs(x[k]) <= 5:                        # same region as 10 <= x <= 20
            err = max(err, abs(rho[k] - rho_exact))

    print(f"h = {h:<5} sites = {len(x):<6} dt = {dt:<8} E = {err:.4e}  "
          f"steady = {steady:.1e}  time = {T:.2f} s")
    E_py.append(err)
    N_py.append(len(x))
    T_py.append(T)
    dt_py.append(dt)

for k in range(len(hs) - 1):
    print(f"pyTDGL p({hs[k]} -> {hs[k+1]}) = {np.log2(E_py[k] / E_py[k+1]):.4f}")

# ---- plots ----
N_mine = [(L / h + 1)**2 for h in hs]

plt.figure()
plt.loglog(N_mine, E_mine, "o-",  label="mine")
plt.loglog(N_py,   E_py,   "s--", label="pyTDGL")
plt.xlabel("number of grid points / mesh sites  N")
plt.ylabel("max | |psi|^2 - (1 - q^2) |")
plt.legend(); plt.grid(True)
plt.savefig("error_vs_N.pdf")

plt.figure()
plt.loglog(T_mine, E_mine, "o-",  label="mine")
plt.loglog(T_py,   E_py,   "s--", label="pyTDGL")
plt.xlabel("wall time (s)")
plt.ylabel("max | |psi|^2 - (1 - q^2) |")
plt.legend(); plt.grid(True)
plt.savefig("error_vs_time.pdf")