import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
import tdgl
from tdgl.geometry import box
import matplotlib.pyplot as plt

hs = [2.0, 1.0, 0.5, 0.25]
L = 30.0
J = 0.10        # same as current_max in the Fortran
tmax = 200.0

out = open("pytdgl_results.txt", "w")
out.write("# h N T rho_c rho_e steady\n")

RC, RE = [], []
for h in hs:
    # constriction outline, same formula as the Fortran is_blockade (in xi)
    xs = np.linspace(-L/2, L/2, int(L/h) + 1)
    hw  = 13.5 - (13.5 - 0.57*13.5) / (1 + (xs/6.0)**2)
    pts = np.concatenate([np.column_stack([xs, hw]),
                        np.column_stack([xs[::-1], -hw[::-1]])])

    layer = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=0)
    film = tdgl.Polygon("film", points=pts)
    source = tdgl.Polygon("source", points=box(L/100, 1.1*L)).translate(dx=-L/2)
    drain = source.scale(xfact=-1).set_name("drain")
    device = tdgl.Device("constriction", layer=layer, film=film,
                        terminals=[source, drain], length_units="um")

    W_lead = 2 * hw[0]                                   # lead width in xi
    I = J * device.K0.to("uA/um").magnitude * W_lead / 4

    device.make_mesh(max_edge_length=h, smooth=100)

    m = device.mesh
    print("N =", len(m.sites), "  min area =", m.areas.min(), "  median area =", np.median(m.areas))

    if h >= 0.25:                                        # mesh plot too slow at 0.125
        fig, ax = plt.subplots(figsize=(10, 9))
        m.plot(ax=ax, show_edges=True, show_dual_edges=True)
        bad = np.argsort(m.areas)[:20]
        ax.plot(m.sites[bad, 0], m.sites[bad, 1], "ro", ms=8)
        ax.set_aspect("equal")
        plt.savefig(f"mesh_h={h}.png", dpi=200)
        plt.close(fig)

    dt = 0.01
    while True:
        steps = int(round(tmax / dt))
        opts = tdgl.SolverOptions(solve_time=tmax, adaptive=False, dt_init=dt, dt_max=dt,
                                  save_every=steps // 10, current_units="uA",
                                  output_file=f"pytdgl_constr_h={h}.h5")
        try:
            t0  = time.perf_counter()
            sol = tdgl.solve(device, opts, terminal_currents=dict(source=I, drain=-I))
            T   = time.perf_counter() - t0
            break
        except RuntimeError:
            dt = dt / 2
            print("unstable, halving dt to", dt)

    rho = np.abs(sol.tdgl_data.psi)**2
    sol.load_tdgl_data(solve_step=-2)
    steady = np.max(np.abs(rho - np.abs(sol.tdgl_data.psi)**2))

    x = device.mesh.sites[:, 0]
    y = device.mesh.sites[:, 1]
    rho_c = rho[np.argmin(x*x + y*y)]
    rho_e = 1e9
    for k in range(len(x)):
        if abs(x[k]) <= 1.0:
            rho_e = min(rho_e, rho[k])

    print(f"h = {h:<5} N = {len(x):<6} dt = {dt:<8} rho_c = {rho_c:.6f}  "
          f"rho_e = {rho_e:.6f}  steady = {steady:.1e}  time = {T:.2f} s")
    out.write(f"{h} {len(x)} {T} {rho_c:.10f} {rho_e:.10f} {steady:.3e}\n")
    out.flush()
    RC.append(rho_c); RE.append(rho_e)

out.close()

for k in range(len(hs) - 2):
    pc = np.log2(abs(RC[k] - RC[k+1]) / abs(RC[k+1] - RC[k+2]))
    pe = np.log2(abs(RE[k] - RE[k+1]) / abs(RE[k+1] - RE[k+2]))
    print(f"p from h = {hs[k]}, {hs[k+1]}, {hs[k+2]}:  center {pc:.3f}   edge {pe:.3f}")