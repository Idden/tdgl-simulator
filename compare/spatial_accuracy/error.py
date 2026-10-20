import numpy as np
import matplotlib.pyplot as plt

hs = [2.0, 1.0, 0.5, 0.25]
L = 30.0 # ymax in the Fortran
J = 0.25 # current_max in the Fortran

# Exact uniform current-carrying state: J = q(1 - q^2), take the smaller positive root
roots = np.roots([-1, 0, 1, -J])
q = min(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
rho_exact = 1 - q**2
print(f"q = {q:.6f}   exact |psi|^2 = {rho_exact:.6f}\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
E = []

for h in hs:
    n = int(round(L / h)) + 1                       # grid points per side
    psi = np.fromfile(f"data_h={h}/psi.bin", dtype=np.complex128).reshape(-1, n, n)
    last, prev = psi[-1], psi[-2]                   # psi[record, x, y]
    x = np.arange(n) * h

    # steady-state check: change between the last two saved records
    steady = np.max(np.abs(np.abs(last)**2 - np.abs(prev)**2))

    # error: max over the central region (10 <= x <= 20, all y), away from the leads
    mid = (x >= 10) & (x <= 20)
    rho = np.abs(last)**2
    err = np.max(np.abs(rho[mid, :] - rho_exact))
    E.append(err)

    # profiles along the middle row in y
    yc = n // 2
    row = last[:, yc]
    xm = x[:-1] + h / 2                             # edge midpoints
    dtheta = np.abs(np.diff(np.unwrap(np.angle(row)))) / h
    Js = np.abs(np.imag(np.conj(row[:-1]) * row[1:])) / h   # same formula as calculate_Js

    ax[0].plot(x, np.abs(row)**2, "o-", ms=3, label=f"h={h}")
    ax[1].plot(xm, dtheta, "o-", ms=3, label=f"h={h}")
    ax[2].plot(xm, Js, "o-", ms=3, label=f"h={h}")

    midm = (xm >= 10) & (xm <= 20)
    print(f"h={h:<5} E={err:.4e}   steady={steady:.1e}   "
          f"dtheta/dx={dtheta[midm].mean():.6f} (q={q:.6f})   Js={Js[midm].mean():.6f} (J={J})")

E = np.array(E)
print("\nobserved order p =", np.round(np.log2(E[:-1] / E[1:]), 4))

ax[0].axhline(rho_exact, color="k", ls="--", label="exact")
ax[1].axhline(q, color="k", ls="--", label="exact q")
ax[2].axhline(J, color="k", ls="--", label="applied J")
for a, t in zip(ax, [r"$|\psi|^2$", r"$|d\theta/dx|$", r"$|J_s|$"]):
    a.set_xlabel(r"x ($\xi$)")
    a.set_title(t + r"  along $y = 15$")
    a.legend()
fig.tight_layout()
fig.savefig("steady_profiles.pdf")

plt.figure()
plt.loglog(hs, E, "o-")
plt.minorticks_off()
plt.xticks(hs, [str(h) for h in hs])
plt.xlabel(r"h ($\xi$)")
plt.ylabel(r"max $\left||\psi|^2 - (1-q^2)\right|$  in $10\leq x \leq 20$")
plt.title("Spatial error, uniform current state")
plt.grid(True, which="both", alpha=0.3)
plt.savefig("spatial_error.pdf")