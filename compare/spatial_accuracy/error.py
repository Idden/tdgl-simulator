import numpy as np
import matplotlib.pyplot as plt

hs = [2.0, 1.0, 0.5, 0.25]
L  = 30.0   # ymax
J  = 0.25   # current_max

# solve J = q(1 - q^2) for the small root by bisection on [0, 1/sqrt(3)]
lo, hi = 0.0, 1.0 / np.sqrt(3.0)
for k in range(100):
    q = 0.5 * (lo + hi)
    if q * (1 - q*q) < J:
        lo = q
    else:
        hi = q
rho_exact = 1 - q*q
print("q =", q, "  exact |psi|^2 =", rho_exact)

E = []
for h in hs:
    n   = int(L / h) + 1
    psi = np.fromfile(f"data_h={h}/psi.bin", dtype=np.complex128).reshape(-1, n, n)
    rho      = np.abs(psi[-1])**2   # last record,        rho[x, y]
    rho_prev = np.abs(psi[-2])**2   # second-to-last record

    err = 0.0
    for i in range(n):              # x index
        x = i * h
        if x < 10 or x > 20:        # only the middle, away from the leads
            continue
        for j in range(n):          # y index
            err = max(err, abs(rho[i, j] - rho_exact))

    steady = np.max(np.abs(rho - rho_prev))
    print(f"h = {h:<5}  E = {err:.4e}   steady = {steady:.1e}")
    E.append(err)

for k in range(len(hs) - 1):
    print(f"p({hs[k]} -> {hs[k+1]}) = {np.log2(E[k] / E[k+1]):.4f}")

plt.loglog(hs, E, "o-")
plt.xlabel("h")
plt.ylabel("max | |psi|^2 - (1 - q^2) |")
plt.title("Spatial error")
plt.grid(True)
plt.savefig("spatial_error.pdf")

print(E)