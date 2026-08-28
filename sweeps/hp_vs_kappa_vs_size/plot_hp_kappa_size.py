import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# one entry per grid size: data folder, legend label, and how many leading
# kappa points to drop -- the smallest kappa is unreliable on the larger grids
# ---------------------------------------------------------------------------
SERIES = [
    ("25x25",   r"25$\xi$ x 25$\xi$",   0),
    ("50x50",   r"50$\xi$ x 50$\xi$",   1),
    ("100x100", r"100$\xi$ x 100$\xi$", 1),
    ("200x200", r"200$\xi$ x 200$\xi$", 1),
]

YERR = 0.005        # error bar half-height on H_p


def group_by_kappa(folder):
    """Map each kappa to the list of array-task tables that swept it."""
    groups = {}
    for f in sorted(glob.glob(f"{folder}/grid_k*_task*.txt")):
        m = re.search(r"grid_k(\d+)_task\d+\.txt", f)
        if m:
            groups.setdefault(int(m.group(1)) / 100.0, []).append(f)
    return groups


def penetration_field(groups):
    """Lowest H at which a vortex entered, one point per kappa."""
    ks, Hp = [], []
    for kappa in sorted(groups):
        rows = np.vstack([np.loadtxt(f, ndmin=2) for f in groups[kappa]])
        H = rows[:, 2]
        vortex = rows[:, 4].astype(int)
        hit = H[vortex == 1]
        if hit.size:
            ks.append(kappa)
            Hp.append(hit.min())
        else:
            print(f"kappa = {kappa:g}: no vortex at any H (skipped)")
    return ks, Hp


plt.figure(figsize=(16, 9.5))

# line-of-best-fit tests
# ks_a, Hp_a = np.array(ks, float), np.array(Hp, float)
# kfine = np.linspace(ks_a.min(), ks_a.max(), 300)

# models = {
#     "a/k + c": (lambda k, a, c: a / k + c, (0.2, 0.0)),
#     "a/k^2 + c": (lambda k, a, c: a / k**2 + c, (0.2, 0.0)),
# }

# for name, (f, p0) in models.items():
#     try:
#         popt, _ = curve_fit(f, ks_a, Hp_a, p0=p0, maxfev=20000)
#     except RuntimeError:
#         print(f"{name}: fit did not converge")
#         continue
#     resid = Hp_a - f(ks_a, *popt)
#     r2 = 1 - np.sum(resid**2) / np.sum((Hp_a - Hp_a.mean())**2)
#     print(f"{name:20s} params = {np.round(popt, 4)}   R^2 = {r2:.4f}")
#     plt.plot(kfine, f(kfine, *popt), "--", label=f"{name}  $R^2$={r2:.3f}")
# plt.legend(fontsize=8)

for folder, label, skip in SERIES:
    ks, Hp = penetration_field(group_by_kappa(folder))
    plt.errorbar(ks[skip:], Hp[skip:], yerr=YERR, fmt='-o', label=label)

plt.xlabel(r"$\kappa$", fontsize=35)
plt.ylabel("$H_{p} \\, / \\, H_{c_2}$", fontsize=35)
plt.ylim(0.0, 0.5)
# plt.xlim(1.5, 8.0)
plt.title("Vortex Penetration Field vs. Kappa", fontsize=45)
plt.grid(alpha=0.3)
plt.legend(fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)

xtest = np.linspace(2, 8, 100)
ytest = np.log(xtest) / (2*xtest**2) + 0.05
plt.plot(xtest, ytest, '--')

plt.savefig("Hp_vs_kappa.png", dpi=200)
