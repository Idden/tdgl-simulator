import numpy as np

hs = [2.0, 1.0, 0.5, 0.25]
L  = 30.0

out = open("mine_results.txt", "w")
out.write("# h N T rho_c rho_e steady\n")

RC, RE = [], []
for h in hs:
    n   = int(L / h) + 1
    psi = np.fromfile(f"data_h={h}/psi.bin", dtype=np.complex128).reshape(-1, n, n)
    T   = float(open(f"data_h={h}/time.txt").read())

    rho      = np.abs(psi[-1])**2     # rho[x, y]
    rho_prev = np.abs(psi[-2])**2
    steady = np.max(np.abs(rho - rho_prev))

    N = 0
    rho_c = 0.0; best = 1e9
    rho_e = 1e9
    for i in range(n):                # x index
        x = (i - (n - 1) / 2) * h
        for j in range(n):            # y index
            y = (j - (n - 1) / 2) * h
            if rho[i, j] == 0.0:      # blockade site
                continue
            N = N + 1
            if x*x + y*y < best:
                best = x*x + y*y; rho_c = rho[i, j]
            if abs(x) <= 1.0:
                rho_e = min(rho_e, rho[i, j])

    print(f"h = {h:<5} N = {N:<6} T = {T:<9.3f} rho_c = {rho_c:.6f}  rho_e = {rho_e:.6f}  steady = {steady:.1e}")
    out.write(f"{h} {N} {T} {rho_c:.10f} {rho_e:.10f} {steady:.3e}\n")
    RC.append(rho_c); RE.append(rho_e)

out.close()

for k in range(len(hs) - 2):
    pc = np.log2(abs(RC[k] - RC[k+1]) / abs(RC[k+1] - RC[k+2]))
    pe = np.log2(abs(RE[k] - RE[k+1]) / abs(RE[k+1] - RE[k+2]))
    print(f"p from h = {hs[k]}, {hs[k+1]}, {hs[k+2]}:  center {pc:.3f}   edge {pe:.3f}")