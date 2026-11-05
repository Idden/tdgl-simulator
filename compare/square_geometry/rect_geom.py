# Rectangular-notch constriction (lengths in xi, origin at the center of the 30 x 30 domain).
# These numbers must match the parameters in the Fortran is_blockade.
#
#   y
#   15 +--------------------------------------+  <- domain edge (blockade above y = W)
#   14 |-----------------+    +---------------|  <- lead edge  y = +W
#      |                 |    |               |
#    8 |                 +----+               |  <- notch face y = +B
#      |  source    ->        channel     ->  drain
#   -8 |                 +----+               |
#      |                 |    |               |
#  -14 |-----------------+    +---------------|
#  -15 +--------------------------------------+
#                       -A    A

import numpy as np

L = 30.0    # domain size
W = 14.0    # lead half-width
B = 8.0     # channel half-width at the constriction
A = 3.0     # notch half-length along x


def corners():
    # film outline, counter-clockwise, starting at the bottom-left corner
    return np.array([
        [-L/2, -W], [-A, -W], [-A, -B], [A, -B], [A, -W], [L/2, -W],
        [L/2,  W],  [A,  W],  [A,  B],  [-A, B], [-A, W], [-L/2, W],
    ])


def boundary_points(h):
    # same outline with every edge split into pieces of length <= h,
    # corners kept exactly (so the pyTDGL boundary spacing is ~ h, like the Lorentzian case)
    c = corners()
    pts = []
    for k in range(len(c)):
        p0 = c[k]
        p1 = c[(k + 1) % len(c)]
        length = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        m = int(np.ceil(length / h - 1e-9))
        for s in range(m):
            pts.append(p0 + (p1 - p0) * s / m)
    return np.array(pts)


def outline_xy():
    # closed outline for plotting
    c = corners()
    return np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1])
