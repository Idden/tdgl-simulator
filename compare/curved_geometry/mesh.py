# Plots the pyTDGL constriction mesh for each h:
#   tiny blue dots = mesh sites, red dots = sites with malformed Voronoi cells
# Saves mesh_h=<h>.png

import logging, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import tdgl
from tdgl.geometry import box

hs = [2.0, 1.0, 0.5, 0.25]
L  = 30.0

# catch pyTDGL's "Malformed Voronoi cell surrounding boundary site N" messages
bad_sites = []

class Catch(logging.Handler):
    def emit(self, record):
        m = re.search(r"boundary site (\d+)", record.getMessage())
        if m:
            bad_sites.append(int(m.group(1)))

logging.getLogger("tdgl").addHandler(Catch())

for h in hs:
    # same geometry as pytdgl.py
    xs  = np.linspace(-L/2, L/2, int(L/h) + 1)
    hw  = 13.5 - (13.5 - 0.57*13.5) / (1 + (xs/6.0)**2)
    pts = np.concatenate([np.column_stack([xs, hw]),
                          np.column_stack([xs[::-1], -hw[::-1]])])

    layer  = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=0)
    film   = tdgl.Polygon("film", points=pts)
    source = tdgl.Polygon("source", points=box(L/100, 1.1*L)).translate(dx=-L/2)
    drain  = source.scale(xfact=-1).set_name("drain")
    device = tdgl.Device("constriction", layer=layer, film=film,
                         terminals=[source, drain], length_units="um")

    bad_sites.clear()
    with warnings.catch_warnings(record=True) as w:      # in case pyTDGL uses warnings instead of logging
        warnings.simplefilter("always")
        device.make_mesh(max_edge_length=h, smooth=100)
    for item in w:
        m = re.search(r"boundary site (\d+)", str(item.message))
        if m:
            bad_sites.append(int(m.group(1)))

    mesh = device.mesh
    x = mesh.sites[:, 0]
    y = mesh.sites[:, 1]
    bad = np.array(sorted(set(bad_sites)), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.triplot(x, y, mesh.elements, lw=0.1, color="0.75")
    ax.plot(x, y, "b.", ms=0.5)
    if len(bad) > 0:
        ax.plot(x[bad], y[bad], "r.", ms=3)
    ax.set_aspect("equal")
    ax.set_title(f"h = {h}   N = {len(x)}   malformed = {len(bad)}")
    plt.savefig(f"mesh_h={h}.pdf")
    plt.close(fig)

    print(f"h = {h:<5} N = {len(x):<7} malformed = {len(bad)}")