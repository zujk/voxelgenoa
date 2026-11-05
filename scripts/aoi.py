"""Area of interest shared by all pipeline steps.

World coords: EPSG:32632 (UTM 32N), x = easting, y = northing.
Grid coords: (i, j) continuous, i = column (east), j = row (south); cell (c, r) spans [c, c+1) x [r, r+1).
The grid is anchored on Piazza De Ferrari and not rotated (THETA = 0): Genoa's streets follow the hills and the coast
rather than one grid.
"""
import math

import numpy as np
from pyproj import Transformer

FERRARI = (8.93395, 44.40735)  # Piazza De Ferrari, lon/lat
CX, CY = Transformer.from_crs(4326, 32632, always_xy=True).transform(*FERRARI)
THETA = 0.0
# 4800 x 2800 m: from the Lanterna and Ponte Parodi in the west to Brignole and the Foce in the east, from the harbour
# up to Castelletto and Righi.
W, D = 4800, 2800
I0, J0 = 3100, 1600  # grid position of Piazza De Ferrari
RAW = "data/raw"

EU = np.array([math.cos(THETA), math.sin(THETA)])   # grid +i in world
EV = np.array([math.sin(THETA), -math.cos(THETA)])  # grid +j in world


def to_grid(x, y):
    dx, dy = np.asarray(x) - CX, np.asarray(y) - CY
    return I0 + dx * EU[0] + dy * EU[1], J0 + dx * EV[0] + dy * EV[1]


def to_world(i, j):
    u, v = np.asarray(i) - I0, np.asarray(j) - J0
    return CX + u * EU[0] + v * EV[0], CY + u * EU[1] + v * EV[1]


def world_bounds(margin=0):
    xs, ys = to_world(np.array([0, W, W, 0]), np.array([0, 0, D, D]))
    return xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin


def lonlat_bounds(margin=0):
    t = Transformer.from_crs(32632, 4326, always_xy=True)
    x0, y0, x1, y1 = world_bounds(margin)
    (lo0, la0), (lo1, la1) = t.transform(x0, y0), t.transform(x1, y1)
    return la0, lo0, la1, lo1  # S, W, N, E
