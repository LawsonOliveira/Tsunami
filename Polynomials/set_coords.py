import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

import numpy as np

def _set_coords_circle(nnodes_max=20):
    coords = np.empty((nnodes_max, 2), dtype=np.float64)
    for i in range(0, nnodes_max):
        coords[i, 0] = np.cos((i/nnodes_max)*2*np.pi)
        coords[i, 1] = np.sin((i/nnodes_max)*2*np.pi)
    return coords


def _set_coords_circle_concat(nnodes_max=20):
    # Second method (not so useful)
    theta = np.linspace(0, 2*np.pi, nnodes_max, dtype=np.float64)
    X = np.cos(theta).reshape((nnodes_max, 1))
    Y = np.sin(theta).reshape((nnodes_max, 1))
    coords = np.concatenate((X, Y), axis=1)
    return coords


def _set_coords_square(nnodes_max=20):
    import random
    coords = np.empty((nnodes_max, 2), dtype=np.float64)
    for i in range(0, nnodes_max):
        coords[i, 0] = random.uniform(-1, 1)
        coords[i, 1] = random.uniform(-1, 1)
    return coords


def _set_coords_circle_inside(nnodes_max=20):
    import random
    coords = np.empty((nnodes_max, 2), dtype=np.float64)
    for i in range(0, nnodes_max):
        coords[i, 0] = random.uniform(-np.pi, np.pi)
        coords[i, 1] = random.uniform(-np.pi, np.pi)
        coords[i, 0] = np.cos((coords[i, 0]))
        coords[i, 1] = np.sin((coords[i, 1]))
    return coords


def _set_coords_circle_bord_with_radius_interval(nnodes_max=20, rmin=1.0):
    import random
    coords = np.empty((nnodes_max, 2), dtype=np.float64)
    for i in range(0, nnodes_max):
        r = random.uniform(rmin, 1.0)
        aux = random.uniform(-np.pi, np.pi)
        coords[i, 0] = r*np.cos(aux)
        coords[i, 1] = r*np.sin(aux)
    return coords
