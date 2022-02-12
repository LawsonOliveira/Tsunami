import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))

import sympy as sm
import numpy as np
import time
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Polynomials.polynome4students_v2 import _eval_polynome_numpy
# only to show a graph
from Polynomials.polynomial_matrix_lambdify import _set_polynome_xpy_numpy_matrix, _set_polynome_sinxpy_numpy_matrix, _set_polynome_expxpy_numpy_matrix
from Polynomials.polynome4students_v2 import _eval_polynome_numpy, _set_polynome_expxpy_numpy_real, _set_coords_circle_concat


def _display_surface(poly, N_axis=100, X_bound=1,points_min=None):
    '''
    Plot the surface covered by poly on (X,Y) with X and Y bounded by X_bound
    (poly is defined on R^2)

    Variables :
        poly (sympy expression)
        N_axis (int)
        X_bound (int) 
    '''
    fig = plt.figure()

    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    X = np.linspace(-X_bound, X_bound, N_axis)
    # To be careful about "effet de bord"
    Y = X

    # Evaluate
    Z = np.empty((N_axis, N_axis), dtype=np.float64)
    # Could be optimized (getting rid of "for" loops)
    for i in range(N_axis):
        for j in range(N_axis):
            Z[i, j] = _eval_polynome_numpy(poly, X[i], Y[j])

    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z)
    if points_min!=None:
        ax.scatter(points_min[1],points_min[0],points_min[2],marker="x",c="red")
    plt.show()


if __name__ == '__main__':
    # Set
    nnodes_max = 20

    coords = _set_coords_circle_concat(nnodes_max)

    # Plot circle :
    # X_circle = coords[:, 0]
    # Y_circle = coords[:, 1]
    # ax_proj = fig.gca(projection='3d')
    # ax_proj.plot(X_circle, Y_circle, zs=0, zdir='z', label='circle in (x,y,0)')

    # Set poly
    poly_mat = _set_polynome_xpy_numpy_matrix(coords)
    poly_mat_sin = _set_polynome_sinxpy_numpy_matrix(coords)
    poly_mat_exp = _set_polynome_expxpy_numpy_matrix(coords)
    poly_real = _set_polynome_expxpy_numpy_real(coords)
    _display_surface(poly_mat_sin)
