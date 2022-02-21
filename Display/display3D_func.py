import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))
'''
'import sys' to 'sys.path'... to put in the first lines after each saving
'''

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Polynomials.set_coords import _set_coords_circle_bord_with_radius_interval
from Polynomials.polynomial_matrix_lambdify import _set_polynome_xpy_numpy_matrix, _set_polynome_sinxpy_numpy_matrix, _set_polynome_expxpy_numpy_matrix
from Polynomials.polynome4students_v2 import _eval_polynome_numpy, _set_polynome_expxpy_numpy_real, _set_coords_circle_concat



# only to show a graph


def _display_surface(poly, coords, Z_plan=0, N_axis=100, X_bound=1, points_min=None):
    '''
    Plot the surface covered by poly on (X,Y) with X and Y bounded by X_bound
    (poly is defined on R^2)

    Variables :
        poly (sympy expression)
        N_axis (int)
        X_bound (int) 
    '''
    fig = plt.figure()

    ax = Axes3D(fig)
    fig.add_axes(ax)
    X = np.linspace(-X_bound, X_bound, N_axis)
    # To be careful about "effet de bord"
    Y = X

    # Evaluate
    Z = np.empty((N_axis, N_axis), dtype=np.float64)
    Z2 = np.empty((N_axis, N_axis), dtype=np.float64)

    # Could be optimized (getting rid of "for" loops)
    for i in range(N_axis):
        for j in range(N_axis):
            Z[i, j] = _eval_polynome_numpy(poly, X[i], Y[j])
            Z2[i, j] = Z_plan
    # Points au bord
    for i in range(coords.shape[0]):
        ax.scatter(coords[i, 0], coords[i, 1], 0, marker="x",
                   c="black", alpha=1, linewidths=3)

    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, cmap='spring', linewidths=0.3)

    # Minimun point
    if points_min != None:
        ax.scatter(points_min[1], points_min[0],
                   points_min[2], marker="X", c="green", linewidths=3)
    ax.plot_surface(X, Y, Z2, cmap='Paired', linewidths=0.3)
    plt.show()


if __name__ == '__main__':
    print("Begin computation...")
    # Set
    nnodes_max = 20
    coords = _set_coords_circle_concat(nnodes_max)

    # coords = _set_coords_circle_bord_with_radius_interval(nnodes_max)
    # Plot circle :
    # X_circle = coords[:, 0]
    # Y_circle = coords[:, 1]
    # ax_proj = fig.gca(projection='3d')
    # ax_proj.plot(X_circle, Y_circle, zs=0, zdir='z', label='circle in (x,y,0)')

    # Set poly
    poly_mat = _set_polynome_xpy_numpy_matrix(coords)
    # poly_mat_sin = _set_polynome_sinxpy_numpy_matrix(coords)
    # poly_mat_exp = _set_polynome_expxpy_numpy_matrix(coords)
    # poly_real = _set_polynome_expxpy_numpy_real(coords)

    print("Voici poly")

    print("Let's check the behaviour of the set polynome on the boundary :")
    l_check = []
    tol = 10**(-9)
    for x in coords:
        epsilon = 10**(-9)
        v = _eval_polynome_numpy(poly_mat, x[0], x[1])
        v_decal = _eval_polynome_numpy(poly_mat, x[0]+epsilon, x[1]+epsilon)
        print("v_decal:",v_decal)
        l_check.append(v)
        if v > tol:
            print(tol)
            print(f"{v} is considered too large compare to {tol} !")
    print("Values of the set polynome :")
    print(l_check)

    _display_surface(poly_mat, coords, 0)

    print("Program has ended.")
