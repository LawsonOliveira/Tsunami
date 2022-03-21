import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))
import sympy as sm
import numpy as np

from itertools import combinations

from Polynomials.searching_combinations import f_sort, combi_indep

from time import time
from Polynomials.polynome4students_v2 import _set_coords_circle, _eval_polynome_numpy


def evaluate_poly_term(coord, combination):
    '''
    Variables :
        - coord (tuple or list or array): coordinates on a specific node
        - combination (tuple): the exponents for the coordinates
    Returns :
        integer being the value of a specific term in a polynome
    '''
    res = 1
    for i, power in enumerate(combination):
        if power:
            res *= coord[i]**power
    return res


def sys_matrix(coords):
    '''
    Variables :
        coords (array) : array of the coordinates where the futur polynome is evaluated in the system of equations
                        ex : coords[i,0] is the first coordonate of point number i
    Returns :
        - an array representing the system of equation we want to solve
        - the list of combinations kept to construct the matrix
    '''
    n = coords.shape[0]
    assert n
    D = coords.shape[1]

    l_combi = sorted(combi_indep(n, D), key=f_sort)
    N_col = len(l_combi)
    M = np.zeros((n, N_col), dtype=np.float64)

    # Evaluate the matrix with all the possible columns
    for i in range(n):
        for j, c in enumerate(l_combi):
            M[i, j] = evaluate_poly_term(coords[i, :], c)

    # Choose the columns to keep
    # Do we want to keep the first column ? Here we do (easy to compute)
    tol = 10**(-9)
    for c in combinations(list(range(1, N_col)), n-1):
        if np.abs(np.linalg.det(M[:, (0,)+c])) > tol:
            return M[:, (0,)+c], [l_combi[i] for i in (0,)+c]

    print("Did not find any invertible matrix")
    return None


def _set_A_xpy_numpy_matrix(coords, G):
    '''
    Variables :
        coords (array) : array of the coordinates where the futur polynome is evaluated in the system of equations
                        ex : coords[i,0] is the first coordonate of point number i
        G (numpy.array): column array of the condition values
    Returns :
        a polynome in a symbolic way thanks to sympy, such that poly(coords[i]) = G[i]
    '''
    D = coords.shape[1]
    # set variables
    X = [sm.symbols('x%i' % i) for i in range(D)]
    M, chosen_combi = sys_matrix(coords)

    C = np.dot(np.linalg.inv(M), G)

    sym_row = []
    for i in range(len(chosen_combi)):
        sym_row.append(evaluate_poly_term(X, chosen_combi[i]))
    expr = np.dot(np.array(sym_row), C)
    expr = sm.lambdify(X, expr, 'numpy')
    return expr


if __name__ == '__main__':
    nnodes_max = 10
    coords = _set_coords_circle(nnodes_max)

    G = np.ones((nnodes_max, 1), dtype=np.float64)

    poly_A = _set_A_xpy_numpy_matrix(coords, G)
    print([_eval_polynome_numpy(poly_A, coords[i, 0], coords[i, 1])
          for i in range(nnodes_max)])

    # quickly get mistakes after nnodes_max = 6 included
    # may be caused by np.linalg.inv and the approximation of np.linalg.det
    # => tol has been added and correct the error

    # get a long time to run
