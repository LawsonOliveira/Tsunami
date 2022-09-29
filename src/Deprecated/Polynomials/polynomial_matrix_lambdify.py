import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from time import time
import sympy as sm
import numpy as np
from itertools import combinations
from Polynomials.searching_combinations import f_sort, combi_indep



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


def sys_augmented_matrix(coords):
    '''
    Variables :
        coords (array) : array of the coordinates where the futur polynome is meant to be null 
                        ex : coords[i,0] is the first coordonate of point number i
    Returns :
        - an array representing the system of equations we want to solve (one equation was added)
        - the list of combinations kept to construct the matrix
    '''

    n = coords.shape[0]
    assert n
    D = coords.shape[1]

    l_combi = sorted(combi_indep(n, D), key=f_sort)
    N_col = len(l_combi)
    A = np.zeros((n, N_col))

    # Evaluate the matrix with all the possible columns
    for i in range(n):
        for j, c in enumerate(l_combi):
            A[i, j] = evaluate_poly_term(coords[i, :], c)

    # Choose the columns to keep
    # Do we want to keep the first column ? Here we do (easy to compute)
    # How to choose tol ?
    tol = 10**(-3)
    i = 0
    for c in combinations(list(range(1, N_col-1)), n-1):
        print("c : ", c)
        det = np.abs(np.linalg.det(A[:, (0,)+c]))
        print("det : ", det)
        i += 1
        print("i : ", i)
        max_det = -np.infty
        if det > max_det:
            max_det = det
        if det > tol:
            # add an arbitrary coefficient and set it to 1
            # this coefficient is here chosen for the next possible combination
            B = A[:, (0,)+c+(c[-1]+1,)]
            last_row = np.zeros((1, n+1))
            last_row[0, -1] = 1
            B = np.concatenate((B, last_row), axis=0)
            print("Picked determinant :", det)
            return B, [l_combi[i] for i in (0,)+c+(c[-1]+1,)]
    print("Did not find any invertible matrix. \n max_det = :", max_det)
    return None


def _set_polynome_xpy_numpy_matrix(coords):
    '''
    Variables :
        coords (array) : array of the coordinates of various points where the polynome is meant to be null
    Returns :
        a polynome in a symbolic way thanks to sympy, null on coords
    '''
    D = coords.shape[1]
    # set variables
    X = [sm.symbols('x%i' % i) for i in range(D)]
    B, chosen_combi = sys_augmented_matrix(coords)
    # thanks to the last row we added in the equation to set the last coefficient to 1
    C = np.linalg.inv(B)[:, -1]
    sym_row = []
    for i in range(len(chosen_combi)):
        sym_row.append(evaluate_poly_term(X, chosen_combi[i]))
    expr = np.dot(np.array(sym_row), C)

    expr = sm.lambdify(X, expr, 'numpy')
    return expr


def _set_polynome_sinxpy_numpy_matrix(coords):
    '''
    Variables :
        coords (array) : array of the coordinates of various points where the polynome is meant to be null
    Returns :
        a polynome in a symbolic way thanks to sympy, null on coords
    '''

    # the same as before :
    D = coords.shape[1]
    # set variables
    X = [sm.symbols('x%i' % i) for i in range(D)]
    B, chosen_combi = sys_augmented_matrix(coords)
    # thanks to the last row we added in the equation to set the last coefficient to 1
    C = np.linalg.inv(B)[:, -1]
    sym_row = []
    for i in range(len(chosen_combi)):
        sym_row.append(evaluate_poly_term(X, chosen_combi[i]))
    expr = np.dot(np.array(sym_row), C)

    # difference here :
    expr = sm.sin(expr)
    expr = sm.lambdify(X, expr, 'numpy')
    return expr


def _set_polynome_expxpy_numpy_matrix(coords):
    '''
    Variables :
        coords (array) : array of the coordinates of various points where the polynome is meant to be null
    Returns :
        a polynome in a symbolic way thanks to sympy, null on coords
    '''

    # the same as before :
    D = coords.shape[1]
    # set variables
    X = [sm.symbols('x%i' % i) for i in range(D)]
    B, chosen_combi = sys_augmented_matrix(coords)
    # thanks to the last row we added in the equation to set the last coefficient to 1
    C = np.linalg.inv(B)[:, -1]
    sym_row = []
    for i in range(len(chosen_combi)):
        sym_row.append(evaluate_poly_term(X, chosen_combi[i]))
    expr = np.dot(np.array(sym_row), C)

    # difference here :
    expr = 1 - sm.exp(-10 * expr**2)
    expr = sm.lambdify(X, expr, 'numpy')
    return expr


if __name__ == '__main__':
    print('rien')
