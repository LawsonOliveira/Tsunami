import sys  
from pathlib import Path  
import time
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))
from time import time
import sympy as sm
import numpy as np
import random as rd

from itertools import permutations, combinations_with_replacement, combinations
# Have to look for how to do without this library : 3 functions to code

"""
A l'aide de sympy, on obtient les coefficients pour les polynômes P3 du document
"""


def f_sort(t):
    '''
    A criteria to sort the possible combinations. It will prefer small exponents and more dependance between different dimension. 
    The less interesting is a combination, the higher is the image.

    Variable : 
        t : a tuple
    Returns :
        res : a float to sort a list of tuples
    '''
    res = sum(t)
    n_zeros = 0
    for x in t:
        if not(x):
            n_zeros += 1
    res += n_zeros/len(t)

    return res


def combi(n_points, D):
    '''
    returns sorted possible combinations of exponents.
    Ex : (x,y,z) =>  []

    Variables :
        - n_points (int) = the number of points taken to set the futur polynome. So it is the higher possible exponent.
        - D (int) = the dimension.
    Returns :
        list of sorted possible combinations of exponents
    '''
    L = list(range(n_points))
    # print(L)
    combi_poss = set()
    for c in combinations_with_replacement(L, D):
        # print(c)
        if not c in combi_poss:
            for c in permutations(c):
                combi_poss = combi_poss | {c}
    combi_poss = list(combi_poss)
    return sorted(combi_poss, key=f_sort)


def evaluate(coord, combination):
    '''
    Variables :
        - coord (tuple or list): coordinates on a specific node
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
        coords (list) : list of the coordinates where the futur polynome is meant to be null
    Returns :
        - an array representing the system of equation we want to solve
        - the list of combinations kept to construct the matrix
    '''

    n = len(coords)
    assert n
    D = len(coords[0])

    l_combi = combi(n, D)
    N_col = len(l_combi)
    A = np.zeros((n, N_col))

    # Evaluate the matrix with all the possible columns
    for i in range(n):
        for j, c in enumerate(l_combi):
            A[i, j] = evaluate(coords[i], c)

    # Choose the columns to keep
    # Do we want to keep the first column ? Here we do (easy to compute)
    for c in combinations(list(range(1, N_col-1)), n-1):
        if np.linalg.det(A[:, (0,)+c]):
            # add an arbitrary coefficient and set it to 1
            # this coefficient is here chosen for the next possible combination
            B = A[:, (0,)+c+(c[-1]+1,)]
            last_row = np.zeros((1, n+1))
            last_row[0, -1] = 1
            B = np.concatenate((B, last_row), axis=0)
            return B, [l_combi[i] for i in (0,)+c+(c[-1]+1,)]

    print("Did not find any invertible matrix")
    return None


def set_polynomial_matrix(coords):
    '''
    Variables :
        coords (list) : list of the coordinates of various points where the polynome is meant to be null
    Returns :
        a polynome in a symbolic way thanks to sympy
    '''
    D = len(coords[0])
    # set variables
    X = [sm.symbols('x%i' % i) for i in range(D)]
    B, chosen_combi = sys_matrix(coords)
    # thanks to the last row we added in the equation to set the last coefficient to 1
    C = np.linalg.inv(B)[:, -1]
    sym_row = []
    for i in range(len(chosen_combi)):
        sym_row.append(evaluate(X, chosen_combi[i]))
    expr = np.dot(np.array(sym_row), C)
    expr = sm.expand(expr)
    return expr


def eval_polynomial(coords, poly):
    '''
    Variables :
        coords (list) : list of the coordinates of various points
        poly : expression of a polynom using symbols thanks to sympy
    Returns :
        list of the evaluations of poly at the given coordinates
    '''
    res = []
    for p in coords:
        res.append(poly.subs([("x%i" % i, p[i])
                   for i in range(len(coords[0]))]))
    return res


if __name__ == "__main__":
    # coords = [(0, 1, 2), (0, 2, 1), (0, 2, 3)]
    # B, chosen_combi = sys_matrix(coords)
    # print("B :")
    # print(B)
    # print("chosen combination :")
    # print(chosen_combi)
    # P = set_polynomial_matrix(coords)

    def test():
        coords = []
        max = 20
        for i in range(max):
            coords.append((np.cos(i/max*2*np.pi), np.sin(i/max*2*np.pi)))

        N = 1000
        points = [(rd.uniform(-1, 1), rd.uniform(-1, 1)) for i in range(N)]

        debut = time()
        expr3 = set_polynomial_matrix(coords)
        fin_set = time()
        l_res = eval_polynomial(points, expr3)
        fin_evaluate = time()

        print("Solution matricielle : ")
        print("Temps d'obtention des coefficients : ", fin_set - debut)
        print("Temps de calcul pour N points : ", fin_evaluate - fin_set)
        print("Evaluation en chaque point : ")
        # print(l_res)
    test()
