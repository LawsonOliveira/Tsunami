import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))
import sympy as sm
import numpy as np


from time import time
from Polynomials.polynome4students_v2 import _set_coords_circle, _eval_polynome_numpy

import matplotlib.pyplot as plt

'''
DOES NOT work with sympy
case disjunction with symbolic expression
'''


def phi(x, a, b):
    if x >= b or x <= a:
        return 0
    else:
        return np.exp(-1/((x-a)*(b-x)))


def norm_phi(x, a, b):
    center_image = phi((b+a)/2, a, b)
    return phi(x, a, b)/center_image

# X = np.linspace(-2, 2, 1000)
# Y = [phi(x, -1, 1) for x in X]
# Y1 = [norm_phi(x, -1, 1) for x in X]
# plt.plot(X, Y)
# plt.plot(X, Y1)
# plt.show()


def get_min_dist(axis_coords):
    '''
    Variable:
        axis_coords (array): array of integers of shape (_,) or (,_)
                        ex : _set_coords_circle[:,0] convenes
    Return :
        d_min (int): minimum difference between two elements of X_coords
    '''
    d_min = np.inf
    L = sorted(axis_coords)
    for i in range(len(L)-1):
        d = L[i+1]-L[i]
        if d < d_min:
            d_min = d
    return d_min


def _set_A_numpy_expproduct(coords, G):
    '''
    Variables :
        coords (array) : array of the coordinates where the futur polynome is evaluated in the system of equations
                        ex : coords[i,0] is the first coordonate of point number i
        G (numpy.array): column array of the condition values
    Returns :
        a polynome in a symbolic way thanks to sympy, such that poly(coords[i]) = G[i]
    '''
    D = coords.shape[1]

    Delta = []
    for j in range(D):
        Delta.append(get_min_dist(coords[:, j])/2)

    X = [sm.symbols('x%i' % i) for i in range(D)]
    expr = 0
    # construct a term and sum it with the previous expr
    for i in range(G.shape[0]):
        term = 1
        for j in range(D):
            delta = Delta[j]
            term *= norm_phi(X[j], coords[i, j]-delta, coords[i, j]+delta)
        expr += G[i]*term
    expr = sm.lambdify(X, expr, 'numpy')
    return expr


if __name__ == '__main__':
    nnodes_max = 10
    coords = _set_coords_circle(nnodes_max)

    G = np.ones((nnodes_max, 1), dtype=np.float64)

    poly_A = _set_A_numpy_expproduct(coords, G)
    print([_eval_polynome_numpy(poly_A, coords[i, 0], coords[i, 1])
          for i in range(nnodes_max)])
