import sympy as sm
import numpy as np
import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

def _set_lagrange(axis_coords):
    '''
    Variable:
        axis_coords (array): array of integers of shape (_,) or (,_)
                        ex : _set_coords_circle[:,0] convenes
    Return:
        L_list (list): list of Lagrange polynome related to axis coords with L_list[k] = $L_k$
    '''
    return expr

def _set_A_numpy_lagrange(coords, G):
    '''
    Variables :
        coords (array) : array of the coordinates where the futur polynome is evaluated in the system of equations
                        ex : coords[i,0] is the first coordonate of point number i
        G (numpy.array): column array of the condition values
    Returns :
        a polynome in a symbolic way thanks to sympy, such that poly(coords[i]) = G[i]
    '''
    expr = 0

    expr = sm.lambdify(X, expr, 'numpy')
    return expr
