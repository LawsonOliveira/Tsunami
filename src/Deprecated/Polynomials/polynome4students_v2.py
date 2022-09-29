import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# .. note:: Validated 2022.02.01

from Polynomials.polynomial_matrix_lambdify import _set_polynome_xpy_numpy_matrix, _set_polynome_sinxpy_numpy_matrix, _set_polynome_expxpy_numpy_matrix
from time import time
import sympy as sm
import numpy



__copyright__ = '(c) Magoules Research Group, 1996. All Rights Reserved.'
__date__ = '0000-00-00'
__version__ = '0.0.9'


# Python packages
numpy.set_printoptions(threshold=sys.maxsize)


def _set_polynome_xpy_real(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((x - coords[i, 0])**2 + (y - coords[i, 1])**2)
    #expr = sm.expand(expr)
    return expr


def _set_polynome_xpy_numpy_real(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((x - coords[i, 0])**2 + (y - coords[i, 1])**2)
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_sinxpy_numpy_real(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((sm.sin(x - coords[i, 0]))
                       ** 2 + (sm.sin(y - coords[i, 1]))**2)
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_expxpy_numpy_real(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * \
            (1.0 - sm.exp(- 10.0*(x - coords[i, 0])
             ** 2 - 10.0*(y - coords[i, 1])**2))
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_xpy_cmplx(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(len(coords)):
        expr = expr * (x - coords[i, 0] + 1j*(y - coords[i, 1]))
    #expr = sm.expand(expr)
    return expr


def _set_polynome_xpy_numpy_cmplx(coords):
    x, y = sm.symbols("x, y")
    expr = 1
    for i in range(len(coords)):
        expr = expr * (x - coords[i, 0] + 1j*(y - coords[i, 1]))
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _eval_polynome(expr, xi, yi):
    x, y = sm.symbols("x, y")
    temp = expr.evalf(subs={x: xi, y: yi})
    return temp


def _eval_polynome_numpy(expr, xi, yi):
    temp = expr(xi, yi)
    return temp


def _set_coords_circle(nnodes_max=20):
    coords = numpy.empty((nnodes_max, 2), dtype=numpy.float64)
    for i in range(0, nnodes_max):
        coords[i, 0] = numpy.cos((i/nnodes_max)*2*numpy.pi)
        coords[i, 1] = numpy.sin((i/nnodes_max)*2*numpy.pi)
    return coords


def _set_coords_circle_concat(nnodes_max=20):
    # Second method (not so useful)
    theta = numpy.linspace(0, 2*numpy.pi, nnodes_max,
                           endpoint=False, dtype=numpy.float64)
    X = numpy.cos(theta).reshape((nnodes_max, 1))
    Y = numpy.sin(theta).reshape((nnodes_max, 1))
    coords = numpy.concatenate((X, Y), axis=1)
    return coords


def _set_coords_square(nnodes_max=20):
    import random
    coords = numpy.empty((nnodes_max, 2), dtype=numpy.float64)
    for i in range(0, nnodes_max):
        coords[i, 0] = random.uniform(-1, 1)
        coords[i, 1] = random.uniform(-1, 1)
    return coords


def _set_coords_rectangle(l, L, nnodes_max=20):
    coords = []
    elements = numpy.linspace(0, 2*l+2*L, nnodes_max+1, endpoint=False)
    for el in elements:
        if el < l:
            coords.append([el, 0])
        elif el < l+L:
            coords.append([l, el - l])
        elif el < 2*l + L:
            coords.append([l-(el - (l+L)), L])
        else:
            coords.append([0, L-(el - (2*l+L))])
    return numpy.array(coords)


def _set_bnd_dirichlet(coords):
    # ..todo:: to do later
    bnd_dirichlet = numpy.empty(coords.shape[0], dtype=numpy.float64)
    # **********
    bnd_dirichlet[:] = 0.0
    return bnd_dirichlet


def _run_eval_polynome():

    time_start = time()
    coords = _set_coords_circle(20)
    time_end = time()
    print("coords.shape: ", coords.shape)
    print("_set_coords: ", time_end - time_start)

    time_start = time()
    coords_v2 = _set_coords_circle_concat(20)
    time_end = time()
    print("coords_v2.shape: ", coords.shape)
    print("_set_coords_v2: ", time_end - time_start)

    time_start = time()
    points = _set_coords_square(500)
    time_end = time()
    print("points.shape: ", points.shape)
    print("_set_points: ", time_end - time_start)

    # Set
    # -- polynome with real coefficients
    time_start = time()
    expr_xpy_real = _set_polynome_xpy_real(coords)
    time_end = time()
    print("_set_polynome_xpy_real: ", time_end - time_start)

    time_start = time()
    expr_xpy_numpy_real = _set_polynome_xpy_numpy_real(coords)
    time_end = time()
    print("_set_polynome_xpy_numpy_real: ", time_end - time_start)

    # -- polynome with complex coefficients
    time_start = time()
    expr_xpy_cmplx = _set_polynome_xpy_cmplx(coords)
    time_end = time()
    print("_set_polynome_xpy_cmplx: ", time_end - time_start)

    time_start = time()
    expr_xpy_numpy_cmplx = _set_polynome_xpy_numpy_cmplx(coords)
    time_end = time()
    print("_set_polynome_xpy_numpy_cmplx: ", time_end - time_start)

    time_start = time()
    expr_sinxpy_numpy_real = _set_polynome_sinxpy_numpy_real(coords)
    time_end = time()
    print("_set_polynome_sinxpy_numpy_real: ", time_end - time_start)

    time_start = time()
    expr_expxpy_numpy_real = _set_polynome_expxpy_numpy_real(coords)
    time_end = time()
    print("_set_polynome_expxpy_numpy_real: ", time_end - time_start)

    # -- polynome obtained thanks to a matrix method
    time_start = time()
    expr_xpy_numpy_matrix = _set_polynome_xpy_numpy_matrix(coords)
    time_end = time()
    print("_set_polynome_xpy_numpy_matrix: ", time_end - time_start)

    time_start = time()
    expr_sinxpy_numpy_matrix = _set_polynome_sinxpy_numpy_matrix(coords)
    time_end = time()
    print("_set_polynome_sinxpy_numpy_matrix: ", time_end - time_start)

    time_start = time()
    expr_expxpy_numpy_matrix = _set_polynome_expxpy_numpy_matrix(coords)
    time_end = time()
    print("_set_polynome_expxpy_numpy_matrix: ", time_end - time_start)

    # Evaluation
    # -- polynome with real coefficients
    time_start = time()
    val_xpy_real = numpy.empty(points.shape[0])
    for i in range(0, points.shape[0]):
        xi, yi = points[i, 0], points[i, 1]
        val_xpy_real[i] = _eval_polynome(expr_xpy_real, xi, yi)
        print(val_xpy_real[i])
        time_end = time()
    print("_eval_polynome (xpy_real): ", time_end - time_start)

    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i, 0], points[i, 1]
        temp = _eval_polynome_numpy(expr_xpy_numpy_real, xi, yi)
        time_end = time()
    print("_eval_polynome_numpy (xpy_numpy_real): ", time_end - time_start)

    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i, 0], points[i, 1]
        temp = _eval_polynome_numpy(expr_sinxpy_numpy_real, xi, yi)
        time_end = time()
    print("_eval_polynome_numpy (sinxpy_numpy_real): ", time_end - time_start)

    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i, 0], points[i, 1]
        temp = _eval_polynome_numpy(expr_expxpy_numpy_real, xi, yi)
        time_end = time()
    print("_eval_polynome_numpy (expxpy_numpy_real): ", time_end - time_start)

    # -- polynome with complex coefficients
    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i][0], points[i][1]
        temp = _eval_polynome(expr_xpy_cmplx, xi, yi)
    time_end = time()
    print("_eval_polynome (xpy_cmplx): ", time_end - time_start)

    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i][0], points[i][1]
        temp = _eval_polynome_numpy(expr_xpy_numpy_cmplx, xi, yi)
    time_end = time()
    print("_eval_polynome (xpy_numpy_cmplx): ", time_end - time_start)

    # -- polynome obtained thanks to a matrix method
    time_start = time()
    for i in range(0, points.shape[0]):
        xi, yi = points[i][0], points[i][1]
        temp = _eval_polynome_numpy(expr_xpy_numpy_matrix, xi, yi)
    time_end = time()
    print("_eval_polynome (xpy_numpy_matrix): ", time_end - time_start)

    # time_start = time()
    # for i in range(0, points.shape[0]):
    #     xi, yi = points[i][0], points[i][1]
    #     temp = _eval_polynome_numpy(expr_sinxpy_numpy_matrix, xi, yi)
    # time_end = time()
    # print("_eval_polynome (xpy_numpy_matrix): ", time_end - time_start)

    # time_start = time()
    # for i in range(0, points.shape[0]):
    #     xi, yi = points[i][0], points[i][1]
    #     temp = _eval_polynome_numpy(expr_expxpy_numpy_matrix, xi, yi)
    # time_end = time()
    # print("_eval_polynome (xpy_numpy_matrix): ", time_end - time_start)

    print("End")


if __name__ == "__main__":

    _ = _run_eval_polynome()

    print("End")
