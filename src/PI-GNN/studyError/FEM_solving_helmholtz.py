# -*- coding: utf-8 -*-
"""
..warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""


# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

# MRG packages
import zsolutions4students as solutions
import utils


WAVENUMBER = 10
VMIN, VMAX = 0.0, 1.0
XMIN, XMAX, YMIN, YMAX = VMIN, VMAX, VMIN, VMAX

# ..todo: Uncomment for displaying limited digits
# numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_errors_L1_L2_helmholtz_dddd(nelemsx=10, nelemsy=10, verbosity=0):
    print("    --- get_errors_L1_L2_helmholtz_dddd ---")

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        XMIN, XMAX, YMIN, YMAX, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    if verbosity:
        fig = matplotlib.pyplot.figure(1)
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        solutions._plot_mesh(p_elem2nodes, elem2nodes,
                             node_coords, color='orange')
        matplotlib.pyplot.show()

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate(
        (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # nodes_on_boundary = ...
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*WAVENUMBER*(x+y))
        laplacian_of_solexact[i] = -2*(WAVENUMBER**2) * solexact[i]

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (WAVENUMBER ** 2) * solexact[i]

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - WAVENUMBER**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    solappro = scipy.linalg.solve(A, B)

    # -- get error
    print("      solappro.shape:", solappro.shape)
    print("      solexact.shape:", solexact.shape)

    solerr_normalized_L1 = utils.get_normalized_norm_L1(solappro - solexact)
    solerr_normalized_L2 = utils.get_normalized_norm_L2(solappro - solexact)

    return solerr_normalized_L1, solerr_normalized_L2


def get_alpha_error():
    h_values = numpy.array([.1/deno for deno in [1, 2, 4, 8]])
    nelems_values = (1/h_values).astype(int)
    solerr_normalized_L1_values = numpy.zeros(h_values.shape[0])
    solerr_normalized_L2_values = numpy.zeros(h_values.shape[0])
    for i, nelems in enumerate(nelems_values):
        print(f"study error for h={h_values[i]}")
        time_start = time.time()
        solerr_normalized_L1, solerr_normalized_L2 = get_errors_L1_L2_helmholtz_dddd(
            nelemsx=nelems, nelemsy=nelems)
        solerr_normalized_L1_values[i] = solerr_normalized_L1
        solerr_normalized_L2_values[i] = solerr_normalized_L2
        time_end = time.time()
        print(f"    took: {time_end-time_start} s")

    # -- analyse L1 error
    lin_reg_res = scipy.stats.linregress(
        numpy.log(h_values), numpy.log(solerr_normalized_L1_values))
    print('Linear regression results MAE:')
    print(r'with $MAE = C h^{\alpha}$')
    print('C = ', numpy.exp(lin_reg_res.intercept))
    print(r'$\alpha$ =', lin_reg_res.slope, '\n')
    matplotlib.pyplot.plot(numpy.log(h_values),
                           numpy.log(solerr_normalized_L1_values), label='log(MAE)')
    matplotlib.pyplot.plot(numpy.log(
        h_values), lin_reg_res.slope*numpy.log(h_values)+lin_reg_res.intercept, label='regression')
    matplotlib.pyplot.title(
        f"Study of MAE in function of h for Helmholtz equation:\n C={numpy.exp(lin_reg_res.intercept)}, alpha={lin_reg_res.slope}")
    matplotlib.pyplot.legend()
    dst_file_path = "./results/study_MAE_h_Helmholtz"
    matplotlib.pyplot.savefig(dst_file_path)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    # -- analyse L2 error
    lin_reg_res = scipy.stats.linregress(
        numpy.log(h_values), numpy.log(solerr_normalized_L2_values))
    print('Linear regression results MSE:')
    print(r'with $MSE = C h^{\alpha}$')
    print('C = ', numpy.exp(lin_reg_res.intercept))
    print(r'$\alpha$ =', lin_reg_res.slope)
    matplotlib.pyplot.plot(numpy.log(h_values),
                           numpy.log(solerr_normalized_L2_values), label="log(MSE)")
    matplotlib.pyplot.plot(numpy.log(
        h_values), lin_reg_res.slope*numpy.log(h_values)+lin_reg_res.intercept, label='regression')
    matplotlib.pyplot.title(
        f"Study of MSE in function of h for Helmholtz equation:\n C={numpy.exp(lin_reg_res.intercept)}, alpha={lin_reg_res.slope}")
    matplotlib.pyplot.legend()
    dst_file_path = "./results/study_MSE_h_Helmholtz"
    matplotlib.pyplot.savefig(dst_file_path)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


if __name__ == '__main__':

    # print(get_errors_L1_L2_helmholtz_dddd(10, 10))
    get_alpha_error()
    print('End.')
