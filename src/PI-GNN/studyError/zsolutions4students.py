# -*- coding: utf-8 -*-
"""
.. warning:: This file contains some functions to ease the programming
of the exercises. This file also contains some solutions of the exercises.

Except otherwise mentioned, you should not open and read this file!
"""


# Python packages
import matplotlib
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys


VTK_NTYPES = 15
VTK_VERTEX = 1
VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_PIXEL = 8
VTK_QUAD = 9
VTK_TETRA = 10
VTK_QUADRATIC_TETRA = 24
VTK_HEXAHEDRON = 12


def _set_square_trimesh(xmin, xmax, ymin, ymax, nx, ny):

    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    nodes_per_elem = 3
    nelems = nx * ny * 2
    p_elem2nodes = numpy.empty((nelems + 1,), dtype=numpy.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = numpy.empty((nelems * nodes_per_elem,), dtype=numpy.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = j * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
            k += nodes_per_elem
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = (j + 1) * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i
            k += nodes_per_elem
    # elem_type = numpy.empty((nelems,), dtype=numpy.int64)
    # elem_type[:] = VTK_TRIANGLE

    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.0
            k += 1

    # local to global numbering
    node_l2g = numpy.arange(0, nnodes, 1, dtype=numpy.int64)

    return node_coords, p_elem2nodes, elem2nodes, node_l2g


def _set_square_quadmesh(xmin, xmax, ymin, ymax, nx, ny):

    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    nodes_per_elem = 4
    nelems = nx * ny
    p_elem2nodes = numpy.empty((nelems + 1,), dtype=numpy.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = numpy.empty((nelems * nodes_per_elem,), dtype=numpy.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = j * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
            elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
            k += nodes_per_elem
    # elem_type = numpy.empty((nelems,), dtype=numpy.int64)
    # elem_type[:] = VTK_TRIANGLE

    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.0
            k += 1

    # local to global numbering
    node_l2g = numpy.arange(0, nnodes, 1, dtype=numpy.int64)

    return node_coords, p_elem2nodes, elem2nodes, node_l2g


def _set_square_nodes_boundary(nx, ny):

    myl2gs = [[]] * 4
    for i in range(0, nx + 1):
        myl2gs[0].append(i)
    for j in range(0, ny + 1):
        myl2gs[1].append((nx + 1) * j - 1 + nx + 1)
    for i in range(0, nx + 1):
        myl2gs[2].append(ny * (nx + 1) + i)
    for j in range(0, ny + 1):
        myl2gs[3].append(j * (nx + 1))
    list_nodes = numpy.unique(numpy.concatenate((myl2gs[0], myl2gs[1], myl2gs[2], myl2gs[3])), )

    return list_nodes


def _set_square_nodes_boundary_south(node_coords):

    nodes = []
    nnodes = node_coords.shape[0]
    for i in range(0, nnodes):
        if node_coords[i, 1] == 0.0:
            nodes.append(i)

    return nodes


def _set_square_nodes_boundary_east(node_coords):

    nodes = []
    nnodes = node_coords.shape[0]
    for i in range(0, nnodes):
        if node_coords[i, 0] == 1.0:
            nodes.append(i)

    return nodes


def _set_square_nodes_boundary_north(node_coords):

    nodes = []
    nnodes = node_coords.shape[0]
    for i in range(0, nnodes):
        if node_coords[i, 1] == 1.0:
            nodes.append(i)

    return nodes


def _set_square_nodes_boundary_west(node_coords):

    nodes = []
    nnodes = node_coords.shape[0]
    for i in range(0, nnodes):
        if node_coords[i, 0] == 0.0:
            nodes.append(i)

    return nodes


def _set_trifem_elementary_stiffness(node_coords, area):

    mat_e = numpy.zeros((3, 3), dtype=numpy.float64)
    for i in range(0, 3):
        for j in range(0, 3):
            mat_e[i, j] = (node_coords[(j + 1) % 3, 1] - node_coords[(j + 2) % 3, 1]) * \
                       (node_coords[(i + 1) % 3, 1] - node_coords[(i + 2) % 3, 1]) + \
                       (node_coords[(j + 2) % 3, 0] - node_coords[(j + 1) % 3, 0]) * \
                       (node_coords[(i + 2) % 3, 0] - node_coords[(i + 1) % 3, 0])
    mat_e *= 0.25 / area

    return mat_e


def _set_trifem_elementary_mass(node_coords, area):

    mat_e = numpy.ones((3, 3), dtype=numpy.float64)
    mat_e *= area / 12.0
    mat_e[range(3), range(3)] *= 2.0

    return mat_e


def _set_trifem_elementary_term(node_coords, area, f=(0.0, 0.0, 0.0)):

    mat_e = _set_trifem_elementary_mass(node_coords, area)
    vec_e = mat_e.dot(f)

    return vec_e


def _set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k=0, coef_m=0):

    nnodes = numpy.shape(node_coords)[0]
    nelems = numpy.shape(p_elem2nodes)[0] - 1
    K = numpy.zeros((nnodes, nnodes), dtype=numpy.complex128)
    M = numpy.zeros((nnodes, nnodes), dtype=numpy.complex128)
    F = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for e in range(0, nelems):
        nodes = elem2nodes[p_elem2nodes[e]:p_elem2nodes[e + 1]]
        coords = node_coords[nodes]
        if len(nodes) == 3:
            Pe = numpy.ones((3, 3), dtype=numpy.float64)
            Pe[:, 1:3] = coords[:, 0:2]
            area = abs(numpy.linalg.det(Pe)) / 2.0
            Ke = _set_trifem_elementary_stiffness(coords, area)
            Me = _set_trifem_elementary_mass(coords, area)
            f = [0.0, 0.0, 0.0]
            f[0] = f_unassembled[nodes[0]]
            f[1] = f_unassembled[nodes[1]]
            f[2] = f_unassembled[nodes[2]]
            Fe = _set_trifem_elementary_term(coords, area, f)
            for i in range(0, 3):
                F[nodes[i]] = F[nodes[i]] + Fe[i]
                for j in range(0, 3):
                    K[nodes[i], nodes[j]] = K[nodes[i], nodes[j]] + coef_k[e] * Ke[i, j]
                    M[nodes[i], nodes[j]] = M[nodes[i], nodes[j]] + coef_m[e] * Me[i, j]

    return K, M, F


def _set_dirichlet_condition(nodes, values, mat, vec):
    """
    nodes = p indices
    values = bp

    ( Aii Aip ) (xi) = (bi)
    ( Api App ) (xp)   (bp)
    <=>
    Aii xi = bi - Aip bp
        xp = bp
    """
    temp = numpy.dot(mat[:, nodes], values[nodes])  # Aip bp; App bp
    values = values.reshape(values.size,1)
    vec[:] = vec[:] - temp.reshape((temp.size,1)) # bi = bi - Aip bp; bp = bp - App bp
    vec[nodes] = values[nodes]  # bp = bp

    mat[nodes, :] = 0.0  # Api App
    mat[:, nodes] = 0.0  # Aip App
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            if (nodes[i] == nodes[j]):
                mat[nodes[i], nodes[j]] = 1.0  # App = 0
            else:
                mat[nodes[i], nodes[j]] = 0.0  # Aip = 0; Api = 0

    return mat, vec


def _plot_node(p_elem2nodes, elem2nodes, node_coords, node, color='red', marker='o'):

    matplotlib.pyplot.plot(node_coords[node, 0], node_coords[node, 1], color=color, marker=marker)

    return


def _plot_elem(p_elem2nodes, elem2nodes, node_coords, elem, color='red'):

    xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
    if xyz.shape[0] == 3:
        matplotlib.pyplot.plot((xyz[0,0], xyz[1,0], xyz[2,0], xyz[0,0]), (xyz[0,1], xyz[1,1], xyz[2,1], xyz[0,1]), color=color)
    elif xyz.shape[0] == 4:
        matplotlib.pyplot.plot((xyz[0,0], xyz[1,0], xyz[2,0], xyz[3,0], xyz[0,0]), (xyz[0,1], xyz[1,1], xyz[2,1], xyz[3,1], xyz[0,1]), color=color)
    elif xyz.shape[0] == 2:
        matplotlib.pyplot.plot((xyz[0,0], xyz[1,0], xyz[0,0]), (xyz[0,1], xyz[1,1], xyz[0,1]), color=color)

    return


def _plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='blue'):

    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    nnodes = numpy.shape(node_coords)[0]
    nelems = numpy.shape(p_elem2nodes)[0]
    for elem in range(0, nelems-1):
        xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
        if xyz.shape[0] == 3:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
        elif xyz.shape[0] == 4:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)
        elif xyz.shape[0] == 2:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[0, 1]), color=color)

    return


def _plot_trifield(nelemsx, nelemsy, sol):

    n_max, m_max = nelemsx+1, nelemsy+1
    x = numpy.linspace(0, 1, n_max)
    y = numpy.linspace(0, 1, m_max)
    X, Y = numpy.meshgrid(x, y)

    Z = numpy.real(sol.reshape((n_max, m_max)))
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.imshow(Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet') # 'RdGy')
    v1 = numpy.linspace(Z.min(), Z.max(), 8, endpoint=True)
    cb = matplotlib.pyplot.colorbar(ticks=v1)
    cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    Z = numpy.imag(sol.reshape((n_max, m_max)))
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.imshow(Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet') # 'RdGy')
    v1 = numpy.linspace(Z.min(), Z.max(), 8, endpoint=True)
    cb = matplotlib.pyplot.colorbar(ticks=v1)
    cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    # matplotlib.pyplot.contourf(X, Y, Z, 20, alpha=0.9, cmap='jet') # 'RdGy')
    # matplotlib.pyplot.colorbar()
    # matplotlib.pyplot.show()
    # matplotlib.pyplot.close()

    return


def _plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, node_data, **kwargs):
    """Plot node data parameter on the mesh.
    """
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    z = node_data

    # create triangles for triangulation
    triangles = []
    # print(nelems)
    for i in range(nelems):
        cs = p_elem2nodes[i]
        ce = p_elem2nodes[i + 1]
        triangles.append(elem2nodes[cs:ce])
        # print(triangles[i])
        #triangles[i,:] = elem2nodes[cs:ce].tolist()

    # creates triangulation
    triang = matplotlib.tri.Triangulation(x, y, triangles)

    fig = matplotlib.pyplot.figure()
    axs = fig.add_subplot(projection='3d')
    # axs.scatter(x, y, z, c=z, cmap='viridis')
    axs.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');
    matplotlib.pyplot.show()

    return


def _plot_trielem(p_elem2nodes, elem2nodes, node_coords, elem):

    xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
    matplotlib.pyplot.plot((xyz[0,0], xyz[1,0], xyz[2,0], xyz[0,0]), (xyz[0,1], xyz[1,1], xyz[2,1], xyz[0,1]), color='red')

    return


def _plot_eigvec_fem2d(nelemsx, nelemsy, eig_val, eig_vec, numb_modes, multiplicity):

    n = eig_val.shape[0]
    n_max, m_max = nelemsx+1, nelemsy+1
    x = numpy.linspace(0, 1, n_max)
    y = numpy.linspace(0, 1, m_max)
    X, Y = numpy.meshgrid(x, y)

    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.scatter(numpy.real(eig_val), numpy.imag(eig_val))
    matplotlib.pyplot.show()

    for j in range(0, n):
        Z = numpy.real(eig_vec[:, j].reshape((n_max, m_max)))
        fig = matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        matplotlib.pyplot.imshow(Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet') # 'RdGy')
        v1 = numpy.linspace(Z.min(), Z.max(), 8, endpoint=True)
        cb = matplotlib.pyplot.colorbar(ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        #matplotlib.pyplot.show()
        filename = 'xxxfig_' + 'FEM2D' + str(multiplicity) + '_eigvec_imshow_' + str(j) + '.jpg'
        matplotlib.pyplot.savefig(filename)
        matplotlib.pyplot.close()
        matplotlib.pyplot.contourf(X, Y, Z, 20, alpha=0.9, cmap='jet') # 'RdGy')
        matplotlib.pyplot.colorbar()
        #matplotlib.pyplot.show()
        # filename = 'xxxfig_' + 'FEM2D' + str(multiplicity) + '_eigvec_contourf_' + str(j) + '.jpg'
        matplotlib.pyplot.savefig(filename)
        matplotlib.pyplot.close()

    return
