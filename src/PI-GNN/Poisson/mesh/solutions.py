# -*- coding: utf-8 -*-
"""
.. warning:: This file contains some functions to ease the programming
of the exercises. This file also contains some solutions of the exercises.

Except otherwise mentioned, you should not open and read this file!
"""


# Python packages
import matplotlib
import numpy as np
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys


def _set_trimesh(xmin, xmax, ymin, ymax, nx, ny):

    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = np.empty((nnodes, spacedim), dtype=np.float64)
    nodes_per_elem = 3
    nelems = nx * ny * 2
    p_elem2nodes = np.empty((nelems + 1,), dtype=np.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = np.empty((nelems * nodes_per_elem,), dtype=np.int64)

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
    # elem_type = np.empty((nelems,), dtype=np.int64)
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
    node_l2g = np.arange(0, nnodes, 1, dtype=np.int64)

    return node_coords, node_l2g, p_elem2nodes, elem2nodes


def _set_quadmesh(xmin, xmax, ymin, ymax, nx, ny):

    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = np.empty((nnodes, spacedim), dtype=np.float64)
    nodes_per_elem = 4
    nelems = nx * ny
    p_elem2nodes = np.empty((nelems + 1,), dtype=np.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = np.empty((nelems * nodes_per_elem,), dtype=np.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = j * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
            elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
            k += nodes_per_elem
    # elem_type = np.empty((nelems,), dtype=np.int64)
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
    node_l2g = np.arange(0, nnodes, 1, dtype=np.int64)

    return node_coords, node_l2g, p_elem2nodes, elem2nodes


def _set_trimesh_boundary(nx, ny):

    myl2gs = [[]] * 4
    for i in range(0, nx + 1):
        myl2gs[0].append(i)
    for j in range(0, ny + 1):
        myl2gs[1].append((nx + 1) * j - 1 + nx + 1)
    for i in range(0, nx + 1):
        myl2gs[2].append(ny * (nx + 1) + i)
    for j in range(0, ny + 1):
        myl2gs[3].append(j * (nx + 1))
    list_nodes = np.unique(np.concatenate(
        (myl2gs[0], myl2gs[1], myl2gs[2], myl2gs[3])), )

    return list_nodes


def _set_trifem_elementary_stiffness(node_coords, area):

    mat_e = np.zeros((3, 3), dtype=np.float64)
    for i in range(0, 3):
        for j in range(0, 3):
            mat_e[i, j] = (node_coords[(j + 1) % 3, 1] - node_coords[(j + 2) % 3, 1]) * \
                (node_coords[(i + 1) % 3, 1] - node_coords[(i + 2) % 3, 1]) + \
                (node_coords[(j + 2) % 3, 0] - node_coords[(j + 1) % 3, 0]) * \
                (node_coords[(i + 2) % 3, 0] -
                 node_coords[(i + 1) % 3, 0])
    mat_e *= 0.25 / area

    return mat_e


def _set_trifem_elementary_mass(node_coords, area):

    mat_e = np.ones((3, 3), dtype=np.float64)
    mat_e *= area / 12.0
    mat_e[range(3), range(3)] *= 2.0

    return mat_e


def _set_trifem_elementary_term(node_coords, area, f=(1.0, 1.0, 1.0)):

    mat_e = _set_trifem_elementary_mass(node_coords, area)
    vec_e = mat_e.dot(f)

    return vec_e


def _set_linefem_elementary_stiffness(node_coords, area):

    mat_e = np.array([[1.0, -1.0], [-1.0, 1.0]], np.float64)
    mat_e /= area

    return mat_e


def _set_linefem_elementary_mass(node_coords, area):

    mat_e = np.array([[1.0/3, 1.0/6], [1.0/6, 1.0/3]], np.float64)
    mat_e *= area

    return mat_e


def _set_linefem_elementary_term(node_coords, area, f=(1.0, 1.0, 1.0)):

    f0 = 2*f[0]+f[1]
    f1 = f[0]+2*f[1]
    vec_e = np.array([f0, f1], dtype=np.float)
    vec_e *= (L/6)

    return vec_e


def _set_trifem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled):

    nnodes = np.shape(node_coords)[0]
    nelems = np.shape(p_elem2nodes)[0] - 1
    K = np.zeros((nnodes, nnodes), dtype=np.float64)
    M = np.zeros((nnodes, nnodes), dtype=np.float64)
    F = np.zeros((nnodes, 1), dtype=np.float64)
    for e in range(0, nelems):
        nodes = elem2nodes[p_elem2nodes[e]:p_elem2nodes[e + 1]]
        coords = node_coords[nodes]
        Pe = np.ones((3, 3), dtype=np.float64)
        Pe[:, 1:3] = coords[:, 0:2]
        area = abs(np.linalg.det(Pe)) / 2.0
        # area = np.linalg.norm(coords[0, :] - coords[1, :])

        Ke = _set_trifem_elementary_stiffness(coords, area)
        Me = _set_trifem_elementary_mass(coords, area)
        f = [0.0, 0.0, 0.0]
        f[0] = f_unassembled[nodes[0]]
        f[1] = f_unassembled[nodes[1]]
        f[2] = f_unassembled[nodes[2]]
        Fe = _set_trifem_elementary_term(node_coords, area, f)
        for i in range(0, 3):
            F[nodes[i]] = F[nodes[i]] + Fe[i]
            for j in range(0, 3):
                K[nodes[i], nodes[j]] = K[nodes[i], nodes[j]] + Ke[i, j]
                M[nodes[i], nodes[j]] = M[nodes[i], nodes[j]] + Me[i, j]

    return K, M, F


def _set_dirichlet_condition(nodes, values, mat, vec):
    '''
    ( Aii Aip ) (xi) = (bi)
    ( Api App ) (xp)   (bp)
    <=>
    Aii xi = bi - Aip pp
        xp = bp
    '''
    temp = np.dot(mat[:, nodes], values[nodes])
    values = values.reshape(values.size, 1)
    vec[:] = vec[:] - temp.reshape((temp.size, 1))
    vec[nodes] = values[nodes]

    mat[nodes, :] = 0.0
    mat[:, nodes] = 0.0
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            if (nodes[i] == nodes[j]):
                mat[nodes[i], nodes[j]] = 1.0
            else:
                mat[nodes[i], nodes[j]] = 0.0

    return mat, vec


def _plot_node(p_elem2nodes, elem2nodes, node_coords, node, color='red', marker='o'):

    matplotlib.pyplot.plot(
        node_coords[node, 0], node_coords[node, 1], color=color, marker=marker)

    return


def _plot_elem(p_elem2nodes, elem2nodes, node_coords, elem, color='red'):

    xyz = node_coords[elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
    if xyz.shape[0] == 3:
        matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]), (
            xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
    else:
        matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]), (
            xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)

    return


def _plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='blue'):

    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    nnodes = np.shape(node_coords)[0]
    nelems = np.shape(p_elem2nodes)[0]
    for elem in range(0, nelems-1):
        xyz = node_coords[elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
        if xyz.shape[0] == 3:
            # triangle
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
        else:
            # quadrangle
            # print("elem2nodes xyz\n",
            #       elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]])
            # print("xyz:\n", xyz)
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)

    return


def _plot_trifield(nelemsx, nelemsy, sol):

    n_max, m_max = nelemsx+1, nelemsy+1
    x = np.linspace(0, 1, n_max)
    y = np.linspace(0, 1, m_max)
    X, Y = np.meshgrid(x, y)

    Z = np.real(sol.reshape((n_max, m_max)))
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.imshow(
        Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet')  # 'RdGy')
    v1 = np.linspace(Z.min(), Z.max(), 8, endpoint=True)
    cb = matplotlib.pyplot.colorbar(ticks=v1)
    cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    Z = np.imag(sol.reshape((n_max, m_max)))
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.imshow(
        Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet')  # 'RdGy')
    v1 = np.linspace(Z.min(), Z.max(), 8, endpoint=True)
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
    """Plot node data parameter on the mesh."""

    if 'dpi' in kwargs and kwargs['dpi']:
        dpi = kwargs['dpi']
    else:
        dpi = 60

    x = node_coords[:, 0]
    y = node_coords[:, 1]
    z = node_data
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)

    # create triangles for triangulation
    triangles = []
    for i in range(nelems):
        cs = p_elem2nodes[i]
        ce = p_elem2nodes[i + 1]
        triangles.append(elem2nodes[cs:ce])  # fast
        # triangles.append(list(mymesh.elem2nodes[cs:ce])) # slow

    # creates triangulation
    triang = matplotlib.tri.Triangulation(x, y, triangles)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    xi, yi = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # creates interpolation
    #    interp_lin = mtri.LinearTriInterpolator(triang, z)
    #    zi_lin = interp_lin(xi, yi)
    #    interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
    #    zi_cubic_geom = interp_cubic_geom(xi, yi)
    interp_cubic_min_E = matplotlib.tri.CubicTriInterpolator(
        triang, z, kind='min_E')
    zi_cubic_min_E = interp_cubic_min_E(xi, yi)

    #    # plot triangulation
    #    fig, axs = matplotlib.pyplot.subplots()
    #    axs.tricontourf(triang, z)
    #    axs.triplot(triang, 'ko-')
    #    axs.set_title('Triangular grid')
    #    axs.set_axis_off()
    #    # plot linear interpolation to quad grid
    #    fig, axs = matplotlib.pyplot.subplots()
    #    axs.contourf(xi, yi, zi_lin)
    #    #axs.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    #    #axs.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    #    axs.set_title("Linear interpolation")
    #    axs.set_axis_off()
    #    # plot cubic interpolation to quad grid, kind=geom
    #    fig, axs = matplotlib.pyplot.subplots()
    #    axs.contourf(xi, yi, zi_cubic_geom)
    #    #axs.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    #    #axs.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    #    axs.set_title("Cubic interpolation,\nkind='geom'")
    #    axs.set_axis_off()
    # plot cubic interpolation to quad grid, kind=min_E
    fig, axs = matplotlib.pyplot.subplots(dpi=dpi)
    axs.axis('equal')
    if True:
        # try:
        # im = axs.contourf(xi, yi, zi_cubic_min_E,
        #               300,
        #               cmap=matplotlib.pyplot.cm.get_cmap('rainbow', 256)
        # )
        if 'title' in kwargs and kwargs['title']:
            title = kwargs['title']
            matplotlib.pyplot.title(title)

        # to fix unique legend
        array_to_visualize = zi_cubic_min_E
        cmap = matplotlib.cm.get_cmap(name='rainbow', lut=256)
        cmap = matplotlib.cm.get_cmap(name='jet')  # , lut=256)

        if 'limits' in kwargs and kwargs['limits']:
            color_min, color_max = kwargs['limits']
            norm = matplotlib.colors.Normalize(color_min, color_max)
        else:
            norm = matplotlib.colors.Normalize()

        mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(array_to_visualize)

        if 'limits' in kwargs and kwargs['limits']:
            pass
        else:
            mappable.autoscale()

        matplotlib.pyplot.colorbar(mappable, ax=matplotlib.pyplot.gca())
        mappable.changed()
        im = axs.contourf(xi, yi, zi_cubic_min_E,
                          300,
                          cmap=cmap,
                          norm=norm
                          )
        matplotlib.pyplot.cool()

        if 'title' in kwargs and kwargs['title']:
            title = kwargs['title']
            matplotlib.pyplot.title(title)

        #
        #         im = axs.contourf(xi, yi, zi_cubic_min_E,
        #                       300,
        #                       cmap=matplotlib.pyplot.cm.get_cmap('rainbow', 256)#, vmin=0, vmax=0.07
        #         )
        #         #                      cmap = matplotlib.pyplot.cm.get_cmap('jet', 256)
        #         #                      cmap=matplotlib.pyplot.cm.get_cmap('rainbow', 60)
        # #        im = axs.pcolormesh(xi, yi, zi_cubic_min_E)
        #        matplotlib.pyplot.cool()
        #        fig.colorbar(im, ax=axs)
        #        axs.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        #        axs.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        #        axs.set_title("Cubic interpolation,\nkind='min_E'")
        #        axs.set_axis_off()

        # create windows
        x_range = x_max - x_min
        y_range = y_max - y_min
        matplotlib.pyplot.xlim(x_min - 0.05 * x_range,
                               x_max + 0.05 * x_range)  # 5% more and less
        matplotlib.pyplot.ylim(y_min - 0.05 * y_range,
                               y_max + 0.05 * y_range)  # 5% more and less

        if 'filename' in kwargs and kwargs['filename']:
            output_file = kwargs['filename']
            (root, ext) = os.path.splitext(output_file)
            # , bbox_inches='tight')
            matplotlib.pyplot.savefig(root + '_contourf' + ext, format=ext[1:])
            matplotlib.pyplot.close()
        else:
            matplotlib.pyplot.show()
            matplotlib.pyplot.close()

        matplotlib.pyplot.close(fig)

    # except:
    #     err_msg = _env._handle_sys_getframe()
    #     err_msg += f'\n  Error: ambiguous functionality\n'
    #     #sys.exit(err_msg)
    #     print(z.dot(z))
    #     sys.stderr.write(err_msg)
    #     pass

    return


def _plot_trielem(p_elem2nodes, elem2nodes, node_coords, elem):

    xyz = node_coords[elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
    matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                           (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color='red')

    return


def _plot_eigvec_fem2d(nelemsx, nelemsy, eig_val, eig_vec, numb_modes, multiplicity):

    n_max, m_max = nelemsx+1, nelemsy+1
    x = np.linspace(0, 1, n_max)
    y = np.linspace(0, 1, m_max)
    X, Y = np.meshgrid(x, y)

    # for m in range(1, m_max):
    #    for n in range(1, n_max):
    for j in range(0, n_max):
        Z = np.real(eig_vec[:, j].reshape((n_max, m_max)))
        fig = matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        matplotlib.pyplot.imshow(
            Z, extent=[0, 1, 0, 1], vmin=Z.min(), vmax=Z.max(), cmap='jet')  # 'RdGy')
        v1 = np.linspace(Z.min(), Z.max(), 8, endpoint=True)
        cb = matplotlib.pyplot.colorbar(ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        # matplotlib.pyplot.show()
        filename = 'xxxfig_' + 'FEM2D' + \
            str(multiplicity) + '_eigvec_imshow_' + str(j) + '.jpg'
        matplotlib.pyplot.savefig(filename)
        matplotlib.pyplot.close()
        matplotlib.pyplot.contourf(
            X, Y, Z, 20, alpha=0.9, cmap='jet')  # 'RdGy')
        matplotlib.pyplot.colorbar()
        # matplotlib.pyplot.show()
        filename = 'xxxfig_' + 'FEM2D' + \
            str(multiplicity) + '_eigvec_contourf_' + str(j) + '.jpg'
        matplotlib.pyplot.savefig(filename)
        matplotlib.pyplot.close()

    return
