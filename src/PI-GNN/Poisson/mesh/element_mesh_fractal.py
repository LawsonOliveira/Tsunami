# Python packages
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mesh.solutions as solutions
import mesh.element_mesh as exo4


def build_node2elems(p_elem2nodes, elem2nodes):
    # elem2nodes connectivity matrix

    e2n_coef = np.ones(len(elem2nodes), dtype=np.int)
    e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))

    # node2elems connectivity matrix
    n2e_mtx = e2n_mtx.transpose()
    n2e_mtx = n2e_mtx.tocsr()

    # output
    p_node2elems = n2e_mtx.indptr
    node2elems = n2e_mtx.indices

    return p_node2elems, node2elems


def get_unordered_boundary(node_coords, p_elem2nodes, elem2nodes):
    dic_count = {}
    l_unordered_boundary2nodes = []
    for nodeid in elem2nodes:
        if nodeid in dic_count:
            dic_count[nodeid] += 1
        else:
            dic_count[nodeid] = 1
    for nodeid, count in dic_count.items():
        if count <= 3:
            l_unordered_boundary2nodes.append(nodeid)
    return np.array(l_unordered_boundary2nodes)




def set_simple_mesh(l_bound=[0.0, 1.0, 0.0, 1.0]):
    '''
    l_bound = [xmin, xmax, ymin, ymax]
    '''
    xmin, xmax, ymin, ymax = l_bound
    nelemsx, nelemsy = 1, 1
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # twice the same node in it at the beginning and the end => easier to plot and use
    # always in anti trigonometric order
    boundary2nodes = np.array([0, 2, 3, 1, 0])
    return node_coords, node_l2g, p_elem2nodes, elem2nodes, boundary2nodes


def test_set_simple_mesh():
    node_coords, node_l2g, p_elem2nodes, elem2nodes, boundary2nodes = set_simple_mesh()
    # print("node_coords:", node_coords)
    solutions._plot_mesh(p_elem2nodes, elem2nodes,
                         node_coords, color='black')
    bar_coords = exo4.compute_barycenter_of_element(
        node_coords, p_elem2nodes, elem2nodes)
    plt.scatter(
        bar_coords[:, 0], bar_coords[:, 1], c='g', label="Barycenter")
    plt.plot(node_coords[boundary2nodes][:, 0],
             node_coords[boundary2nodes][:, 1], label='boundary', c='r')
    plt.legend()
    plt.show()

    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    print("p_elem2nodes", p_elem2nodes)
    print("elem2nodes", elem2nodes)
    print("p_node2elems", p_node2elems)
    print("node2elems", node2elems)

    """
    results:
    p_elem2nodes [0 4]
    elem2nodes [0 1 3 2]
    p_node2elems [0 1 2 3 4]
    node2elems [0 0 0 0]
    """


def check_node_already_exists_in_mesh(node_coords, p_elem2nodes, elem2nodes, node_to_check):
    '''
    Args:
     - ...
     - node_to_check of shape (dim,)
    Returns:
      - bool
    '''
    flag_matrix = np.zeros(node_coords.shape)
    for i in range(node_to_check.shape[-1]):
        flag_matrix[:, i] = np.where(
            node_coords[:, i] == node_to_check[i], 1, 0)

    N_nodes_in_this_coord = np.sum(np.prod(flag_matrix, axis=1))
    assert N_nodes_in_this_coord <= 1, 'some nodes are merged'
    return N_nodes_in_this_coord >= 1


def get_global_ids_nodes(node_coords, p_elem2nodes, elem2nodes, nodes_to_check):
    global_ids = np.zeros(nodes_to_check.shape[0], dtype=int)
    for local_id in range(nodes_to_check.shape[0]):
        flag_matrix = np.zeros(node_coords.shape)
        for i in range(nodes_to_check.shape[-1]):
            flag_matrix[:, i] = np.where(
                node_coords[:, i] == nodes_to_check[local_id, i], 1, 0)
        # next line can output an error if the same node is twice in node_coords
        global_ids[local_id] = int(np.arange(flag_matrix.shape[0])[
            np.prod(flag_matrix, axis=1) == 1])

    return global_ids


def subdivide(node_coords, p_elem2nodes, elem2nodes, i_p_elem2nodes, n_subdivision=4):
    '''
    Args:
     - node_coords, p_elem2nodes, elem2nodes, boundary2nodes
     - elemid : the id of the nodes constituting the element to subdivide
     - n_subdivision : number of elements added on one axis
    returns:
     - node_coords, p_elem2nodes, elem2nodes with the element composed of elemid nodes subdivided
    '''
    assert i_p_elem2nodes < p_elem2nodes.shape[0] - \
        1, "want to subdivide a non-existing elem"
    if n_subdivision == 1:
        return node_coords, p_elem2nodes, elem2nodes

    # find elemid in p_elem2nodes
    # delete element
    elemid = elem2nodes[p_elem2nodes[i_p_elem2nodes]
        :p_elem2nodes[i_p_elem2nodes+1]]
    n_nodes_elemid = elemid.shape[0]
    elem2nodes = np.concatenate(
        [elem2nodes[:p_elem2nodes[i_p_elem2nodes]], elem2nodes[p_elem2nodes[i_p_elem2nodes+1]:]], axis=-1)
    p_elem2nodes = np.concatenate(
        [p_elem2nodes[:i_p_elem2nodes], p_elem2nodes[i_p_elem2nodes+1:]-n_nodes_elemid], axis=-1)

    # --- add n_subdivision**2 elements within previous elemid ---
    vector_edge = (node_coords[elemid[1], :] -
                   node_coords[elemid[0], :])/n_subdivision

    # remark: could put directly in node_coords_one_row
    new_nodes = np.zeros((n_subdivision-1, node_coords.shape[-1]))
    for i in range(0, n_subdivision-1):
        new_nodes[i, :] = node_coords[elemid[0], :]+(i+1)*vector_edge
    node_coords_one_row = np.concatenate(
        [[node_coords[elemid[0], :]], new_nodes, [node_coords[elemid[1], :]]], axis=0)

    elem_nodes_coords = np.zeros(((n_subdivision+1)**2, node_coords.shape[-1]))
    u_move = (node_coords[elemid[-1], :] -
              node_coords[elemid[0], :])/n_subdivision
    u_move = np.expand_dims(u_move, axis=0)
    matrix_move = np.ones((node_coords_one_row.shape[0], 1))@u_move
    for i in range(0, n_subdivision+1):
        # print(i)
        # print("move\n", node_coords_one_row+i*matrix_move,
        #       (node_coords_one_row+i*matrix_move).shape)
        # print("elem_nodes_coords\n", elem_nodes_coords[(n_subdivision+1)*i:(n_subdivision+1) *
        #                                    (i+1), :])
        elem_nodes_coords[(n_subdivision+1)*i:(n_subdivision+1) *
                          (i+1), :] = node_coords_one_row+i*matrix_move

    # print("local_nodes:\n", elem_nodes_coords)

    # add nodes to mesh
    for node_cand_to_add in elem_nodes_coords:
        node_coords, p_elem2nodes, elem2nodes = exo4.add_node_to_mesh(node_coords, p_elem2nodes,
                                                                          elem2nodes, node_cand_to_add, join_to_new_elem=False)
        # remove duplicates 
    node_coords = np.unique(node_coords,axis=0)

    # add elements to mesh
    global_ids = get_global_ids_nodes(
        node_coords, p_elem2nodes, elem2nodes, elem_nodes_coords)
    # print("node_coords\n", node_coords)
    # print("elem_nodes_coords\n", elem_nodes_coords)
    # print("global_ids\n", global_ids)

    for j in range(n_subdivision):
        for i in range(n_subdivision):
            # j handles column movement and i row movement
            # local ids
            new_elemid_nodes = np.array(
                [j+i*(n_subdivision+1), (j+1)+i*(n_subdivision+1),
                 (j+1)+(i+1)*(n_subdivision+1), j+(i+1)*(n_subdivision+1)])
            # global ids
            new_elemid_nodes = global_ids[new_elemid_nodes]
            node_coords, p_elem2nodes, elem2nodes = exo4.add_elem_to_mesh(
                node_coords, p_elem2nodes, elem2nodes, new_elemid_nodes)

    return node_coords, p_elem2nodes, elem2nodes


def find_nodeid_from_nodecoord(node_coords, p_elem2nodes, elem2nodes, node_coord_to_find, tolerance=1e-6):
    node_coord_to_find = np.expand_dims(node_coord_to_find, axis=0)
    Mdist = np.ones((node_coords.shape[0], 1))@node_coord_to_find-node_coords
    Mdist = np.sqrt(Mdist**2@np.ones((node_coords.shape[-1], 1)))
    where_coord_correct = np.where(Mdist <= tolerance, 1, 0)
    indexes = np.arange(node_coords.shape[0])
    nodeid = indexes[where_coord_correct[:, 0] == 1]
    assert nodeid.shape[0] <= 1, (' there is several times the same node in node_coords.\nActual result is : ' +
                                  str(nodeid.shape[0]) +
                                  "\nhere is node_coords:\n", node_coords, "\nhere is node_coord_to_find\n:", node_coord_to_find)
    if nodeid.shape[0] == 0:
        return False, None
    return True, nodeid[0]


def update_boundary(node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision):
    N0_boundary = boundary2nodes.shape[0]
    # boundary_node_coords = get_unordered_boundary(
    #     node_coords, p_elem2nodes, elem2nodes)
    for i in range(N0_boundary-1):
        i = i*n_subdivision
        u_step = (node_coords[boundary2nodes[i+1]] -
                  node_coords[boundary2nodes[i]])/n_subdivision
        l_new_nodeids = []
        for j in range(1, n_subdivision):
            new_node_for_boundary_coord = node_coords[boundary2nodes[i]]+j*u_step
            _, new_nodeid = find_nodeid_from_nodecoord(
                node_coords,  p_elem2nodes, elem2nodes, new_node_for_boundary_coord)
            l_new_nodeids.append(new_nodeid)
        boundary2nodes = np.concatenate(
            [boundary2nodes[:i+1], l_new_nodeids, boundary2nodes[i+1:]], axis=-1)
    return boundary2nodes


def is_in_boundary(boundary2nodes, glob_sequence_idnodes):
    for i in range(boundary2nodes.shape[0]-glob_sequence_idnodes.shape[0]+1):
        if exo4.is_same_matrices(boundary2nodes[i:i+glob_sequence_idnodes.shape[0]], glob_sequence_idnodes):
            return i, True
    return None, False


def test_subdivide():
    print("--- test subdivide ---")
    node_coords, node_l2g, p_elem2nodes, elem2nodes, boundary2nodes = set_simple_mesh()
    print('boundary2nodes before:', boundary2nodes)
    print("node_coords\n", node_coords)
    node_coords, p_elem2nodes, elem2nodes, boundary2nodes = subdivide_all(
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=5)
    print('boundary2nodes after:', boundary2nodes)
    print("node_coords\n", node_coords)
    solutions._plot_mesh(p_elem2nodes, elem2nodes,
                         node_coords, color='black')
    bar_coords = exo4.compute_barycenter_of_element(
        node_coords, p_elem2nodes, elem2nodes)

    plot_with_boundary(node_coords, p_elem2nodes, elem2nodes,
                       boundary2nodes, bar_coords, plot_boundary=False,title='Test subdivide all')

    # p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    # print("p_elem2nodes", p_elem2nodes)
    # print("elem2nodes", elem2nodes)
    # print("p_node2elems", p_node2elems)
    # print("node2elems", node2elems)


def subdivide_all(node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=4):
    '''
    Args:
     - node_coords, p_elem2nodes, elem2nodes
     - elemid : the id of the nodes constituting the element to subdivide
     - n_subdivision : number of elements added on one axis
    returns:
     - node_coords, p_elem2nodes, elem2nodes with the element composed of elemid nodes subdivided
    '''
    for _ in range(p_elem2nodes.shape[0]-1):
        node_coords, p_elem2nodes, elem2nodes = subdivide(
            node_coords, p_elem2nodes, elem2nodes, 0, n_subdivision=n_subdivision)

    boundary2nodes = update_boundary(
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision)

    return node_coords, p_elem2nodes, elem2nodes, boundary2nodes


def apply_pattern_on_boundary(node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords, pattern_size=3):
    N0_boundary2nodes = boundary2nodes.shape[0]
    i_start, i_end = 0, pattern_size
    for counter in range(N0_boundary2nodes//pattern_size):
        # print("--- counter", counter)
        # print('coords of first node and last node :',
        #       node_coords[boundary2nodes[i_start]], node_coords[boundary2nodes[i_end]])

        # print("i_start, i_end", i_start, i_end)
        bound_to_modify = boundary2nodes[i_start:i_end+1]
        # print("bound_to_modify", bound_to_modify)
        vector_small_edge = node_coords[bound_to_modify[1]
                                        ]-node_coords[bound_to_modify[0]]
        vector_small_edge = np.expand_dims(vector_small_edge, axis=0)
        # print("vector_small_edge\n", vector_small_edge, vector_small_edge.shape)
        ortho_vector_small_edge = (get_rotationxy_matrix(
            -np.pi/2)@vector_small_edge.T).T

        # print("ortho_vector_small_edge", ortho_vector_small_edge)

        # we do not touch the extremities of the edge
        for i in range(1, bound_to_modify.shape[0]-2):
            if i == 1:
                # to change for more complex patterns
                # here we only extrude one edge
                # get common element of nodes i and i+1
                # useful line when we want to delete an element :
                # i_p_elem2nodes = get_common_elem(
                #     elem2nodes, p_elem2nodes, bound_to_modify[i], bound_to_modify[i+1])

                new_nodes_coords = node_coords[[bound_to_modify[i],
                                                bound_to_modify[i+1]]]
                new_nodes_coords = new_nodes_coords + \
                    np.ones((2, 1))@ortho_vector_small_edge

                # print("new_nodes_coords\n", new_nodes_coords)

                # add nodes to mesh
                nodes_will_merge = False
                for new_node_coords in new_nodes_coords:
                    # print("merge checked")
                    a_node_already_exists_here, _ = find_nodeid_from_nodecoord(
                        node_coords, p_elem2nodes, elem2nodes, new_node_coords, tolerance=1e-6)
                    if a_node_already_exists_here:
                        # print('one merge avoided')
                        nodes_will_merge = True

                if not (nodes_will_merge):
                    for new_node_coords in new_nodes_coords:
                        node_coords, p_elem2nodes, elem2nodes = exo4.add_node_to_mesh(
                            node_coords, p_elem2nodes, elem2nodes, new_node_coords, join_to_new_elem=False)

                    # add new element
                    new_elemid_nodes = np.array([bound_to_modify[i],
                                                 node_coords.shape[0]-2, node_coords.shape[0]-1, bound_to_modify[i+1]])
                    node_coords, p_elem2nodes, elem2nodes = exo4.add_elem_to_mesh(
                        node_coords, p_elem2nodes, elem2nodes, new_elemid_nodes)

                    # update barycenter_coords
                    nodes_of_elem = node_coords[elem2nodes[p_elem2nodes[-2]                                                           :p_elem2nodes[-1]]]
                    Mone = np.ones((1, nodes_of_elem.shape[0]))
                    new_barycenter = 1 / \
                        nodes_of_elem.shape[0]*Mone@nodes_of_elem
                    barycenter_coords = np.concatenate(
                        [barycenter_coords, new_barycenter], axis=0)

                    # update boundary2nodes
                    # add [node_coords.shape[0]-2,node_coords.shape[0]-1] between i_start+1 and i_start+2
                    # print('\n---\n')
                    # print("boundary2nodes before\n", boundary2nodes)
                    boundary2nodes = np.concatenate([boundary2nodes[:i_start+i+1], [
                        node_coords.shape[0]-2, node_coords.shape[0]-1], boundary2nodes[i_start+i+1:]], axis=0)

                    # print("boundary2nodes\n", boundary2nodes)
                    # print("bound_coords\n", node_coords[boundary2nodes])

        i_start += (boundary2nodes.shape[0]-N0_boundary2nodes)+pattern_size
        i_end = i_start+pattern_size
        N0_boundary2nodes = boundary2nodes.shape[0]
        # print()

    return node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords


def get_common_elem(elem2nodes, p_elem2nodes, nodeid1, nodeid2):
    assert p_elem2nodes.shape[0] > 1, "not even one element in p_elem2nodes"
    elem_type_length = p_elem2nodes[1]-p_elem2nodes[0]

    indexes = np.arange(elem2nodes.shape[0])
    where_nodeid1 = np.where(elem2nodes == nodeid1, 1, 0)
    where_nodeid2 = np.where(elem2nodes == nodeid2, 1, 0)
    # print("where_nodeid1", where_nodeid1)
    i_p_elem2nodes_nodeid1 = indexes[where_nodeid1 == 1]//elem_type_length
    i_p_elem2nodes_nodeid2 = indexes[where_nodeid2 == 1]//elem_type_length
    # print("i_p_elem2nodes_nodeid1", i_p_elem2nodes_nodeid1)
    # print("i_p_elem2nodes_nodeid2", i_p_elem2nodes_nodeid2)
    for idx1 in i_p_elem2nodes_nodeid1:
        if idx1 in i_p_elem2nodes_nodeid2:
            return idx1
    # print('the two nodes seem to not share any element')
    return None


def get_rotationxy_matrix(theta):
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    return R


def apply_fractal(node_coords, p_elem2nodes, elem2nodes, boundary2nodes, order=1, pattern_size=3):
    for i in range(order):
        barycenter_coords = exo4.compute_barycenter_of_element(
            node_coords, p_elem2nodes, elem2nodes)
        # plot_with_boundary(node_coords, p_elem2nodes, elem2nodes,
        #                    boundary2nodes, barycenter_coords, title="order number"+str(i))
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes = subdivide_all(
            node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=pattern_size)

        barycenter_coords = exo4.compute_barycenter_of_element(
            node_coords, p_elem2nodes, elem2nodes)
        # plot_with_boundary(node_coords, p_elem2nodes, elem2nodes,
        #                    boundary2nodes, barycenter_coords, title="order number"+str(i))
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords = apply_pattern_on_boundary(
            node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords, pattern_size)

        # print("order", i, "\nboundary2nodes\n", boundary2nodes)
        # print('bound_coords\n', node_coords[boundary2nodes])
    return node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords


def test_apply_fractal(order=1, pattern_size=3):
    print('--- test fractal ---')
    node_coords, node_l2g, p_elem2nodes, elem2nodes, boundary2nodes = set_simple_mesh()
    bar_coords = exo4.compute_barycenter_of_element(
        node_coords, p_elem2nodes, elem2nodes)
    plot_with_boundary(node_coords, p_elem2nodes, elem2nodes,
                       boundary2nodes, bar_coords, plot_barycenter=True, plot_boundary=False, title='before fractal')

    ###
    node_coords, p_elem2nodes, elem2nodes, boundary2nodes, bar_coords = apply_fractal(
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, order=order, pattern_size=pattern_size)
    plot_with_boundary(node_coords, p_elem2nodes, elem2nodes,
                       boundary2nodes, bar_coords, plot_barycenter=False, plot_boundary=False, title='after fractal')


def plot_with_boundary(node_coords, p_elem2nodes, elem2nodes, boundary2nodes, barycenter_coords, plot_barycenter=True, plot_boundary=True, add_label_to_boundary=False, title=''):
    solutions._plot_mesh(p_elem2nodes, elem2nodes,
                         node_coords, color='black')
    if plot_barycenter:
        plt.scatter(
            barycenter_coords[:, 0], barycenter_coords[:, 1], c='g', label="Barycenter")
    if plot_boundary:
        plt.plot(node_coords[boundary2nodes][:, 0],
                 node_coords[boundary2nodes][:, 1], 'ro-', label='boundary')

    if plot_boundary and add_label_to_boundary:
        # zip joins x and y coordinates in pairs
        for label, (x, y) in enumerate(zip(node_coords[boundary2nodes][:, 0], node_coords[boundary2nodes][:, 1])):
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 0),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_set_simple_mesh()
    test_subdivide()
    # test_apply_fractal(order=3, pattern_size=3)

    # node_coords, node_l2g, p_elem2nodes, elem2nodes, boundary2nodes = set_simple_mesh()
    # print("node_coords\n", node_coords)
    # print('boundary2nodes before:', boundary2nodes)
    # node_coords, p_elem2nodes, elem2nodes, boundary2nodes = subdivide(
    #     node_coords, p_elem2nodes, elem2nodes, boundary2nodes, 0, n_subdivision=2)
    # print('boundary2nodes after:', boundary2nodes)
    # print("node_coords\n", node_coords,
    #       "\nnumber of nodes:", node_coords.shape[0])
    # print("")
    # print("second subdivision")
    # node_coords, p_elem2nodes, elem2nodes, boundary2nodes = subdivide(
    #     node_coords, p_elem2nodes, elem2nodes, boundary2nodes, 0, n_subdivision=2, do_update_boundary=False)
    # print('boundary2nodes after:', boundary2nodes)
    # print("node_coords\n", node_coords,
    #       "\nnumber of nodes:", node_coords.shape[0])
    # node_coords, p_elem2nodes, elem2nodes, boundary2nodes = subdivide(
    #     node_coords, p_elem2nodes, elem2nodes, boundary2nodes, 1, n_subdivision=2, do_update_boundary=False)
    # print('boundary2nodes after:', boundary2nodes)
    # print("node_coords\n", node_coords,
    #       "\nnumber of nodes:", node_coords.shape[0])
