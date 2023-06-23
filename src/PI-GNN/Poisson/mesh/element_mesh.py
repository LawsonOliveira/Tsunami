# -*- coding: utf-8 -*-

# Python packages
import numpy as np



def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, new_elemid_nodes):
    elem2nodes = np.concatenate(
        (elem2nodes, new_elemid_nodes), axis=-1)
    p_elem2nodes = np.concatenate(
        (p_elem2nodes, [p_elem2nodes[-1]+new_elemid_nodes.shape[-1]]), axis=-1)
    return node_coords, p_elem2nodes, elem2nodes


def get_id_n_nearest_nodes(node_coords, new_node_coords, n=2):
    assert node_coords.shape[0] >= n
    assert new_node_coords.shape[-1] == new_node_coords.shape[-1], "not the same dimension of nodes"
    Mdist = np.ones((node_coords.shape[0], 1))@new_node_coords-node_coords
    Mdist = Mdist**2@np.ones((node_coords.shape[-1], 1))
    # print("Mdist:", Mdist[:, 0], "shape:", Mdist.shape)
    # print("nearest nodes id:", np.argpartition(Mdist[:, 0], n)[:n])
    return np.argpartition(Mdist[:, 0], n)[:n]


def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, new_node_coords, join_to_new_elem=True):
    '''
    new_node_coords (array of shape=(dim,))
    link this node to the 2 closest nodes of node_coords
    '''
    n_nodes, space_dim = node_coords.shape
    assert new_node_coords.shape[-1] == space_dim, (
        "wrong dimension of new_node: ", new_node_coords.shape, 'while space_dim =', space_dim)

    if len(new_node_coords.shape) == 1:
        new_node_coords = np.expand_dims(new_node_coords, axis=0)

    node_coords = np.concatenate((node_coords, new_node_coords), axis=0)

    
    # print("new_elem_node:", new_elem_node)
    if join_to_new_elem:
        id_2nearest_nodes = get_id_n_nearest_nodes(node_coords, new_node_coords)  # , is_inside_element, elem_included_in
        new_elem_node = np.concatenate((id_2nearest_nodes, np.array([n_nodes])), axis=-1)
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, new_elem_node)

    return node_coords, p_elem2nodes, elem2nodes


def is_same_matrices(A, B):
    if A.shape != B.shape:
        print("comparison between 2 matrices of different shape :\n",
              A.shape, 'vs', B.shape)
        return False
    Aflat, Bflat = np.ndarray.flatten(A), np.ndarray.flatten(B)
    for i, a in enumerate(Aflat):
        if Bflat[i] != a:
            return False
    return True


def remove_orphans(node_coords, p_elem2nodes, elem2nodes):
    '''
    use it at the end of all the call of remove_elem_to_mesh and remove_node_to_mesh to clean the mesh
    delete every node not included in an element
    '''

    i = 0
    imax = node_coords.shape[0]-1
    # print("while loop enter: (i,imax) = ", (i, imax))
    while i <= imax:
        if not (i in elem2nodes):
            # orphan found
            # delete it
            # Mfirst = node_coords[:i, :]
            # Mlast = node_coords[i+1:, :]
            node_coords = np.delete(node_coords, i, axis=0)
            # update next nodeids
            elem2nodes = np.where(elem2nodes > i, elem2nodes-1, elem2nodes)
            imax -= 1
            # keep this i as the new node in i was the node in i+1
        else:
            # not an orphan, go to next node
            i += 1
        # print("i:", i)
    return node_coords, p_elem2nodes, elem2nodes


def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid, factorized=False):
    '''
    check where elemid is and delete it after
    elemid : the id of the nodes forming the element to remove
    factorized : bool if True delete orphan nodes, else no deletion of orphan nodes
    '''
    n_nodes_elemid = elemid.shape[-1]
    for i, elem2nodes_idstart in enumerate(p_elem2nodes[:-1]):
        # p_elem2nodes[i+1] is elem2nodes_idend
        n_nodes_cand_elem = p_elem2nodes[i+1]-elem2nodes_idstart
        if n_nodes_cand_elem == n_nodes_elemid:
            elemid_cand = elem2nodes[elem2nodes_idstart:p_elem2nodes[i+1]]
            if is_same_matrices(elemid, elemid_cand):
                elem2nodes = np.concatenate(
                    [elem2nodes[:elem2nodes_idstart], elem2nodes[p_elem2nodes[i+1]:]], axis=-1)
                p_elem2nodes = np.concatenate(
                    [p_elem2nodes[:i], p_elem2nodes[i+1:]-n_nodes_elemid], axis=-1)
                break

    if not factorized:
        # to factorize if this function is called several times
        node_coords, p_elem2nodes, elem2nodes = remove_orphans(
            node_coords, p_elem2nodes, elem2nodes)

    return node_coords, p_elem2nodes, elem2nodes


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid, factorized=False):
    # break elements where nodeid is
    # the delete orphans including nodeid

    idx_elem2nodes = np.arange(elem2nodes.shape[-1])
    # indices in elem2nodes where the value is nodeid
    id_nodeid_elem2nodes = idx_elem2nodes[elem2nodes == nodeid]

    # list of elements to remove described the idx of the nodes that shape it
    l_elemid_to_remove = []

    check_id_nodeid_elem2nodes = 0
    print("Begin to list all the elements linked to this node")
    for i in range(p_elem2nodes.shape[0]-1):
        if (id_nodeid_elem2nodes[check_id_nodeid_elem2nodes] >= p_elem2nodes[i]) and (id_nodeid_elem2nodes[check_id_nodeid_elem2nodes] < p_elem2nodes[i+1]):
            # nodeid belongs to this element
            l_elemid_to_remove.append(
                elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]])
            # check in which element the next id of id_nodeid_elem2nodes is refering to
            # we can do it like this as the elements are sorted :
            # if an id of id_nodeid_elem2nodes is a given element, then the next id of id_nodeid_elem2nodes is an element among the next ones
            if check_id_nodeid_elem2nodes < id_nodeid_elem2nodes.shape[0]-1:
                check_id_nodeid_elem2nodes += 1
            else:
                # every id of id_nodeid_elem2nodes has been checked
                break

    print("Remove these elements")
    # remove the elements
    for elemid in l_elemid_to_remove:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, elemid, factorized=True)

    if not factorized:
        print("Remove orphan nodes")
        node_coords, p_elem2nodes, elem2nodes = remove_orphans(
            node_coords, p_elem2nodes, elem2nodes)

    return node_coords, p_elem2nodes, elem2nodes


def remove_window_to_mesh(node_coords, p_elem2nodes, elem2nodes, window_coords):
    '''
    window_coords (numpy array of shape (2,node_coords.shape[-1])): describes the 2 opposite points of a rectangular parallelepiped
    '''
    Xmin = np.squeeze(np.min(window_coords, axis=1))
    Xmax = np.squeeze(np.max(window_coords, axis=1))
    is_greater = np.where(node_coords >= Xmin, 1, 0)
    is_smaller = np.where(node_coords <= Xmax, 1, 0)
    is_within = is_greater*is_smaller
    id_is_within = np.prod(is_within, axis=1)
    nodeid_within = np.arange(node_coords.shape[0])[id_is_within == 1]

    for nodeid in nodeid_within:
        node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, nodeid, factorized=True)

    print("Remove orphan nodes")
    node_coords, p_elem2nodes, elem2nodes = remove_orphans(
        node_coords, p_elem2nodes, elem2nodes)

    return node_coords, p_elem2nodes, elem2nodes


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):
    elem_coords = None

    for i in range(0, p_elem2nodes.shape[0]-1):
        nodes_of_elem = node_coords[elem2nodes[p_elem2nodes[i]
            :p_elem2nodes[i+1]]]
        Mone = np.ones((1, nodes_of_elem.shape[0]))
        new_barycenter = 1/nodes_of_elem.shape[0]*Mone@nodes_of_elem
        # print("new_barycenter:", new_barycenter,
        #       "shape:", new_barycenter.shape)
        if i == 0:
            elem_coords = new_barycenter
        else:
            elem_coords = np.concatenate([elem_coords, new_barycenter], axis=0)

    return elem_coords


def get_unit_vectors(node_coords_elem):
    N = node_coords_elem.shape[0]
    results = np.zeros(node_coords_elem.shape)
    for i in range(N):
        vector_i = node_coords_elem[(i+1) % N, :]-node_coords_elem[i, :]
        results[i, :] = vector_i*(1/np.sqrt(np.dot(vector_i, vector_i.T)))
    return results


def compute_Q_one_quandrangle_elem(edge_unit_vectors):
    scalar_prod = np.dot(edge_unit_vectors, edge_unit_vectors.T)
    scalar_keep = np.concatenate(
        [scalar_prod.diagonal(1), scalar_prod.diagonal(3)], axis=0)
    Q = 1-1/4*np.sum(np.abs(scalar_keep))
    return Q


def get_area(u, v):
    cross_prod = np.cross(u, v)
    result = 1/2*np.sqrt(np.dot(cross_prod, cross_prod.T))
    return result


def compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes):
    assert len(p_elem2nodes) >= 2, "not even 1 element in the mesh"
    isQuadrangle = p_elem2nodes[1]-p_elem2nodes[0] == 4

    Q_values = []
    if isQuadrangle:
        for i in range(p_elem2nodes.shape[0]-1):
            nodeids_elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
            node_coords_elem = node_coords[nodeids_elem, :]
            edge_unit_vectors = get_unit_vectors(node_coords_elem)
            Q_elem = compute_Q_one_quandrangle_elem(edge_unit_vectors)
            Q_values.append(Q_elem)
        return np.min(Q_values)
    else:
        # use the formula rho = area/(perimeter/2 )
        for i in range(p_elem2nodes.shape[0]-1):
            nodeids_elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
            node_coords_elem = node_coords[nodeids_elem, :]

            l_dist = []
            for i in range(3):
                v_dist = node_coords_elem[(i+1) % 3, :]-node_coords_elem[i, :]
                dist = np.sqrt(v_dist@v_dist.T)
                l_dist.append(dist)
            l_dist_sorted = np.sort(l_dist)
            h_max = l_dist_sorted[-1]
            perimeter = np.sum(l_dist)
            vector1, vector2 = node_coords_elem[0, :]-node_coords_elem[1,
                                                                       :], node_coords_elem[0, :]-node_coords_elem[-1, :]
            area = get_area(vector1, vector2)
            Q_values.append(h_max*perimeter/(6*np.sqrt(3)*area))
        return np.min(Q_values)


def compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes):
    assert len(p_elem2nodes) >= 2, "not even 1 element in the mesh"

    edge_length_factors = []

    for i in range(p_elem2nodes.shape[0]-1):
        nodeids_elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        node_coords_elem = node_coords[nodeids_elem, :]

        l_dist = []
        for i in range(3):
            v_dist = node_coords_elem[(i+1) % 3, :]-node_coords_elem[i, :]
            dist = np.sqrt(v_dist@v_dist.T)
            l_dist.append(dist)
        l_dist_sorted = np.sort(l_dist)
        edge_length_factors.append(l_dist_sorted[0]/l_dist_sorted[-1])

    return np.min(edge_length_factors)


def compute_pointedness_of_one_quadrangle(node_coords_elem, quadrangle_coords):
    vectors_from_barycenter = []
    areas = []
    # optimiser en décalant de 1
    for j in range(4):
        vectors_from_barycenter.append(
            node_coords_elem[j, :]-quadrangle_coords)
    for j in range(4):
        areas.append(
            get_area(vectors_from_barycenter[j], vectors_from_barycenter[(j+1) % 4]))
    return 4*np.min(areas)/np.sum(areas)


def compute_pointedness_of_element(node_coords, p_elem2nodes, elem2nodes):
    assert len(p_elem2nodes) >= 2, "not even one element in the mesh"
    assert p_elem2nodes[1] - \
        p_elem2nodes[0] == 4, "the mesh is not made of quadrangles"  # homogeneous mesh hypothesis
    elem_coords = compute_barycenter_of_element(
        node_coords, p_elem2nodes, elem2nodes)

    pointedness_values = []
    for i in range(p_elem2nodes.shape[0]-1):
        nodeids_elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        node_coords_elem = node_coords[nodeids_elem, :]
        pointedness = compute_pointedness_of_one_quadrangle(
            node_coords_elem, elem_coords[i])
        pointedness_values.append(pointedness)
    return np.min(pointedness_values)

    


def convert_quadrangles_to_triangles(node_coords, p_elem2nodes, elem2nodes):
    '''
    in place
    '''
    assert len(p_elem2nodes) >= 2, "not even one element in the mesh"

    l_elem_to_remove = []
    l_elem_to_add = []
    for i in range(p_elem2nodes.shape[0]-1):
        if p_elem2nodes[i+1]-p_elem2nodes[i] == 4:
            # it is a quadrangle
            extracted_quadrangle = elem2nodes[p_elem2nodes[i]                                              :p_elem2nodes[i+1]]
            l_elem_to_remove.append(extracted_quadrangle)

            triangle_1 = extracted_quadrangle[0:3]
            triangle_2 = np.concatenate(
                [extracted_quadrangle[2:], extracted_quadrangle[0:1]], axis=0)
            l_elem_to_add.append(triangle_1)
            l_elem_to_add.append(triangle_2)

    for i in range(len(l_elem_to_remove)):
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, l_elem_to_remove[i])
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, l_elem_to_add[2*i])
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, l_elem_to_add[2*i+1])

    # add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, new_elemid_nodes)
    # remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid, factorized=False)
    return node_coords, p_elem2nodes, elem2nodes



if __name__ == '__main__':
    # get_histogram_quality_random_quadrangles(50)
    # get_histogram_quality_random_triangles(50)
    # test_convert_quadrangles_to_triangles()
    print('End.')
