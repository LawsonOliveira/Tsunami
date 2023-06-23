# librairies
import jax.numpy as jnp
import jraph
import networkx as nx
import matplotlib.pyplot as plt

# personal librairies 
import mesh.element_mesh_fractal as element_mesh_fractal
import utils


######################## Graph creation ########################

def convert_elem_mesh2undirected_graph(node_coords, p_elem2nodes, elem2nodes) -> jraph.GraphsTuple:
    node_features = jnp.array(node_coords)

    for idstart in range(p_elem2nodes.shape[0]-1):
        nodes_elem = elem2nodes[p_elem2nodes[idstart]:p_elem2nodes[idstart+1]]

        for j_node in range(nodes_elem.shape[0]):
            new_edge = jnp.array(
                [[nodes_elem[j_node], nodes_elem[(j_node+1) % nodes_elem.shape[0]]]])
            if idstart == 0 and j_node == 0:
                edges = new_edge
            else:
                edges = jnp.concatenate([edges, new_edge], axis=0)
    
    # remove duplicates
    edges = jnp.unique(edges,axis=0)


    # undirected graph
    edges = jnp.concatenate([edges, edges[:, ::-1]], axis=0)
    senders = edges[:, 0]
    receivers = edges[:, 1]

    # optinal : add edge features
    edge_features = jnp.zeros((edges.shape[0],))
    for i in range(edges.shape[0]):
        dist = jnp.linalg.norm(
            node_coords[edges[i, 0], :]-node_coords[edges[i, 1], :])
        edge_features = edge_features.at[i].set(dist)

    # We then save the number of nodes and the number of edges.
    # This information is used to make running GNNs over multiple graphs
    # in a GraphsTuple possible.
    n_node = jnp.array([node_coords.shape[0]])
    n_edge = jnp.array([edges.shape[0]])

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=None
    )
    return graph


def build_toy_graph(n_subdivision=2):
    print(f'build_toy_graph with n_subdivision = {n_subdivision}')
    node_coords, _, p_elem2nodes, elem2nodes, boundary2nodes = element_mesh_fractal.set_simple_mesh()
    # remove z coord
    node_coords = node_coords[:, :2]
    node_coords, p_elem2nodes, elem2nodes, boundary2nodes = element_mesh_fractal.subdivide_all(
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=n_subdivision)
    graph = convert_elem_mesh2undirected_graph(
        node_coords, p_elem2nodes, elem2nodes)
    return graph

######################## Get finite element mesh from vtk file ########################

def map_from_vtk_polydata_file(file_name):
    """For UNSTRUCTURED GRID VTK and POLYDATA"""
    import pyvista
    import numpy

    # file_name is a string, for example, "mesh.vtk"
    pvmesh = pyvista.read(file_name).cast_to_unstructured_grid()
    nelems = pvmesh.number_of_cells

    # faces and p_elem2nodes_shifted
    faces = pvmesh.cells
    p_elem2nodes_shifted = pvmesh.offset + range(nelems + 1)

    # Now we will find p_elem2nodes, elem2nodes, node_coords and elem2subdoms
    elem2nodes = numpy.delete(faces, p_elem2nodes_shifted[:-1])
    p_elem2nodes = p_elem2nodes_shifted - range(nelems + 1)
    node_coords = pvmesh.points
    if pvmesh.GetCellData().GetScalars() != None:
        elem2subdoms = numpy.array(pvmesh.GetCellData().GetScalars(), dtype=numpy.int64)
        #elem2subdoms = pvmesh.active_scalars  # Second option, but the line above is more flexible
    else:
        elem2subdoms = numpy.zeros(nelems, dtype=numpy.int64)

    elem_type = pvmesh.cast_to_unstructured_grid().celltypes

    return node_coords, p_elem2nodes, elem2nodes, elem_type, elem2subdoms

######################## Graph plot ########################

def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(
                int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    print(jraph_graph.nodes.shape)
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(nx_graph, "edge_feature")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels)

    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')

    plt.show()





def test_build_and_draw_graph():
    graph=build_toy_graph(10)
    draw_jraph_graph_structure(graph)

if __name__=="__main__":
    test_build_and_draw_graph()

