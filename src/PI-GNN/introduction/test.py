# librairies
import numpy as np
import jax.numpy as jnp
import jraph
import networkx as nx

# personal librairies 
import element_mesh_fractal
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
            elif not (utils.check_new_row_already_exists_in_matrix(edges, new_edge)):
                edges = jnp.concatenate([edges, new_edge], axis=0)

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

######################## Graph creation ########################

def new_convert_elem_mesh2undirected_graph(node_coords, p_elem2nodes, elem2nodes) -> jraph.GraphsTuple:
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

def new_build_toy_graph(n_subdivision=2):
    print(f'build_toy_graph with n_subdivision = {n_subdivision}')
    node_coords, _, p_elem2nodes, elem2nodes, boundary2nodes = element_mesh_fractal.set_simple_mesh()
    # remove z coord
    node_coords = node_coords[:, :2]
    node_coords, p_elem2nodes, elem2nodes, boundary2nodes = element_mesh_fractal.subdivide_all(
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=n_subdivision)
    graph = new_convert_elem_mesh2undirected_graph(
        node_coords, p_elem2nodes, elem2nodes)
    return graph

######################## Time study ########################

import time
import matplotlib as plt

n_subdivision_max = 2
time_records = np.zeros((2,n_subdivision_max+1))
for n_subdivision in range(1,n_subdivision_max+1):
    print(f"n_subdivision {n_subdivision}")
    
    start_time = time.time()
    build_toy_graph(n_subdivision=n_subdivision)
    end_time = time.time()
    time_records[0,n_subdivision] = end_time-start_time

    start_time = time.time()
    new_build_toy_graph(n_subdivision=n_subdivision)
    end_time = time.time()
    time_records[1,n_subdivision] = end_time-start_time

plt.plot(np.arange(n_subdivision_max),time_records[0,:],label="old version")
plt.plot(np.arange(n_subdivision_max),time_records[1,:],label="new version")

