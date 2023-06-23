import numpy as np
from copy import deepcopy
from jax import random as jran
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mesh.graph_handling import build_toy_graph
import jraph
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle

def get_id_n_nearest_nodes(node_coords, new_node_coords, n=2):
    assert node_coords.shape[0] >= n
    assert new_node_coords.shape[-1] == new_node_coords.shape[-1], "not the same dimension of nodes"
    Mdist = np.ones((node_coords.shape[0], 1))@new_node_coords-node_coords
    Mdist = Mdist**2@np.ones((node_coords.shape[-1], 1))
    # print("Mdist:", Mdist[:, 0], "shape:", Mdist.shape)
    # print("nearest nodes id:", np.argpartition(Mdist[:, 0], n)[:n])
    return np.argpartition(Mdist[:, 0], n)[:n]

def get_norm_L2(A):
    new_A = deepcopy(A)
    if len(A.shape)==1:
        new_A = A[:,None]
    return np.trace(new_A@np.conjugate(new_A.T))


def check_new_row_already_exists_in_matrix(A,new_row):
    '''
    Args:
     - ...
     - node_to_check of shape (dim,)
    Returns:
      - bool
    '''
    if len(new_row.shape)==1:
        # local operation
        new_row = np.expand_dims(new_row,0)
    flag_matrix = np.zeros(A.shape)
    for i in range(new_row.shape[-1]):
        flag_matrix[:, i] = np.where(
            A[:, i] == new_row[0,i], 1, 0)

    N_identic_rows = np.sum(np.prod(flag_matrix, axis=1))
    assert N_identic_rows <= 1, 'some nodes are merged'
    return N_identic_rows >= 1


################# post process ######################

def plot_comparison_to_ground_truth(solver,test_graph):
    '''
    solver must have a true_solution method informed
    '''
    assert not(solver.true_solution is None) 
    ### Approximated solution ###
    # We plot the solution obtained with our NN
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    plt.subplot(131)

    predictions = solver.solution(solver.params,test_graph)
    XY_test = test_graph.nodes

    plt.scatter(XY_test[:,0],XY_test[:,1], c=predictions, cmap="hot",s=100)
    plt.clim(vmin=jnp.min(predictions),vmax=jnp.max(predictions))
    plt.colorbar()
    plt.title("NN solution")

    ######### True solution #########
    # We plot the true solution, its form was mentioned above
    plt.subplot(132)

    true_sol = solver.true_solution(test_graph)

    plt.scatter(XY_test[:,0],XY_test[:,1], c=true_sol, cmap="hot",s=100)
    plt.clim(vmin=jnp.min(true_sol),vmax=jnp.max(true_sol))
    plt.colorbar()
    plt.title("True solution")

    ### Absolute error ###
    # We plot the absolut error, it's |true solution - neural network output|
    plt.subplot(133)

    error=abs(true_sol-predictions)

    plt.scatter(XY_test[:,0],XY_test[:,1], c=error, cmap="viridis",s=100)
    plt.clim(vmin=0,vmax=jnp.max(error))
    plt.colorbar()
    plt.title("Absolute error")

    plt.show()

def save_graph(graph:jraph.GraphsTuple,file_dst:str):
    with open(file_dst, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(file_src:str)->jraph.GraphsTuple:
    with open(file_src,'rb') as f:
        graph = pickle.load(f)
    return graph