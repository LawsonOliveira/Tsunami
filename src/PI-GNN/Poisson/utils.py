import numpy as np
import time
import scipy
from copy import deepcopy
from jax import random as jran
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mesh.graph_handling import build_toy_graph
import jraph
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle
import os

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

def get_normalized_norm_L1(A):
    normalization_coef = int(sum([s for s in A.shape]))
    norm_L1 = np.sum(np.abs(A))
    return norm_L1 / normalization_coef


def get_normalized_norm_L2(A):
    normalization_coef = int(sum([s for s in A.shape]))
    adj_A = np.conjugate(A.T)
    # abs to ensure Python return a real number
    norm_L2 = np.abs(np.trace(np.dot(adj_A, A)))
    return norm_L2 / normalization_coef



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



################# error study ######################

def get_alpha_error(h_values,fn_get_errors_L1_L2,solver_name,equation_name):
    solerr_normalized_L1_values = np.zeros(h_values.shape[0])
    solerr_normalized_L2_values = np.zeros(h_values.shape[0])
    for i, h in enumerate(h_values):
        print(f"study error for h={h}")
        time_start = time.time()
        solerr_normalized_L1, solerr_normalized_L2 = fn_get_errors_L1_L2(h)
        solerr_normalized_L1_values[i] = solerr_normalized_L1
        solerr_normalized_L2_values[i] = solerr_normalized_L2
        time_end = time.time()
        print(f"    took: {time_end-time_start} s")

    # -- analyse L1 error
    lin_reg_res = scipy.stats.linregress(
        np.log(h_values), np.log(solerr_normalized_L1_values))
    print('Linear regression results MAE:')
    print(r'with $MAE = C h^{\alpha}$')
    print('C = ', np.exp(lin_reg_res.intercept))
    print(r'$\alpha$ =', lin_reg_res.slope, '\n')
    plt.plot(np.log(h_values),
                           np.log(solerr_normalized_L1_values), label='log(MAE)')
    plt.plot(np.log(
        h_values), lin_reg_res.slope*np.log(h_values)+lin_reg_res.intercept, label='regression')
    plt.title(
        f"Study of MAE in function of h for {equation_name} equation:\n C={np.exp(lin_reg_res.intercept)}, alpha={lin_reg_res.slope}")
    plt.legend()
    dst_file_path = f"./src/PI-GNN/{equation_name}/results/{solver_name}_study_MAE_h_{equation_name}"
    plt.savefig(dst_file_path)
    plt.show()
    plt.close()

    # -- analyse L2 error
    lin_reg_res = scipy.stats.linregress(
        np.log(h_values), np.log(solerr_normalized_L2_values))
    print('Linear regression results MSE:')
    print(r'with $MSE = C h^{\alpha}$')
    print('C = ', np.exp(lin_reg_res.intercept))
    print(r'$\alpha$ =', lin_reg_res.slope)
    plt.plot(np.log(h_values),
                           np.log(solerr_normalized_L2_values), label="log(MSE)")
    plt.plot(np.log(
        h_values), lin_reg_res.slope*np.log(h_values)+lin_reg_res.intercept, label='regression')
    plt.title(
        f"Study of MSE in function of h for {equation_name} equation:\n C={np.exp(lin_reg_res.intercept)}, alpha={lin_reg_res.slope}")
    plt.legend()
    dst_file_path = f"./src/PI-GNN/{equation_name}/results/{solver_name}_study_MSE_h_{equation_name}"
    plt.savefig(dst_file_path)
    plt.show()
    plt.close()