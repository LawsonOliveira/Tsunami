#!/usr/bin/env python
# coding: utf-8

# Introduction and first steps to a GNN methodology

# source: https://colab.research.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb#scrollTo=KkkOolOVh3nR

# Imports
import functools
import matplotlib.pyplot as plt
import jax
from jax import value_and_grad, vmap, jit, jacfwd
from functools import partial
import jax.numpy as jnp
from jax import random as jran
import jax.tree_util as tree
import jraph
import flax
import haiku as hk
import optax
import pickle
import numpy as onp
import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Tuple

# our scripts
import utils
import GCN_solver.init_parameters_handling as iph


######################## Code GNN ########################


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    receivers = jnp.concatenate(
        (receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


# One layer

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L506
def GraphConvolution(update_node_fn: Callable,
                     aggregate_nodes_fn: Callable = jax.ops.segment_sum,
                     add_self_edges: bool = False,
                     symmetric_normalization: bool = True) -> Callable:
    """Returns a method that applies a Graph Convolution layer.

    Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,
    NOTE: This implementation does not add an activation after aggregation.
    If you are stacking layers, you may want to add an activation between
    each layer.
    Args:
      update_node_fn: function used to update the nodes. In the paper a single
        layer MLP is used.
      aggregate_nodes_fn: function used to aggregates the sender nodes.
      add_self_edges: whether to add self edges to nodes in the graph as in the
        paper definition of GCN. Defaults to False.
      symmetric_normalization: whether to use symmetric normalization. Defaults to
        True.

    Returns:
      A method that applies a Graph Convolution layer.
    """

    def _ApplyGCN(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Convolution layer."""
        nodes, _, receivers, senders, _, _, _ = graph

        # First pass nodes through the node updater.
        nodes = update_node_fn(nodes)

        # Equivalent to jnp.sum(n_node), but jittable
        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            # In principle, a `GraphsTuple` should partition by n_edge, but in
            # this case it is not required since a GCN is agnostic to whether
            # the `GraphsTuple` is a batch of graphs or a single large graph.
            conv_receivers, conv_senders = add_self_edges_fn(receivers, senders,
                                                             total_num_nodes)
        else:
            conv_senders = senders
            conv_receivers = receivers

        # pylint: disable=g-long-lambda
        if symmetric_normalization:
            # Calculate the normalization values.
            def count_edges(x): return jax.ops.segment_sum(
                jnp.ones_like(conv_senders), x, total_num_nodes)
            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)

            # Pre normalize by sqrt sender degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: x *
                jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                nodes,
            )
            # Aggregate the pre-normalized nodes.
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                             total_num_nodes), nodes)
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x:
                (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))
                 [:, None]),
                nodes,
            )
        else:
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                             total_num_nodes), nodes)
        # pylint: enable=g-long-lambda
        return graph._replace(nodes=nodes)

    return _ApplyGCN


# Multiple layers GCN
class GCN(hk.Module):
    """Defines a graph neural network with GCN layers.  """

    def __init__(self, SEED: int, features: List):
        super().__init__()
        self.features = features
        # self.key=jran.PRNGKey(SEED)
        # self.keys = jran.split(self.key,len(features))

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        layers = []
        for layer,feat in enumerate(self.features[:-1]):
            # TODO: check that the initialzation made by hk is the good one
            # see this article published in 2015 : https://arxiv.org/pdf/1502.03167v3.pdf
            # compare to the ones recommended in Tensorflow O'Reilly of Aurélien Géron
            # key_init = self.keys[layer]
            # in_size,out_size=self.features[layer], self.features[layer+1]
            
            # TODO: check the importance of the activation function on the first layer 
            layers.append(GraphConvolution(
                update_node_fn=lambda n: jax.nn.tanh(hk.Linear(feat)(n)),
                add_self_edges=True))
        #TODO QUICK: just put lambda n: hk.Linear(8)(n) in update_node_fn and 1 layer 
        layers.append(GraphConvolution(
            update_node_fn=hk.Linear(self.features[-1])))

        gcn = hk.Sequential(layers)
        return gcn(graph)


######################## Solve Poisson equation ########################

class PDE_operators2d_graph:
    """
        Class with the most common operators used to solve PDEs
    Input:
        A function that we want to compute the respective operator
    """

    # Class initialization
    def __init__(self, function):
        '''
        Args:
        - function : a function taking params and jraph.GraphTuple as args'''
        self.function = function

    # Compute the two dimensional laplacian
    def laplacian_2d_graph(self, params, input_graph):
        def fun(params, x, y): 
            return self.function(params, input_graph._replace(nodes=jnp.concatenate([x, y], axis=-1)))

        @partial(jit)
        def action(params, x, y):
            u_xx = jacfwd(jacfwd(fun, 1), 1)(params, x, y)
            u_yy = jacfwd(jacfwd(fun, 2), 2)(params, x, y)
            return u_xx + u_yy
        # no automatic vectorization with vmap
        # TODO: optimize it  => do it manually ?
        vec_fun = action
        laplacian = vec_fun(
            params, input_graph.nodes[:, 0:1], input_graph.nodes[:, 1:2])
        return laplacian

    # Compute the partial derivative in x  
    def du_dx(self,params,input_graph):
        def fun(params, x, y): 
            return self.function(params, input_graph._replace(nodes=jnp.concatenate([x, y], axis=-1)))
        
        @partial(jit)    
        def action(params,x,y):
            u_x = jacfwd(fun, 1)(params,x,y)
            return u_x

        # TODO: optimize it  => do it manually ?
        vec_fun = action
        return vec_fun(
            params, input_graph.nodes[:, 0:1], input_graph.nodes[:, 1:2])

    # Compute the partial derivative in y
    def du_dx(self,params,input_graph):
        def fun(params, x, y): 
            return self.function(params, input_graph._replace(nodes=jnp.concatenate([x, y], axis=-1)))
        
        @partial(jit)    
        def action(params,x,y):
            u_x = jacfwd(fun, 2)(params,x,y)
            return u_x

        # TODO: optimize it  => do it manually ?
        vec_fun = action
        return vec_fun(
            params, input_graph.nodes[:, 0:1], input_graph.nodes[:, 1:2])

####### Physics Informed Graph Neural Networks #######
class PIGNN:
    """
    Solve a PDE using Physics Informed Graph Neural Networks
    Input:
        The evaluation function of the neural network
    """

    # Class initialization params
    def __init__(self, update_node_fn, not_memorized_graph, true_solution = None):
        self.operators = PDE_operators2d_graph(self.solution)
        self.laplacian = self.operators.laplacian_2d_graph
        # initialiaze the parameters of the model 
        self.update_node_module = hk.without_apply_rng(
            hk.transform(update_node_fn))
        self.params = self.update_node_module.init(
            jax.random.PRNGKey(42), not_memorized_graph)
        # if we have it
        self.true_solution = true_solution

    # Definition of the function A(x,y) mentioned above
    @partial(jit, static_argnums=(0,))
    def A_function(self, inputX, inputY):
        return jnp.zeros_like(inputX).reshape((-1,1))

    # Definition of the function F(x,y) mentioned above
    @partial(jit, static_argnums=(0,))
    def F_function(self, inputX, inputY):
        F1 = jnp.multiply(jnp.sin(inputX), jnp.sin(
            inputX-jnp.ones_like(inputX)))
        F2 = jnp.multiply(jnp.sin(inputY), jnp.sin(
            inputY-jnp.ones_like(inputY)))
        return jnp.multiply(F1, F2).reshape((-1, 1))

    # Definition of the function f(x,y) mentioned above
    @partial(jit, static_argnums=(0,))
    def target_function(self, inputs):
        return 2*jnp.pi**2*jnp.sin(jnp.pi*inputs[:, 0:1])*jnp.sin(jnp.pi*inputs[:, 1:2])

    # Compute the solution of the PDE on the points (x,y)
    @partial(jit, static_argnums=(0,))
    def solution(self, params, input_graph):
        out_graph = self.update_node_module.apply(params, input_graph)
        NN = out_graph.nodes

        inputX, inputY = input_graph.nodes[:,0], input_graph.nodes[:,1]
        F = self.F_function(inputX, inputY)
        A = self.A_function(inputX, inputY)
        return jnp.add(jnp.multiply(F, NN), A).reshape(-1,1)

    # Compute the loss function
    @partial(jit, static_argnums=(0,))
    def loss_function(self, params, input_graph):
        targets = self.target_function(input_graph.nodes)
        preds = -self.laplacian(params, input_graph)
        return jnp.linalg.norm(preds-targets)/input_graph.n_node[0]
    


    # Train step
    # Notice: useless to jit, tried with an inner_train_step and it was worse
    def train_step(self, opt_update, opt_state, input_graph):
        loss, grad = value_and_grad(self.loss_function)(
            self.params, input_graph)
        # print("grad:\n",grad)
        updates, opt_state = opt_update(grad, opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return loss, self.params, opt_state
    
    
    def MAE_error_function(self,input_graph):
        if not(self.true_solution is None):
            preds = self.solution(self.params,input_graph)
            exact_sol = self.true_solution(input_graph) 
            MAE_error = jnp.linalg.norm(preds-exact_sol,ord=1)/input_graph.n_node[0]
        else:
            MAE_error = 'no solution of reference'
        return MAE_error


    def train(self, input_graph: jraph.GraphsTuple, report_steps : int, num_train_steps: int) -> Tuple[Dict, hk.Params]:
        """Training loop."""
        # Initialize the optimizer.
        # TODO: learning rate to adapt
        opt_init, opt_update = optax.adam(1e-2)
        opt_state = opt_init(self.params)

        history = {"loss": []}
        for idx in range(num_train_steps):
            loss, self.params, opt_state = self.train_step(
                opt_update, opt_state, input_graph)
            history["loss"].append(loss)
            if idx % report_steps == 0:
                MAE_error = self.MAE_error_function(input_graph)
                print(f'step: {idx}, loss MSE: {loss}, MAE error: {MAE_error}')
        print('Training is over\n')
        return history, self.params


