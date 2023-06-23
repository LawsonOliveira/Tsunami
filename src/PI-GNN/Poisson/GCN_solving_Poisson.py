# import modules
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

# our scripts
from GCN_solver.GCN_model import PIGNN,GCN
from mesh.graph_handling import build_toy_graph
import utils


class PIGNN_Poisson(PIGNN):
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

    # Compute the loss function
    @partial(jit, static_argnums=(0,))
    def loss_function(self, params, input_graph):
        targets = self.target_function(input_graph.nodes)
        preds = -self.laplacian(params, input_graph)
        return jnp.linalg.norm(preds-targets)/input_graph.n_node[0]

def get_errors_PIGNN_L1_L2_poisson_dddd(h,n_train_steps=10000):
    # Neural network parameters
    SEED = 351
    n_origin_features, n_targets = 2, 1            # Input and output dimension
    features = [n_origin_features,10, n_targets]      # Layers structure

    # Initialization
    # --Use GCN model to define the update node function
    def update_node_fn(x): return GCN(SEED,features=features)(x)
    

    nelem = int(1/h)
    graph = build_toy_graph(nelem)

    print("graph.n_node:",graph.n_node)
    print()

    # --Useful to see progress during training when we have it
    def true_solution(input_graph):
        inputX, inputY = input_graph.nodes[:,0:1], input_graph.nodes[:,1:2]
        return jnp.multiply(jnp.sin(jnp.pi*inputX),jnp.sin(jnp.pi*inputY))

    solver = PIGNN_Poisson(update_node_fn, graph, true_solution=true_solution)
    solver.train(graph,int(n_train_steps/50), n_train_steps)

    solappro = solver.solution(solver.params,graph)
    solexact = solver.true_solution(graph)

    solerr_normalized_L1 = utils.get_normalized_norm_L1(solappro - solexact)
    solerr_normalized_L2 = utils.get_normalized_norm_L2(solappro - solexact)

    return solerr_normalized_L1, solerr_normalized_L2

if __name__ == "__main__":
    print('--begin--')
    h_values = np.array([.1/deno for deno in [2, ]])
    utils.get_alpha_error(h_values,get_errors_PIGNN_L1_L2_poisson_dddd,"PIGCN","Poisson")