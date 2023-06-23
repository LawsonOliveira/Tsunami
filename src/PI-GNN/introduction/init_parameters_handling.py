from jax import random as jran
import jax.numpy as jnp
import numpy as np
import jax
import haiku as hk

############# parameters initializations #############

def init_GlorotNormal(key_init, in_size,out_size):
    std_dev = jnp.sqrt(2/(in_size + out_size ))
    weights=jran.truncated_normal(key_init, -2, 2, shape=(out_size, in_size), dtype=np.float32)*std_dev
    bias=jran.truncated_normal(key_init, -1, 1, shape=(out_size, 1), dtype=np.float32).reshape((out_size,))
    return weights,bias

#
def init_GlorotUniform(key_init,in_size,out_size):
    std_dev = jnp.sqrt(2/(in_size + out_size ))
    weights=jran.truncated_normal(key_init, -2, 2, shape=(out_size, in_size), dtype=np.float32)*std_dev
    bias=jran.truncated_normal(key_init, -1, 1, shape=(out_size, 1), dtype=np.float32).reshape((out_size,))
    return weights,bias

#
def init_HeNormal(key_init,in_size,out_size):
    std_dev = jnp.sqrt(2/(in_size + out_size ))
    weights=jran.truncated_normal(key_init, -2, 2, shape=(out_size, in_size), dtype=np.float32)*std_dev
    bias=jran.truncated_normal(key_init, -1, 1, shape=(out_size, 1), dtype=np.float32).reshape((out_size,))
    return weights,bias

#
def init_HeUniform(key_init,in_size,out_size):
    std_dev = jnp.sqrt(2/(in_size + out_size ))
    weights=jran.truncated_normal(key_init, -2, 2, shape=(out_size, in_size), dtype=np.float32)*std_dev
    bias=jran.truncated_normal(key_init, -1, 1, shape=(out_size, 1), dtype=np.float32).reshape((out_size,))
    return weights,bias


############# parameters initializations #############
def get_initialized_layer(key_init,in_size,out_size,features,activation_fun):
    '''
    TODO: The core of the function is made. But more options are still to be implemented.
    '''
    if activation_fun=="tanh":
        w_init, b_init = init_GlorotNormal(key_init,in_size,out_size)
        return jax.nn.tanh(hk.Linear(features,w_init=w_init,b_init=b_init))
    else:
        return jax.nn.tanh(hk.Linear(features))