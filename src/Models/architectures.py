import jax
import functools



class Real_MLP:
    """
        Create a multilayer perceptron and initialize the neural network
    Inputs :
        A SEED number and the layers structure
    """
    def __init__(self, seed, layers):
        self.key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(self.key,len(layers))
        self.layers = layers
        self.params = []


    def initialize_params(self):
        """
        Initialize weigths and bias

        Parameters
        ----------
        Returns
        -------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        """
        for layer in range(0, len(self.layers)-1):
            in_size,out_size = self.layers[layer], self.layers[layer+1]
            weights = jax.nn.initializers.glorot_normal()(self.keys[layer], (out_size, in_size), jax.numpy.float32)
            bias = jax.nn.initializers.lecun_normal()(self.keys[layer], (out_size, 1), jax.numpy.float32).reshape((out_size, ))
            self.params.append((weights, bias))
        return self.params
        

    @functools.partial(jax.jax.jit, static_argnums=(0,))    
    def evaluation(self, params, inputs):
        """
        Evaluate an input

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, input_size]]
            -- points
        Returns
        -------
        output : jax.numpy.array[batch_size, output_size]
            -- neural network applied to inputs 
        """
        for layer in range(0, len(params)-1):
            weights, bias = params[layer]
            inputs = jax.nn.tanh(jax.numpy.add(jax.numpy.dot(inputs, weights.T), bias))
        weights, bias = params[-1]
        output = jax.numpy.dot(inputs, weights.T)+bias
        return output






class Complex_MLP:
    """
        Create a complex multilayer perceptron and initialize the neural network
    Inputs :
        A SEED number and the layers structure
    """
    def __init__(self, seed, layers):
        self.key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(self.key, len(layers) - 1)
        self.layers = layers
        self.params = []


    def initialize_params(self):
        """
        Initialize weigths and bias

        Parameters
        ----------
        
        Returns
        -------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        """
        for layer in range(0, len(self.layers)-2):
            in_size,out_size = self.layers[layer], self.layers[layer+1]
            weights = jax.nn.initializers.glorot_normal()(self.keys[layer], (out_size, in_size), jax.numpy.float32)
            bias = jax.nn.initializers.lecun_normal()(self.keys[layer], (out_size, 1), jax.numpy.float32).reshape((out_size, ))
            self.params.append((weights, bias))
        return self.params
        

    @functools.partial(jax.jit, static_argnums=(0,))    
    def evaluation(self, params, inputs):
        """
        Evaluate an input

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, input_size]]
            -- points

        Returns
        -------
        output : jax.numpy.array[batch_size, output_size] 
            -- neural network applied to inputs (Complex array) 
        """
        for layer in range(0, len(params)-1):
            weights, bias = params[layer]
            inputs = jax.nn.tanh(jax.numpy.add(jax.numpy.dot(inputs, weights.T), bias))
        weights, bias = params[-1]
        real_and_imaginary_layers = jax.numpy.dot(inputs, weights.T)+bias  # The first output of the NN is the real part, the second is the imaginary part        
        real_and_imaginary_layers = real_and_imaginary_layers.reshape((-1, self.layers[-1]))
        real_layer = real_and_imaginary_layers[:real_and_imaginary_layers.shape[0]//2]
        imag_layer = real_and_imaginary_layers[real_and_imaginary_layers.shape[0]//2:]
        output = jax.lax.complex(real_layer, imag_layer)
        return output


if '__main__' == __name__:
    import numpy

    seed = 351
    n_features = 2        # Input dimension (x1, x2)
    n_targets = 2        # Output dimension. It's a complex number (y1 + j*y2)
    hidden_layers = [50, 50, 50, 50, 50]   # Hidden layers structure
    layers = [n_features] + hidden_layers + [2*n_targets] + [n_targets]
    n_points = 100
    mlp = Complex_MLP(seed, layers)
    params = mlp.initialize_params()
    xy = numpy.random.uniform(low=0, high=1, size=(n_points, n_features))

    print(mlp.evaluation(params, xy).shape)
