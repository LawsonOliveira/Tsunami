# # Solving PDEs with Jax - Problem 8
# ## Description
# 
# ### Average time of execution 
# Around 15-30 seconds
# 
# ### PDE
# We will try to solve the problem 8 of the article https://ieeexplore.ieee.org/document/712178  
# 
# $\Delta \psi(x,y) +\psi(x,y)\cdot\frac{\functools.partial \psi(x,y)}{\functools.partial y}= f(x,y)$ on $\Omega = [0,1]^2$  
# where $f(x, y)=\sin(\pi x)(2-\pi^2y^2+2y^3\sin(\pi x))$   
# 
# ### Boundary conditions    
# $\psi(0,y)=\psi(1,y)=\psi(x,0)=0$ and $\frac{\functools.partial \psi}{\functools.partial y}(x,1)=2\sin(\pi x)$           
# 
# ### Loss function
# The loss to minimize here is $\mathcal{L} = ||\Delta \psi(x,y) +\psi(x,y)\cdot\frac{\functools.partial \psi(x,y)}{\functools.partial y}-f(x,y) ||_2$  
# 
# ### Analytical solution
# The true function $\psi$ should be $\psi(x, y)=y^2sin(\pi x)$  
# This solution is the same of the problem 7
# 
# ### Approximated solution
# We want find a solution $\psi(x,y)=A(x,y)+F(x,y)N(x,y)$
# s.t:  
# $F(x,y)=\sin(x-1)\sin(y-1)\sin(x)\sin(y)$ 
# $A(x,y)=y\sin(\pi x)$   
# 

###########################################################################################################################
#################################################### Libraries ############################################################
###########################################################################################################################
import jax
import optax
import functools
import matplotlib.pyplot
import numpy
import scipy
import pickle

# Set and verify device
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_disable_jit', True) # Desactive the compilation for better debugging
print(jax.lib.xla_bridge.get_backend().platform)

'''Multilayer Perceptron'''
class MLP:
    """
        Create a multilayer perceptron and initialize the neural network
    Inputs :
        A SEED number and the layers structure
    """
    
    # Class initialization
    def __init__(self, key, layers):
        self.key = key
        self.keys = jax.random.split(self.key,len(layers))
        self.layers = layers
        self.params = []

    def MLP_create(self):
        """
        Initialize the MLP weigths and bias
        Parameters
        ----------
        Returns
        -------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        """
        for layer in range(0, len(self.layers)-1):
            in_size,out_size = self.layers[layer], self.layers[layer+1]
            std_dev = jax.numpy.sqrt(2/(in_size + out_size ))
            weights = jax.random.truncated_normal(self.keys[layer], -2, 2, shape=(out_size, in_size), dtype=numpy.float32)*std_dev
            bias = jax.random.truncated_normal(self.keys[layer], -1, 1, shape=(out_size, 1), dtype=numpy.float32).reshape((out_size,))
            self.params.append((weights,bias))
        return self.params
        

    @functools.partial(jax.jit, static_argnums=(0,))    
    def NN_evaluation(self, params, inputs):
        """
        Evaluate a position XY using the neural network    
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        output : jax.numpy.array[batch_size]
            -- output of the neural network 
        """
        for layer in range(0, len(params)-1):
            weights, bias = params[layer]
            inputs = jax.nn.tanh(jax.numpy.add(jax.numpy.dot(inputs, weights.T), bias))
        weights, bias = params[-1]
        output = jax.numpy.dot(inputs, weights.T)+bias
        return output






###########################################################################################################################
############################################ Two dimensional pde operators ################################################
###########################################################################################################################
class PDE_operators2d:
    """
        Class with the most common operators used to solve PDEs
    Input:
        A function that we want to compute the respective operator
    """
    
    # Class initialization
    def __init__(self, function):
        self.function = function

    def laplacian_2d(self,params,inputs):
        """
        Compute the two dimensional laplacian
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        laplacian : jax.numpy.array[batch_size]
            -- numerical values of the laplacian applied to the inputs
        """
        fun = lambda params,x,y: self.function(params, x,y)
        @jax.jit  
        def action(params,x,y):
            u_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params,x,y)    # Compute the second derivative in x
            u_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params,x,y)    # Compute the second derivative in y
            return u_xx + u_yy
        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))  # vmap vectorize the action function
        laplacian = vec_fun(params, inputs[:,0], inputs[:,1])
        return laplacian


    @functools.partial(jax.jit, static_argnums=(0,))    
    def du_dx(self, params, inputs):
        """
        Compute the first derivative in x of self.function
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        du_dx_values : jax.numpy.array[batch_size]
            -- numerical values of the first derivative in x applied to the inputs
        """
        fun = lambda params,x,y: self.function(params, x,y)
        @jax.jit  
        def action(params,x,y):
            u_x = jax.jacfwd(fun, 1)(params,x,y)    # Compute the first derivative in x
            return u_x
        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))  # vmap vectorize the action function
        du_dx_values = vec_fun(params, inputs[:,0], inputs[:,1])
        return du_dx_values


    @functools.partial(jax.jit, static_argnums = (0,))    
    def du_dy(self, params, inputs):
        """
        Compute the first derivative in y of self.function
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        du_dy_values : jax.numpy.array[batch_size]
            -- numerical values of the first derivative in y applied to the inputs
        """
        fun = lambda params,x,y: self.function(params, x,y)
        @jax.jit
        def action(params,x,y):
            u_y = jax.jacfwd(fun, 2)(params,x,y)    # Compute the first derivative in y
            return u_y
        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))   # vmap vectorize the action function
        du_dy_values = vec_fun(params, inputs[:,0], inputs[:,1])
        return du_dy_values






###########################################################################################################################
############################################ Physics Informed Neural Networks #############################################
###########################################################################################################################
class PINN:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network
    """

    # Class initialization
    def __init__(self,NN_evaluation):
        self.operators = PDE_operators2d(self.solution) # Compute all of operators of solution the (F*NN_evaluation+A)
        self.laplacian = self.operators.laplacian_2d # Laplacian of the solution (F*NN_evaluation+A)
        self.NN_evaluation = NN_evaluation # Neural network output
        self.dsol_dy = self.operators.du_dy # First derivative of the solution (F*NN_evaluation+A)

    # Definition of the function A(x,y) mentioned above
    @functools.partial(jax.jit, static_argnums=(0,))    
    def A_function(self, inputX, inputY):
        """
        Compute A function A(x,y) mentioned above
        Parameters
        ----------
        inputX : jax.numpy.array[batch_size]
            -- points on the x-axis of the mesh
        inputY : jax.numpy.array[batch_size]
            -- points on the y-axis of the mesh
        Returns
        -------
        A : jax.numpy.array[batch_size,batch_size]
            -- A function applied to inputs
        """
        A = jax.numpy.multiply(inputY, jax.numpy.sin(jax.numpy.pi*inputX)).reshape(-1,1)
        return A

    # Definition of the function F(x,y) mentioned above   
    @functools.partial(jax.jit, static_argnums=(0,))    
    def F_function(self, inputX, inputY):
        """
        Compute F function F(x,y) mentioned above
        Parameters
        ----------
        inputX : jax.numpy.array[batch_size]
            -- points on the x-axis of the mesh
        inputY : jax.numpy.array[batch_size]
            -- points on the y-axis of the mesh
        Returns
        -------
        F : jax.numpy.array[batch_size,batch_size]
            -- F function applied to inputs
        """
        F1 = jax.numpy.multiply(jax.numpy.sin(inputX), jax.numpy.sin(inputX - jax.numpy.ones_like(inputX)))
        F2 = jax.numpy.multiply(jax.numpy.sin(inputY), jax.numpy.sin(inputY - jax.numpy.ones_like(inputY)))
        F = jax.numpy.multiply(F1,F2).reshape((-1,1))
        return F

    # Definition of the  
    @functools.partial(jax.jit, static_argnums=(0,))    
    def target_function(self, inputs):
        """
        Compute the function f(x,y) mentioned above
        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        f : jax.numpy.array[batch_size]
            -- f function applied to inputs
        """
        f = jax.numpy.multiply(jax.numpy.sin(jax.numpy.pi*inputs[:,0]),2 - jax.numpy.pi**2*inputs[:,1]**2 + 2*inputs[:,1]**3*jax.numpy.sin(jax.numpy.pi*inputs[:,0])).reshape(-1,1) 
        return f

    @functools.partial(jax.jit, static_argnums = (0,))    
    def solution(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on the points (x,y)
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size]
            -- points on the x-axis of the mesh
        inputY : jax.numpy.array[batch_size]
            -- points on the y-axis of the mesh
        Returns
        -------
        applied_solution : jax.numpy.array[batch_size]
            -- PINN solution applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        F = self.F_function(inputX, inputY)
        A = self.A_function(inputX, inputY)
        applied_solution = jax.numpy.add(jax.numpy.multiply(F,NN), A).reshape(-1,1)
        return applied_solution

    @functools.partial(jax.jit, static_argnums = (0,))    
    def loss_function(self, params, inputs):
        """
        Compute the residual (loss function)
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        """
        targets = self.target_function(inputs)
        laplacian = self.laplacian(params, inputs).reshape(-1,1)
        dsol_dy_values = self.dsol_dy(params,inputs)[:,0].reshape((-1,1))
        preds = laplacian + jax.numpy.multiply(self.solution(params, inputs[:,0], inputs[:,1]), dsol_dy_values).reshape(-1,1)
        loss = jax.numpy.linalg.norm(preds - targets)
        return loss
 
    @functools.partial(jax.jit, static_argnums = (0,))    
    def train_step(self, params, opt_state, inputs):
        """
        Make just one step of the training
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- points in the mesh
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        update_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        """
        loss, gradient = jax.value_and_grad(self.loss_function)(params, inputs)
        updates, new_opt_state = optimizer.update(gradient, opt_state)
        update_params = optax.apply_updates(params, updates)
        return loss, update_params, new_opt_state




###########################################################################################################################
############################################## analytical solution ########################################################
###########################################################################################################################
def analytical_solution(inputs):
    """
    Compute the true solution given a inputs array (x,y).
    Parameters
    ----------
    inputs : jax.numpy.ndarray[[batch_size,batch_size]]
        -- points in the mesh
    Returns
    -------
    analytical_sol : jax.numpy.array[batch_size]
        -- analytical solution applied to inputs
    """
    analytical_sol = jax.numpy.multiply(inputs[:,1]**2, jax.numpy.sin(jax.numpy.pi*inputs[:,0]))
    return analytical_sol




    
###########################################################################################################################
################################################### Parameters ############################################################
###########################################################################################################################
# Neural network parameters
SEED = 351
n_features, n_targets = 2, 1            # Input and output dimension
layers = [n_features,30,n_targets]      # Layers structure

# Train parameters
num_batches = 20000
report_steps = 1
learning_rate = 0.000320408173
h_list = jax.numpy.linspace(0.05, 0.1, 50)
load = False




    
    
    
    

###########################################################################################################################
################################################### Initialization ########################################################
###########################################################################################################################
# Initialization
key = jax.random.PRNGKey(SEED)
NN_MLP = MLP(key,layers)                 
params = NN_MLP.MLP_create()            # Create the MLP
NN_eval = NN_MLP.NN_evaluation            # Evaluate function
solver = PINN(NN_eval)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

loss_history = []
mean_absolute_error_list = []



########################################################################################################
################################################## Training ############################################
########################################################################################################
if not load:
    count = 0
    for h_param in h_list:
        X =  jax.numpy.arange(0, 1, h_param)
        Y = jax.numpy.arange(0, 1, h_param)
        X, Y = jax.numpy.meshgrid(X, Y)
        XY_train = jax.numpy.column_stack((X.flatten(),Y.flatten()))
        error_at_step = jax.numpy.inf
        for ibatch in range(0, num_batches):

            loss, params, opt_state = solver.train_step(params,opt_state, XY_train)
            loss_history.append(float(loss))

            if loss<=numpy.min(loss_history): # save if the current state is the best 
                pickle.dump(params,open("./NN_saves/params", "wb"))
    
        # get error
        params = pickle.load(open("./NN_saves/params", "rb"))
        n_points = 100000
        ran_key, batch_key = jax.random.split(key)
        XY_test = jax.random.uniform(batch_key, shape = (n_points, n_features), minval = 0, maxval = 1)
        predictions = solver.solution(params,XY_test[:,0],XY_test[:,1])[:,0]
        true_sol = analytical_solution(XY_test)
        error_at_step = jax.numpy.mean(abs(predictions-true_sol))
        mean_absolute_error_list.append(error_at_step)

        if count%report_steps==report_steps-1:
            print("Step n°{}: ".format(count+1),' of ',len(h_list))
        count += 1 
    pickle.dump(mean_absolute_error_list,open("./NN_saves/mean_absolute_error_list", "wb"))
else:
    mean_absolute_error_list = pickle.load(open("./NN_saves/mean_absolute_error_list", "rb"))





########################################################################################################
################################################## Plot ################################################
########################################################################################################
mean_absolute_error_list = numpy.array(mean_absolute_error_list)
fig, ax = matplotlib.pyplot.subplots(2)
ax[0].plot(h_list, mean_absolute_error_list)
ax[0].set_title('MAE with respect to point spacing')
ax[0].set(xlabel='h', ylabel='MAE')
ax[1].loglog(h_list, mean_absolute_error_list)
ax[1].set_title('MAE with respect to point spacing, loglog graph')
ax[1].set(xlabel='h', ylabel='MAE')
fig.savefig("./images/MAE_with_mesh_spacing",bbox_inches = 'tight')

lin_reg_res = scipy.stats.linregress(jax.numpy.log(h_list), jax.numpy.log(mean_absolute_error_list))
print('Linear regression results :')
print(r'with $MAE = C h^{\alpha}$')
print('C = ', numpy.exp(lin_reg_res.intercept))
print(r'$\alpha$ =', lin_reg_res.slope)




###########################################################################################################################
################################################### Uniform Mesh performance ##############################################
###########################################################################################################################

"""
h_list = jax.numpy.linspace(0.01, 0.1)
mean_absolute_error_list = jax.numpy.array([])
for h_param in h_list:
    x =  jax.numpy.arange(0, 1, h_param)
    y = jax.numpy.arange(0, 1, h_param)
    xx, yy = jax.numpy.meshgrid(x, y)

    # Training du PINN
    
    mean_absolute_error_list.append(jax.numpy.mean(error))

fig, ax = matplotlib.pyplot.subplot(2)
ax[0].plot(h_list, mean_absolute_error_list)
ax[0].set_title('MAE with respect to point spacing')
ax[0].set(xlabel='h', ylabel='MAE')
ax[1].loglog(h_list, mean_absolute_error_list)
ax[1].set_title('MAE with respect to point spacing, loglog graph')
ax[1].set(xlabel='h', ylabel='MAE')

lin_reg_res = scipy.stats.linregress(jax.numpy.log(h_list), jax.numpy.log(mean_absolute_error_list))
print('Linear regression results :')
print(r'with $MAE = C h^{\alpha}$')
print('C = ', lin_reg_res.intercept)
print(r'$\alpha$ =', lin_reg_res.slope)
"""