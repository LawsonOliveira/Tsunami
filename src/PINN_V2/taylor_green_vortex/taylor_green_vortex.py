######################################################################################################
############################################# Libraries ##############################################
######################################################################################################
import jax, optax, flax
import pickle
import functools
import matplotlib.pyplot, matplotlib.animation
import numpy

# Set and verify device
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_disable_jit', True) # Desactive the compilation for better debugging
print(jax.lib.xla_bridge.get_backend().platform)



######################################################################################################
########################################## Neural network ############################################
######################################################################################################
class MLP(flax.linen.Module):
    """
        Create a multilayer perceptron and initialize the neural network
    Inputs :
        A SEED number and the layers structure, activation function and a bool to indicates if it's occurring a training
     """
    layers: list
    training: bool

    @flax.linen.compact
    def __call__(self, x):
        """
        Compute the output of the neural network
        Parameters
        ----------
        x : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        x : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- numerical output of the neural network	(u,v,p)
        """
        
        x = flax.linen.BatchNorm(use_running_average=not self.training)(x)
        for i in range(1,len(self.layers)-1):
            x = flax.linen.Dense(self.layers[i])(x)
            x = flax.linen.BatchNorm(use_running_average=not self.training)(x)
            x = flax.linen.tanh(x)
            #x = nn.Dropout(rate=0.5, deterministic=not self.training)(x)
        x = flax.linen.Dense(self.layers[-1])(x)
        
        return x



#####################################################################################################
######################################### PDE operators class ########################################
######################################################################################################
class PDE_operators:
    """
        Class with the most common operators used to solve PDEs
    Input:
        A function that we want to compute the respective operator
    """
    
    # Class initialization
    def __init__(self, function):
        self.function = function

    # Compute the two dimensional laplacian
    @functools.partial(jax.jit, static_argnums=(0,))    
    def laplacian2d(self, params, inputs):
        """
        Compute the two dimensional laplacian. Laplacian of u and v
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        laplacian : jax.numpy.ndarray[[batch_size,batch_size]]
            -- numerical values of the laplacian applied to the inputs
        """

        @functools.partial(jax.jit)    
        def laplacian(params, inputs, output):            # Compute the laplacian of output (u or v)
            fun = lambda params, x, y, t: self.function(params, x, y, t)[:,output]

            @functools.partial(jax.jit)    
            def action(params, x, y, t):               # function to vectorize the laplacian
                a_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params, x, y, t)
                a_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params, x, y, t)
                return a_xx + a_yy

            vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
            lap = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
            return lap

        lapu = laplacian(params,inputs,0)       # laplacian for u
        lapv = laplacian(params,inputs,1)       # laplacian for v
        lapu_lapv = jax.numpy.column_stack((lapu,lapv))

        return lapu_lapv

    @functools.partial(jax.jit, static_argnums=(0,))    
    def pression_gradient(self, params, inputs):
        """
        Compute the gradient of the pression
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        gradient : jax.numpy.ndarray[[batch_size,batch_size]]
            -- numerical values of the gradient applied to the inputs
        """

        fun = lambda params, x, y, t: self.function(params, x, y, t)[:,2]

        @functools.partial(jax.jit)    
        def action(params, x, y, t):               # function to vectorize the laplacian
            p_x = jax.jacfwd(fun, 1)(params, x, y, t)
            p_y = jax.jacfwd(fun, 2)(params, x, y, t)
            return jax.numpy.column_stack((p_x, p_y))

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
        gradient = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
        gradient = gradient.reshape((gradient.shape[0], gradient.shape[2]))

        return gradient

    # Compute the two dimensional laplacian
    @functools.partial(jax.jit, static_argnums=(0,))    
    def speed_divergent(self, params, inputs):
        """
        Compute the divergent of (u,v)
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        divergent : jax.numpy.array[batch_size]
            -- numerical values of the divergent of (u,v) applied to the inputs
        """

        funu = lambda params, x, y, t: self.function(params, x, y, t)[:,0] 
        funv = lambda params, x, y, t: self.function(params, x, y, t)[:,1] 

        @functools.partial(jax.jit)    
        def actionu(params, x, y, t):              # function to vectorize the laplacian
            u_x = jax.jacfwd(funu, 1)(params, x, y, t)
            return u_x

        @functools.partial(jax.jit)    
        def actionv(params, x, y, t):              # function to vectorize the laplacian
            v_y = jax.jacfwd(funv, 2)(params, x, y, t)
            return v_y

        vec_fun = jax.vmap(actionu, in_axes = (None, 0, 0, 0))
        du_dx =  vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

        vec_fun = jax.vmap(actionv, in_axes = (None, 0, 0, 0))
        dv_dy =  vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

        divergent = du_dx + dv_dy
        divergent = divergent.reshape(-1,)

        return divergent

    @functools.partial(jax.jit, static_argnums=(0,))    
    def material_derivative(self, params, inputs):
        """
        Compute the material derivative of u and v
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        mat_derivative : jax.numpy.ndarray[[batch_size, batch_size]]
            -- numerical values of the material derivative of (u,v) applied to the inputs
        """

        result = self.function(params, inputs[:,0], inputs[:,1], inputs[:,2])

        @functools.partial(jax.jit)    
        def all_derivatives(params, inputs, output):      # compute all derivatives of output (u or v)
            fun = lambda params, x, y, t: self.function(params, x, y, t)[:,output]

            @functools.partial(jax.jit)    
            def action(params, x, y, t):                   # function to vectorize the laplacian
                a_x = jax.jacfwd(fun, 1)(params, x, y, t)
                a_y = jax.jacfwd(fun, 2)(params, x, y, t)
                a_t = jax.jacfwd(fun, 3)(params, x, y, t)
                return jax.numpy.column_stack((a_x, a_y, a_t))

            vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
            values = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

            return values.reshape((values.shape[0], values.shape[2]))

        vector = jax.numpy.column_stack((result[:,0], result[:,1], jax.numpy.ones_like(result[:,0])))

        all_first_derivatives_u = all_derivatives(params, inputs, 0)
        mat_derivative_u = jax.numpy.einsum('ij,ij->i', all_first_derivatives_u, vector)

        all_first_derivatives_v = all_derivatives(params, inputs, 1)
        mat_derivative_v = jax.numpy.einsum('ij,ij->i', all_first_derivatives_v, vector)

        mat_derivative = jax.numpy.column_stack((mat_derivative_u, mat_derivative_v))

        return mat_derivative



########################################################################################################
###################################### Physics Informed Neural Networks ################################
########################################################################################################
class PDE_operators:
    """
        Class with the most common operators used to solve PDEs
    Input:
        A function that we want to compute the respective operator
    """
    
    # Class initialization
    def __init__(self, function):
        self.function = function

    # Compute the two dimensional laplacian
    @functools.partial(jax.jit, static_argnums=(0,))    
    def laplacian2d(self, params, inputs):
        """
        Compute the two dimensional laplacian. Laplacian of u and v
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        laplacian : jax.numpy.ndarray[[batch_size,batch_size]]
            -- numerical values of the laplacian applied to the inputs
        """

        @functools.partial(jax.jit)    
        def laplacian(params, inputs, output):            # Compute the laplacian of output (u or v)
            fun = lambda params, x, y, t: self.function(params, x, y, t)[:,output]

            @functools.partial(jax.jit)    
            def action(params, x, y, t):               # function to vectorize the laplacian
                a_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params, x, y, t)
                a_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params, x, y, t)
                return a_xx + a_yy

            vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
            lap = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
            return lap

        lapu = laplacian(params,inputs,0)       # laplacian for u
        lapv = laplacian(params,inputs,1)       # laplacian for v
        lapu_lapv = jax.numpy.column_stack((lapu,lapv))

        return lapu_lapv

    @functools.partial(jax.jit, static_argnums=(0,))    
    def pression_gradient(self, params, inputs):
        """
        Compute the gradient of the pression
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        gradient : jax.numpy.ndarray[[batch_size,batch_size]]
            -- numerical values of the gradient applied to the inputs
        """

        fun = lambda params, x, y, t: self.function(params, x, y, t)[:,2]

        @functools.partial(jax.jit)    
        def action(params, x, y, t):               # function to vectorize the laplacian
            p_x = jax.jacfwd(fun, 1)(params, x, y, t)
            p_y = jax.jacfwd(fun, 2)(params, x, y, t)
            return jax.numpy.column_stack((p_x, p_y))

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
        gradient = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
        gradient = gradient.reshape((gradient.shape[0], gradient.shape[2]))

        return gradient

    # Compute the two dimensional laplacian
    @functools.partial(jax.jit, static_argnums=(0,))    
    def speed_divergent(self, params, inputs):
        """
        Compute the divergent of (u,v)
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        divergent : jax.numpy.array[batch_size]
            -- numerical values of the divergent of (u,v) applied to the inputs
        """

        funu = lambda params, x, y, t: self.function(params, x, y, t)[:,0] 
        funv = lambda params, x, y, t: self.function(params, x, y, t)[:,1] 

        @functools.partial(jax.jit)    
        def actionu(params, x, y, t):              # function to vectorize the laplacian
            u_x = jax.jacfwd(funu, 1)(params, x, y, t)
            return u_x

        @functools.partial(jax.jit)    
        def actionv(params, x, y, t):              # function to vectorize the laplacian
            v_y = jax.jacfwd(funv, 2)(params, x, y, t)
            return v_y

        vec_fun = jax.vmap(actionu, in_axes = (None, 0, 0, 0))
        du_dx =  vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

        vec_fun = jax.vmap(actionv, in_axes = (None, 0, 0, 0))
        dv_dy =  vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

        divergent = du_dx + dv_dy
        divergent = divergent.reshape(-1,)

        return divergent

    @functools.partial(jax.jit, static_argnums=(0,))    
    def material_derivative(self, params, inputs):
        """
        Compute the material derivative of u and v
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        mat_derivative : jax.numpy.ndarray[[batch_size, batch_size]]
            -- numerical values of the material derivative of (u,v) applied to the inputs
        """

        result = self.function(params, inputs[:,0], inputs[:,1], inputs[:,2])

        @functools.partial(jax.jit)    
        def all_derivatives(params, inputs, output):      # compute all derivatives of output (u or v)
            fun = lambda params, x, y, t: self.function(params, x, y, t)[:,output]

            @functools.partial(jax.jit)    
            def action(params, x, y, t):                   # function to vectorize the laplacian
                a_x = jax.jacfwd(fun, 1)(params, x, y, t)
                a_y = jax.jacfwd(fun, 2)(params, x, y, t)
                a_t = jax.jacfwd(fun, 3)(params, x, y, t)
                return jax.numpy.column_stack((a_x, a_y, a_t))

            vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
            values = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])

            return values.reshape((values.shape[0], values.shape[2]))

        vector = jax.numpy.column_stack((result[:,0], result[:,1], jax.numpy.ones_like(result[:,0])))

        all_first_derivatives_u = all_derivatives(params, inputs, 0)
        mat_derivative_u = jax.numpy.einsum('ij,ij->i', all_first_derivatives_u, vector)

        all_first_derivatives_v = all_derivatives(params, inputs, 1)
        mat_derivative_v = jax.numpy.einsum('ij,ij->i', all_first_derivatives_v, vector)

        mat_derivative = jax.numpy.column_stack((mat_derivative_u, mat_derivative_v))

        return mat_derivative



########################################################################################################
###################################### Physics Informed Neural Networks ################################
########################################################################################################
class PINN:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network
    """

    # Class initialization
    def __init__(self, NN_evaluation, optimizer):
        self.NN_evaluation = NN_evaluation
        self.optimizer = optimizer
        self.density = 1
        self.viscosity = 0.1

        self.operators = PDE_operators(self.solution)
        self.pression_gradient = self.operators.pression_gradient
        self.speed_divergent = self.operators.speed_divergent
        self.material_derivative = self.operators.material_derivative
        self.laplacian = self.operators.laplacian2d

    @functools.partial(jax.jit, static_argnums=(0,))    
    def solution(self, params, inputX, inputY, inputT):
        """
        Compute the solution of the PDE on the points (x,y,t)
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size]
            -- points on the x-axis of the mesh
        inputY : jax.numpy.array[batch_size]
            -- points on the y-axis of the mesh
        inputT : jax.numpy.array[batch_size]
            -- points on the time-axis
        Returns
        -------
        applied_solution : jax.numpy.ndarray[batch_size,batch_size,batch_size]
            -- PINN solution applied to inputs. return (u,v,p)
        """

        inputs = jax.numpy.column_stack((inputX, inputY, inputT))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)

        return NN

    # Compute the condition on boundary
    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, inputs):      
        """
        Compute the loss function at the boundary
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        bound : jax.numpy.ndarray[[batch_size, batch_size, batch_size]]
            -- (x,y,t) points from boundary
        Returns
        -------
        loss_bound : a float.64
            -- loss function applied to inputs
        """

        exact_bound = jax.numpy.column_stack((jax.numpy.cos(inputs[:,0])*jax.numpy.sin(inputs[:,1])*jax.numpy.exp(-2*self.viscosity*inputs[:,2]), -jax.numpy.cos(inputs[:,1])*jax.numpy.sin(inputs[:,0])*jax.numpy.exp(-2*self.viscosity*inputs[:,2]), -self.density/4*(jax.numpy.cos(2*inputs[:,0]) + jax.numpy.cos(2*inputs[:,1]))*jax.numpy.exp(-2*self.viscosity*inputs[:,2])**2))
        preds_bound = self.solution(params, inputs[:,0], inputs[:,1], inputs[:,2])
        loss_bound = jax.numpy.sum(jax.numpy.linalg.norm((preds_bound - exact_bound), axis = 0))

        return loss_bound

    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_residual_pde(self, params, inputs):
        """
        Compute the residual of the pde
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside : jax.numpy.ndarray[[batch_size, batch_size, batch_size]]
            -- (x,y,t) points from the mesh
        Returns
        -------
        loss_residual : a float.64
            -- loss function applied to inputs
        """

        loss_continuity = jax.numpy.linalg.norm(self.speed_divergent(params, inputs), axis = 0)
        preds_momentum = self.material_derivative(params, inputs) + self.pression_gradient(params, inputs)/self.density - self.laplacian(params, inputs)*self.viscosity
        loss_momentum = jax.numpy.sum(jax.numpy.linalg.norm((preds_momentum), axis=0))
        loss_residual = loss_momentum + loss_continuity

        return loss_residual

    # Compute the condition on boundary
    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_initial_time(self, params, inputs):      
        """
        Compute the loss function at the initial conditions
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, batch_size, batch_size]]
            -- (x,y,t) points from initial condition
        Returns
        -------
        loss_initial : a float.64
            -- loss function applied to inputs
        """

        preds_initial = self.solution(params, inputs[:,0], inputs[:,1], inputs[:,2])
        exact_initial = jax.numpy.column_stack((jax.numpy.cos(inputs[:,0])*jax.numpy.sin(inputs[:,1]), -jax.numpy.sin(inputs[:,0])*jax.numpy.cos(inputs[:,1]), -self.density*(jax.numpy.cos(2*inputs[:,0]) + jax.numpy.cos(2*inputs[:,1]))/4))
        loss_initial = jax.numpy.sum(jax.numpy.linalg.norm((preds_initial - exact_initial), axis = 0))

        return loss_initial

    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_function(self, params, inside, bound, initial):
        """
        Compute the sum of each loss function
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside : jax.numpy.ndarray[[batch_size, batch_size,batch_size]]
            -- (x,y,t) points from the mesh
        bound : jax.numpy.ndarray[[batch_size, batch_size,batch_size]]
            -- (x,y,t) points from boundary
        initial : jax.numpy.ndarray[[batch_size, batch_size, batch_size]]
            -- (x,y,t) points from initial condition
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        losses : numpy.array(loss_residual, loss_b, loss_i)
            -- current values of each loss function
        """

        loss_bound = self.loss_boundary(params, bound)
        loss_initial = self.loss_initial_time(params, initial)
        loss_residual = self.loss_residual_pde(params, inside)
        loss_sum = loss_bound+loss_initial+loss_residual
        losses = jax.numpy.array([loss_residual, loss_bound, loss_initial])

        return loss_sum, losses

    @functools.partial(jax.jit, static_argnums=(0,))    
    def train_step(self, params, opt_state, inside, bound, initial):
        """
        Make just one step of the training
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside : jax.numpy.ndarray[[batch_size, batch_size,batch_size]]
            -- (x,y,t) points from the mesh
        bound : jax.numpy.ndarray[[batch_size, batch_size,batch_size]]
            -- (x,y,t) points from boundary
        initial : jax.numpy.ndarray[[batch_size, batch_size, batch_size]]
            -- (x,y,t) points from initial condition
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : dictionary with the keys (loss_m, loss_b, loss_i)
            -- current values of each loss function
        """

        (loss,losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside, bound, initial)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state, losses





######################################################################################################
############################################# Parameters #############################################
######################################################################################################
# Neural network parameters
SEED = 351
n_features, n_targets = 3, 3            # Input and output dimension 
layers = [n_features, 50, 50, 50, 50, 50, n_targets]             # Layers structure
lr_scheduler = optax.linear_schedule(0.005, 0.00001, transition_steps = 50, transition_begin = 5000)
#lr_scheduler = 0.00001      # learning rate
optimizer = optax.adam(lr_scheduler)

# Training parameters
maximum_num_epochs = 44000
report_steps = 1
options = 2 # If 1 we start a new training

# Data parameters
N_inside = 4096        # number of points inside the mesh
N_bound = 500           # number of points at the boundary
domain_bounds = jax.numpy.column_stack(([0, 0, 0], [2*jax.numpy.pi, 2*jax.numpy.pi, 2*jax.numpy.pi]))        # minimal and maximal value of each axis (x,y,t)

# Animation parameters 
options_animation = False   # if true, we do and plot the animation




########################################################################################################
########################################## Initialization ##############################################
########################################################################################################
training_model = MLP(layers, training=True)     
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(SEED), 3)
x = jax.random.uniform(key1, (10, layers[0]))
params = training_model.init({'params': key2}, x)

eval_model = MLP(layers, training=False)
NN_eval = eval_model.apply   # Evaluate function

solver = PINN(NN_eval, optimizer)
opt_state = optimizer.init(params)





########################################################################################################
#################################################### Data ##############################################
########################################################################################################
#### Boundary data
X, Y = jax.numpy.meshgrid(jax.numpy.linspace(domain_bounds[0,0],domain_bounds[0,1],N_bound//4),jax.numpy.linspace(domain_bounds[1,0],domain_bounds[1,1],2))
X, Y = X.flatten(), Y.flatten()
left_right = jax.numpy.column_stack((X,Y))
front_behind = jax.numpy.column_stack((Y,X))
XY_bound = jax.numpy.concatenate((left_right,front_behind))
XY_bound = jax.numpy.column_stack((XY_bound[:,0],XY_bound[:,1]))

#### Inside data
ran_key, batch_key = jax.random.split(jax.random.PRNGKey(0))
XY_inside = jax.random.uniform(batch_key, shape=(N_inside, n_features-1), minval=domain_bounds[:2,0], maxval=domain_bounds[:2,1])




########################################################################################################
################################################## Data plot ###########################################
########################################################################################################
fig, ax = matplotlib.pyplot.subplots()
title = ax.set_title('Spatial domain')
graph = matplotlib.pyplot.scatter(XY_bound[:,0],XY_bound[:,1],color='b',s=1)
graph = matplotlib.pyplot.scatter(XY_inside[:,0],XY_inside[:,1],color='r',s=1)
__ = ax.legend(['Boundary','Inside'])
matplotlib.pyplot.savefig('./images/domain.png', bbox_inches = 'tight')
matplotlib.pyplot.show()  





########################################################################################################
############################################ Train #####################################################
########################################################################################################
print("Training start")
if options == 1:            # begin a new training
    loss_history = []
    loss_r = []               # residual loss
    loss_b = []               # boundary loss
    loss_i = []               # boundary initial condition

    # Main loop to solve the PDE
    for ibatch in range(maximum_num_epochs+1):
        XYT_bound = jax.numpy.column_stack((XY_bound[:,0],XY_bound[:,1],numpy.random.uniform(domain_bounds[2,0],domain_bounds[2,1],size=(N_bound,))))
        XYT_inside = jax.numpy.column_stack((XY_inside[:,0],XY_inside[:,1],numpy.random.uniform(domain_bounds[2,0],domain_bounds[2,1],size=(N_inside,))))
        XYT_initial = jax.numpy.column_stack((XY_inside[:,0],XY_inside[:,1],jax.numpy.zeros(N_inside)))

        loss, params, opt_state, losses = solver.train_step(params,opt_state, XYT_inside, XYT_bound, XYT_initial)

        loss_r.append(float(losses[0]))
        loss_b.append(float(losses[1]))
        loss_i.append(float(losses[2]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%report_steps==report_steps-1:
            print("Epoch n°{}: ".format(ibatch+1), losssum.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open("./NN_saves/params_checkpoint_vortex", "wb"))
            pickle.dump(opt_state, open("./NN_saves/opt_state_checkpoint_vortex", "wb"))
            pickle.dump(loss_history, open("./NN_saves/loss_history_vortex", "wb"))
            pickle.dump(loss_r, open("./NN_saves/loss_r_vortex", "wb"))
            pickle.dump(loss_b, open("./NN_saves/loss_b_vortex", "wb"))
            pickle.dump(loss_i, open("./NN_saves/loss_i_vortex", "wb"))

elif options == 2:      # continue the last training
    params = pickle.load(open("./NN_saves/params_checkpoint_vortex", "rb"))
    opt_state = pickle.load(open("./NN_saves/opt_state_checkpoint_vortex", "rb"))
    loss_history = pickle.load(open("./NN_saves/loss_history_vortex", "rb"))
    loss_r = pickle.load(open("./NN_saves/loss_r_vortex", "rb"))
    loss_b = pickle.load(open("./NN_saves/loss_b_vortex", "rb"))
    loss_i = pickle.load(open("./NN_saves/loss_i_vortex", "rb"))
    iepoch = len(loss_history)
    
    # Main loop to solve the PDE
    for ibatch in range(iepoch,maximum_num_epochs+1):
        XYT_bound = jax.numpy.column_stack((XY_bound[:,0],XY_bound[:,1],numpy.random.uniform(domain_bounds[2,0],domain_bounds[2,1],size=(N_bound,))))
        XYT_inside = jax.numpy.column_stack((XY_inside[:,0],XY_inside[:,1],numpy.random.uniform(domain_bounds[2,0],domain_bounds[2,1],size=(N_inside,))))
        XYT_initial = jax.numpy.column_stack((XY_inside[:,0],XY_inside[:,1],jax.numpy.zeros(N_inside)))

        loss, params, opt_state, losses = solver.train_step(params,opt_state, XYT_inside, XYT_bound, XYT_initial)

        loss_r.append(float(losses[0]))
        loss_b.append(float(losses[1]))
        loss_i.append(float(losses[2]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%report_steps==report_steps-1:
            print("Epoch n°{}: ".format(ibatch+1), losssum.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open("./NN_saves/params_checkpoint_vortex", "wb"))
            pickle.dump(opt_state, open("./NN_saves/opt_state_checkpoint_vortex", "wb"))
            pickle.dump(loss_history, open("./NN_saves/loss_history_vortex", "wb"))
            pickle.dump(loss_r, open("./NN_saves/loss_r_vortex", "wb"))
            pickle.dump(loss_b, open("./NN_saves/loss_b_vortex", "wb"))
            pickle.dump(loss_i, open("./NN_saves/loss_i_vortex", "wb"))





########################################################################################################
##################################### Plot loss function ###############################################
########################################################################################################
fig, ax = matplotlib.pyplot.subplots(1, 1)
__ = ax.plot(numpy.log10(loss_history))
__ = ax.plot(numpy.log10(numpy.array(loss_r)))
__ = ax.plot(numpy.log10(numpy.array(loss_b)))
__ = ax.plot(numpy.log10(numpy.array(loss_i)))
xlabel = ax.set_xlabel(r'${\rm Step}$')
ylabel = ax.set_ylabel(r'$\log_{10}{\rm (loss)}$')
title = ax.set_title(r'${\rm Training}$')
ax.legend(['loss_sum','residual','boundary','initial_cond'])
matplotlib.pyplot.savefig('./images/loss_function.png', bbox_inches = 'tight')
matplotlib.pyplot.show()




########################################################################################################
############################################ True u ####################################################
########################################################################################################
def true_solution(x,y,t, viscosity=0.1, density=1):

    u = jax.numpy.cos(x)*jax.numpy.sin(y)*jax.numpy.exp(-2*viscosity*t)
    v = -jax.numpy.sin(x)*jax.numpy.cos(y)*jax.numpy.exp(-2*viscosity*t)
    p = -density/4*(jax.numpy.cos(2*x)+jax.numpy.cos(2*y))*jax.numpy.exp(-2*viscosity*t)**2
    true_sol = jax.numpy.column_stack((u,v,p))

    return true_sol




########################################################################################################
###################################### Mean squared error ##############################################
########################################################################################################
ran_key, batch_key = jax.random.split(jax.random.PRNGKey(1178))
XYT_test = jax.random.uniform(batch_key, shape=(10000, n_features), minval=domain_bounds[2,0], maxval=domain_bounds[2,1])
x, y, t = XYT_test[:,0], XYT_test[:,1], XYT_test[:,2]
mean_error = numpy.mean(((true_solution(x, y, t) - solver.solution(params,x, y, t)))**2, axis=0)
print('mean squared error u: ', mean_error[0],'\nmean squared error v: ',mean_error[1],'\nmean squared error p: ',mean_error[2])



########################################################################################################
############################################ Animation #################################################
########################################################################################################
output = 2 # 0--->u, 1--->v, 2--->p

if options_animation==True:
    npoints = 300
    frames = 100

    x, y = numpy.meshgrid(numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], npoints),numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], npoints))
    t = numpy.repeat(numpy.linspace(domain_bounds[2,0], domain_bounds[2,1], frames), npoints).reshape(frames, npoints)
    values = numpy.zeros((npoints, npoints))

    def update_graph(num):
        print("Animating: {} out of {}".format(num+1, frames), end='\r')
        for i in range(npoints):
            values[i,:] = solver.solution(params, x[i,:], y[i,:], t[num,:])[:,output]
        title.set_text('Taylor-Green Vortex 2d - p, time step = {}'.format(num))
        graph.set_array(values.flatten())

    fig, ax = matplotlib.pyplot.subplots()
    title = ax.set_title('Taylor-Green Vortex 2d - p')

    for i in range(npoints):
        values[i,:] = solver.solution(params, x[i,:], y[i,:], t[0,:])[:,output]

    graph = matplotlib.pyplot.pcolormesh(x, y, values, cmap = 'seismic')
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames = frames, blit = False)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show()  
    ani.save('./images/taylor_green_vortex_p.gif', fps = 25)




########################################################################################################
#################################### Animation Absolut error ###########################################
########################################################################################################
if options_animation==True:
    npoints = 300
    frames = 100

    x, y = numpy.meshgrid(numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], npoints),numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], npoints))
    t = numpy.repeat(numpy.linspace(domain_bounds[2,0], domain_bounds[2,1], frames), npoints).reshape(frames, npoints)
    values = numpy.zeros((npoints, npoints))

    def update_graph(num):
        print("Animating: {} out of {}".format(num+1, frames), end='\r')
        for i in range(npoints):
            values[i,:] = ((true_solution(x[i,:], y[i,:], t[num,:])[:,output]-solver.solution(params, x[i,:], y[i,:], t[num,:])[:,output]))**2
        title.set_text('Taylor-Green Vortex 2d - squared error p, time step = {}'.format(num))
        graph.set_array(values.flatten())

    fig, ax = matplotlib.pyplot.subplots()
    title = ax.set_title('Taylor-Green Vortex 2d - squared error p')

    for i in range(npoints):
        values[i,:] = ((true_solution(x[i,:], y[i,:], t[0,:])[:,output]-solver.solution(params, x[i,:], y[i,:], t[0,:])[:,output]))**2

    graph = matplotlib.pyplot.pcolormesh(x, y, values, cmap = 'viridis')
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames = frames, blit = False)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show()  
    ani.save('./images/squared_error_p.gif', fps = 25)

