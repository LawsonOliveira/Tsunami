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
class PDE_operators1d:
    """
        Class with the most common operators used to solve PDEs
    Input:
        A function that we want to compute the respective operator
    """
    
    # Class initialization
    def __init__(self, function):
        self.function = function

    @functools.partial(jax.jit, static_argnums=(0,))    
    def laplacian(self, params, inputs):
        """
        Compute the laplacian of u
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        laplacian : jax.numpy.array[batch_size]
            -- numerical values of the laplacian applied to the inputs
        """

        fun = lambda params, x, t: self.function(params, x, t)[:,0]

        @functools.partial(jax.jit)    
        def action(params,x,t):         # function to vectorize the laplacian
            u_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params, x, t)
            u_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params, x, t)
            return u_xx + u_yy

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))
        lapu = vec_fun(params, inputs[:,0], inputs[:,1])

        return lapu


    # Compute material derivative
    @functools.partial(jax.jit, static_argnums=(0,))    
    def material_derivative(self, params, inputs):
        """
        Compute the material derivative of u
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size,batch_size,batch_size]]
            -- coordinates and time  (x,y,t)
        Returns
        -------
        mat_derivative : jax.numpy.array[batch_size]
            -- numerical values of the material derivative of u applied to the inputs
        """

        result = self.function(params, inputs[:,0], inputs[:,1])

        @functools.partial(jax.jit)    
        def all_derivatives(params, inputs):
            fun = lambda params, x, t: self.function(params, x, t)[:,0]

            @functools.partial(jax.jit)    
            def action(params, x, t):                     # function to vectorize the laplacian
                u_x = jax.jacfwd(fun, 1)(params, x, t)
                u_t = jax.jacfwd(fun, 2)(params, x, t)
                return jax.numpy.column_stack((u_x, u_t))

            vec_fun = jax.vmap(action, in_axes = (None, 0, 0))
            derivatives = vec_fun(params, inputs[:,0], inputs[:,1])
            derivatives = derivatives.reshape(derivatives.shape[0], derivatives.shape[2])
            return derivatives

        vector = jax.numpy.column_stack((result[:,0], jax.numpy.ones_like(result[:,0])))
        all_first_derivatives_u = all_derivatives(params, inputs)
        mat_derivative_u = jax.numpy.einsum('ij,ij->i', all_first_derivatives_u, vector).reshape(-1,1)

        return mat_derivative_u



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

        self.operators = PDE_operators1d(self.solution)
        self.k_coeff = jax.numpy.pi/2
        self.laplacian = self.operators.laplacian

    # Compute the solution of the PDE on the points (x,y)
    @functools.partial(jax.jit, static_argnums=(0,))    
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
        applied_solution : jax.numpy.ndarray[batch_size,batch_size,batch_size]
            -- PINN solution applied to inputs. return u
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN_output = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        return NN_output

    # Loss function respective to burger equation
    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_equation(self, params, inputs):
        """
        Compute the residual of the pde
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from the mesh
        Returns
        -------
        loss_residual : a float.64
            -- loss function applied to inputs
        """

        preds_loss = self.laplacian(params, inputs) + self.k_coeff**2*self.solution(params, inputs[:,0], inputs[:,1])
        loss_value = jax.numpy.linalg.norm(preds_loss)
        
        return loss_value

    # Loss function respective to boundary
    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, inputs):
        """
        Compute the loss function at the boundary
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        bound : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from boundary
        Returns
        -------
        loss_bound : a float.64
            -- loss function applied to inputs
        """

        exact_bound = jax.numpy.sin(self.k_coeff*inputs[:,0])*jax.numpy.sin(self.k_coeff*inputs[:,1])
        preds_bound = self.solution(params, inputs[:,0], inputs[:,1])[:,0]
        loss_bound = jax.numpy.linalg.norm(preds_bound - exact_bound)
        
        return loss_bound

    # Loss function respective to initial time
    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_initial_time(self, params, inputs):  
        """
        Compute the loss function at the initial conditions
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,t) points from initial condition
        Returns
        -------
        loss_initial : a float.64
            -- loss function applied to inputs
        """

        exact = -jax.numpy.sin(jax.numpy.pi*inputs[:,0])
        preds_initial_time = self.solution(params, inputs[:,0], inputs[:,1])[:,0]
        loss_initial_time = jax.numpy.linalg.norm(preds_initial_time - exact)

        return loss_initial_time

    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_function(self, params, inside, bound):
        """
        Compute the sum of each loss function
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from the mesh
        bound : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from boundary
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        losses : dictionary with the keys (loss_i, loss_b)
            -- current values of each loss function
        """

        loss_bound = self.loss_boundary(params, bound)
        loss_inside = self.loss_equation(params, inside)
        loss_sum = loss_bound+loss_inside
        losses = jax.numpy.array([loss_inside, loss_bound])

        return loss_sum, losses

    @functools.partial(jax.jit, static_argnums=(0,))    
    def train_step(self, params, opt_state, inside, bound):
        """
        Make just one step of the training
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from the mesh
        bound : jax.numpy.ndarray[[batch_size, batch_size]]
            -- (x,y) points from boundary
        Returns
        -------
        loss : a float.64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : dictionary with the keys (loss_i, loss_b)
            -- current values of each loss function
        """

        (loss,losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside, bound)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state, losses



######################################################################################################
############################################# Parameters #############################################
######################################################################################################
# Neural network parameters
SEED = 351
n_features, n_targets = 2, 1            # Input and output dimension 
layers = [n_features, 50, 50, 50, 50, 50, n_targets]             # Layers structure
lr_scheduler = optax.linear_schedule(0.0005, 0.00001, transition_steps = 50, transition_begin = 5000)
#lr_scheduler = 0.00001               # learning rate
optimizer = optax.adam(lr_scheduler)

# Training parameters
maximum_num_epochs = 10000       
report_steps = 1000
options = 1 # If 1 we start a new training

# Data parameters
N_inside = 64                # number of points inside the mesh
N_bound = 32                   # number of points at the boundary
domain_bounds = jax.numpy.vstack(([0,11], [0,25]))      # minimal and maximal value of each axis (x,y)



########################################################################################################
########################################## Initialization ##############################################
########################################################################################################
training_model = MLP(layers,training=True)     
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(SEED), 3)
x = jax.random.uniform(key1, (10, layers[0]))
params = training_model.init({'params': key2}, x)

eval_model = MLP(layers,training=False)
NN_eval = eval_model.apply   # Evaluate function

solver = PINN(NN_eval, optimizer)
opt_state = optimizer.init(params)




########################################################################################################
#################################################### Data ##############################################
########################################################################################################
#### Boundary data
## Free boundary top/bottom
X, Y = jax.numpy.meshgrid(jax.numpy.linspace(domain_bounds[0,0],domain_bounds[0,1],N_bound//4),jax.numpy.linspace(domain_bounds[1,0],domain_bounds[1,1],2))
X, Y = X.flatten(), Y.flatten()
top_bottom = jax.numpy.column_stack((X,Y))

## Free boundary right
X, Y = jax.numpy.meshgrid(jax.numpy.array(domain_bounds[0,1]),jax.numpy.linspace(domain_bounds[1,0],domain_bounds[1,1],N_bound//4))
X, Y = X.flatten(), Y.flatten()
right = jax.numpy.column_stack((X,Y))

## Breakwater left
X, Y = jax.numpy.meshgrid(jax.numpy.array(domain_bounds[0,0]),jax.numpy.concatenate(jax.numpy.linspace(0,10,N_bound//10), jax.numpy.linspace(15,25,N_bound//10)))
X, Y = X.flatten(), Y.flatten()
XY_left = jax.numpy.column_stack((X,Y))

## Breakwater opening
X, Y = jax.numpy.meshgrid(jax.numpy.array(domain_bounds[0,0]),jax.numpy.linspace(10,15,N_bound//20))
X, Y = X.flatten(), Y.flatten()
XY_left_open = jax.numpy.column_stack((X,Y))

XY_bound_absorbing = jax.numpy.concatenate((right, top_bottom))

#### Inside data
ran_key, batch_key = jax.random.split(jax.random.PRNGKey(0))
XY_inside = jax.random.uniform(batch_key, shape=(N_inside, n_features-1), minval=domain_bounds[:2,0], maxval=domain_bounds[:2,1])



fig, ax = matplotlib.pyplot.subplots()
fig.set_size_inches(18.5, 10.5)
title = ax.set_title('Spatial domain - 2d')
graph = matplotlib.pyplot.scatter(XY_bound_absorbing[:,0],XY_bound_absorbing[:,1],color='red',s=5)
graph = matplotlib.pyplot.scatter(XY_left_open[:,0],XY_left_open[:,1],color='green',s=5)
graph = matplotlib.pyplot.scatter(XY_left[:,0],XY_left[:,1],color='black',s=5)
graph = matplotlib.pyplot.scatter(XY_inside[:,0],XY_inside[:,1], color='gray', s=5)
__ = ax.legend(['Reflective','Flow-in','Flow-out','Inside'])
print('Number of points on the boundary:', XY_bound_absorbing.shape[0]+XY_left_open.shape[0]+XY_left.shape[0])
print('Number of points inside the domain:', XY_inside.shape[0])

matplotlib.pyplot.savefig('./images/domain2d.png', facecolor='white', bbox_inches = 'tight')
matplotlib.pyplot.show()  

########################################################################################################
############################################ Train #####################################################
########################################################################################################
print("Training start")
if options == 1:            # begin a new training
    loss_history = []
    loss_i = []               # residual loss
    loss_b = []               # boundary loss

    # Main loop to solve the PDE
    for ibatch in range(maximum_num_epochs+1):
        loss, params, opt_state, losses = solver.train_step(params,opt_state, XY_inside, XY_bound)

        loss_i.append(float(losses[0]))
        loss_b.append(float(losses[1]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%report_steps == report_steps-1:
            print("Epoch n°{}: ".format(ibatch+1), loss.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
                pickle.dump(params, open("./NN_saves/params_checkpoint_mild_slope", "wb"))
                pickle.dump(opt_state, open("./NN_saves/opt_state_checkpoint_mild_slope", "wb"))
                pickle.dump(loss_history, open("./NN_saves/loss_history_mild_slope", "wb"))
                pickle.dump(loss_i, open("./NN_saves/loss_i_mild_slope", "wb"))
                pickle.dump(loss_b, open("./NN_saves/loss_b_mild_slope", "wb"))

elif options == 2:   # continue the last training
    params = pickle.load(open("./NN_saves/params_checkpoint_mild_slope", "rb"))
    opt_state = pickle.load(open("./NN_saves/opt_state_checkpoint_mild_slope", "rb"))
    loss_history = pickle.load(open("./NN_saves/loss_history_mild_slope", "rb"))
    loss_i = pickle.load(open("./NN_saves/loss_i_mild_slope", "rb"))
    loss_b = pickle.load(open("./NN_saves/loss_b_mild_slope", "rb"))
    iepoch = len(loss_history)

    # Main loop to solve the PDE
    for ibatch in range(iepoch, maximum_num_epochs+1):
        loss, params, opt_state, losses = solver.train_step(params,opt_state, XY_inside, XY_bound)

        loss_i.append(float(losses[0]))
        loss_b.append(float(losses[1]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%report_steps==report_steps-1:
            print("Epoch n°{}: ".format(ibatch+1), loss.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
                pickle.dump(params, open("./NN_saves/params_checkpoint_mild_slope", "wb"))
                pickle.dump(opt_state, open("./NN_saves/opt_state_checkpoint_mild_slope", "wb"))
                pickle.dump(loss_history, open("./NN_saves/loss_history_mild_slope", "wb"))
                pickle.dump(loss_i, open("./NN_saves/loss_i_mild_slope", "wb"))
                pickle.dump(loss_b, open("./NN_saves/loss_b_mild_slope", "wb"))



########################################################################################################
##################################### Plot loss function ###############################################
########################################################################################################
fig, ax = matplotlib.pyplot.subplots(1, 1)
__ = ax.plot(numpy.log10(loss_history))
__ = ax.plot(numpy.log10(numpy.array(loss_i)))
__ = ax.plot(numpy.log10(numpy.array(loss_b)))
xlabel = ax.set_xlabel(r'${\rm Step}$')
ylabel = ax.set_ylabel(r'$\log_{10}{\rm (loss)}$')
title = ax.set_title(r'${\rm Training}$')
ax.legend(['loss_sum','residual','boundary','initial_cond'])
matplotlib.pyplot.show()



########################################################################################################
############################################# Solution #################################################
########################################################################################################
npoints = 1000
values = numpy.zeros((npoints,npoints))

x, y = numpy.meshgrid(numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], npoints), numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], npoints))
fig, ax = matplotlib.pyplot.subplots()
title = ax.set_title('Mild slope equation 2d')

for i in range(npoints):
    print("Plotting: {} out of {}".format(i+1, npoints), end='\r')
    values[i,:] = solver.solution(params, x[i,:], y[i,:])[:,0]

graph = matplotlib.pyplot.pcolormesh(x, y, values, cmap = 'rainbow')
matplotlib.pyplot.colorbar()
matplotlib.pyplot.savefig('mild_slope.png', bbox_inches = 'tight')
matplotlib.pyplot.show()  


