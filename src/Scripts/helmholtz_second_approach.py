# %% [markdown]
# # Numerical solution of the Helmholtz equation
# ## Description
# 
# ### PDE
# We will try to solve the following pde:
# 
# $\Delta \psi(x, y) +k^2\psi(x, y) = -f(x, y)$ on $\Omega = [0,1]^2$  
# 
# where $f(x, y)=2k^2e^{jk(x+y)}-k^2e^{jk(x+y)} = k^2e^{jk(x+y)}$   
# 
# ### Boundary conditions    
# $\psi(0,y)=e^{jky}, \quad \psi(1,y)=e^{jk(1+y)}, \quad \psi(x,0)=e^{jkx} \quad$ and $\quad \psi(x,1)=e^{jk(x+1)} $
# 
# ### Analytical solution
# The solution $\psi_{th}$ should be $\psi_{th}(x, y)=e^{jk(x+y)}$  
# 
# ### Approximated solution
# We want to find a approximated solution $\psi_{a}(x, y)=N(x, y)$ such that described in the second Lagaris's paper
# 
# ### Loss function
# The loss to minimize here is $\mathcal{L} = \frac{1}{N}||\Delta \psi_{a}(x, y) +k^2\psi_{a}(x, y)+f(x, y) ||_2^2 + \frac{1}{M}\eta||\psi_{a}|_{\partial \Omega}-b||_2^2$, 
# 
# where N and M are the size of the training data inside the domain and boundary. $b$ is the value of each point at the boundary


############################################################################################
######################################## Libraries #########################################
############################################################################################
import jax, optax
import pickle
import numpy
import sys
sys.path.append('../')  # Add the path to the ascending directory
from Models import architectures, pinns, utils

# Set and verify device
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
print(jax.lib.xla_bridge.get_backend().platform)




############################################################################################
####################################### Parameters #########################################
############################################################################################
# Neural network parameters
config = {}
config['seed'] = 351
config['n_features'] = 2        # Input dimension (x1, x2)
config['n_targets'] = 1         # Output dimension. It's a complex number (y1 + j*y2)
config['hidden_layers'] = [50, 50, 50, 50, 50]   # Hidden layers structure
config['layers'] = [config['n_features']] + config['hidden_layers'] + [2*config['n_targets']] + [config['n_targets']]
config['eta'] = 1

# Training parameters
config['learning_rate'] = optax.linear_schedule(0.005, 0.00001, transition_steps = 50, transition_begin = 5000)
config['optimizer'] = optax.adam(config['learning_rate'])
config['checkpoint_path'] = '../Checkpoints/First_approach/Helmholtz/'
config['maximum_num_epochs'] = 50000
config['report_steps'] = 1000
config['options'] = 1           # 1: we start a new training. 2: We continue the last training. 
                                    # Other cases: We just load the last training

# Data parameters
config['n_inside'] = 100        # number of points inside the domain
config['n_bound'] = 60          # number of points at the boundary
config['domain_bounds'] = jax.numpy.column_stack(([0.0, 0.0], [1.0, 1.0]))   # minimal and maximal value of each axis (x, y)
config['img_path'] = '../Images/Second_approach/Helmholtz/'




############################################################################################
######################################### Model ############################################
############################################################################################
NN_MLP = architectures.Complex_MLP(config['seed'], config['layers'])                 
params = NN_MLP.initialize_params()            # Create the MLP
NN_eval = NN_MLP.evaluation            # Evaluation function
solver = pinns.PINN_Helmholtz_second_approach(NN_eval, config['optimizer'], config['eta'])
opt_state = config['optimizer'].init(params)




############################################################################################
######################################### Data #############################################
############################################################################################
XY_inside = utils.gen_inside_square(config['domain_bounds'], config['n_inside'])
XY_bound = utils.gen_boundary_square(config['domain_bounds'], config['n_bound'])
XY_bound_values = solver.analytical_solution(XY_bound[:,0], XY_bound[:,1])

# plot
img_path = config['img_path'] + 'domain2d.png'
title = 'Spatial domain - 2d'
__ = utils.plot_2d_domain_with_boundary(XY_inside, XY_bound, img_path, title)
print(f"Number of points inside the domain: {XY_inside.shape[0]}")
print(f"Number of points at the boundary: {XY_bound.shape[0]}")




############################################################################################
###################################### Training ############################################
############################################################################################
loss_history = []
loss_residual = []               # residual loss
loss_boundary = []               # boundary loss

print("Training start")
if config['options'] == 1:            # start a new training
    # Main loop to solve the PDE
    for ibatch in range(config['maximum_num_epochs']+1):

        loss, params, opt_state, losses = solver.train_step(params,opt_state, XY_inside, XY_bound, XY_bound_values)

        loss_residual.append(float(losses[0]))
        loss_boundary.append(float(losses[1]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%config['report_steps']==config['report_steps']-1:
            print("Epoch n°{}: ".format(ibatch+1), losssum.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open(config['checkpoint_path'] + "params", "wb"))
            pickle.dump(opt_state, open(config['checkpoint_path'] + "opt_state", "wb"))
            pickle.dump(loss_history, open(config['checkpoint_path'] + "loss_history", "wb"))
            pickle.dump(loss_residual, open(config['checkpoint_path'] + "loss_residual", "wb"))
            pickle.dump(loss_boundary, open(config['checkpoint_path'] + "loss_boundary", "wb"))

        
elif config['options'] == 2:      # continue the last training
    params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
    opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
    loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))
    loss_residual = pickle.load(open(config['checkpoint_path'] + "loss_residual", "rb"))
    loss_boundary = pickle.load(open(config['checkpoint_path'] + "loss_boundary", "rb"))
    iepoch = len(loss_history)
    
    # Main loop to solve the PDE
    for ibatch in range(iepoch, config['maximum_num_epochs']+1):

        loss, params, opt_state, losses = solver.train_step(params,opt_state, XY_inside, XY_bound, XY_bound_values)

        loss_residual.append(float(losses[0]))
        loss_boundary.append(float(losses[1]))
        losssum = jax.numpy.sum(losses)
        loss_history.append(float(losssum))

        if ibatch%config['report_steps']==config['report_steps']-1:
            print("Epoch n°{}: ".format(ibatch+1), losssum.item())

        if losssum<=numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open(config['checkpoint_path'] + "params", "wb"))
            pickle.dump(opt_state, open(config['checkpoint_path'] + "opt_state", "wb"))
            pickle.dump(loss_history, open(config['checkpoint_path'] + "loss_history", "wb"))
            pickle.dump(loss_residual, open(config['checkpoint_path'] + "loss_residual", "wb"))
            pickle.dump(loss_boundary, open(config['checkpoint_path'] + "loss_boundary", "wb"))

else:
    params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
    opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
    loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))
    loss_residual = pickle.load(open(config['checkpoint_path'] + "loss_residual", "rb"))
    loss_boundary = pickle.load(open(config['checkpoint_path'] + "loss_boundary", "rb"))




############################################################################################
################################ Loss function plot ########################################
############################################################################################
img_path = config['img_path'] + 'loss_function.png'
losses = [loss_history, loss_residual, loss_boundary]
__ = utils.plot_loss_history(losses, img_path, ['loss_sum', 'residual', 'boundary'])




############################################################################################
######################### Load best params of the training #################################
############################################################################################
params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))




############################################################################################
############################ Approximated solution plot ####################################
############################################################################################
img_path = config['img_path'] + 'approximated.png'
title = 'Approximated solution'
approximated = lambda params, x, y: solver.spatial_solution2d(params, x, y)[:, 0]
utils.plot_2d_colormesh_complex(params, approximated, config['domain_bounds'], img_path, title)




############################################################################################
############################## Analytical solution plot ####################################
############################################################################################
img_path = config['img_path'] + 'analytical.png'
title = 'Analytical solution'
analytical = lambda params, x, y: solver.analytical_solution(x, y)[:, 0]
utils.plot_2d_colormesh_complex(params, analytical, config['domain_bounds'], img_path, title)




############################################################################################
################################# Squared error plot #######################################
############################################################################################
img_path = config['img_path'] + 'squared_error.png'
title = 'Squared error'
squared_error = lambda params, x, y: jax.numpy.linalg.norm(solver.spatial_solution2d(params, x, y) - solver.analytical_solution(x, y), axis = 1)**2
MSE = utils.plot_2d_colormesh(params, squared_error, config['domain_bounds'], img_path, title, color ='turbo')
print("Mean squared error: ", MSE)