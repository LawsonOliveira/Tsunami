# %% [markdown]
# # Solving PDEs with Jax - Poisson
# ## Description
# This file contains our first approach to solve PDEs with neural networks on Jax Library.  
# 
# ### Average time of execution 
# Between 2 and 3 minutes on GPU
# 
# ### PDE
# We will try to solve the poisson Equation :  
# $-\Delta \psi(x,y) = f(x,y)$ on $\Omega = [0,1]^2$  
# 
# ### Boundary conditions   
# $\psi|_{\partial \Omega}=0$ and $f(x, y)=2\pi^2 sin(\pi x) sin(\pi y)$
# 
# ### Loss function
# The loss to minimize here is $\mathcal{L} = ||\Delta \psi(x,y) + f(x,y)||_2$  
# 
# ### Analytical solution
# The true function $\psi$ should be $\psi(x, y)=sin(\pi x) sin(\pi y)$
# 
# ### Approximated solution
# We want find a solution $\psi(x,y)=F(x,y)N(x,y)+A(x,y)$ s.t:  
# $F(x,y)=\sin(x-1)\sin(y-1)\sin(x)\sin(y)$  
# $A(x,y)=0$   
# 


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
config['n_targets'] = 1         # Output dimension
config['hidden_layers'] = [50, 50, 50, 50, 50]   # Hidden layers structure
config['layers'] = [config['n_features']] + config['hidden_layers'] + [config['n_targets']]

# Training parameters
config['learning_rate'] = optax.linear_schedule(0.005, 0.00001, transition_steps = 50, transition_begin = 5000)
config['optimizer'] = optax.adam(config['learning_rate'])
config['checkpoint_path'] = '../Checkpoints/First_approach/Poisson/'
config['maximum_num_epochs'] = 50000
config['report_steps'] = 1000
config['options'] = 1           # 1: we start a new training. 2: We continue the last training. 
                                # Other cases: We just load the last training
config['reg'] = False
# Data parameters
config['batch_size'] = 100
config['domain_bounds'] = jax.numpy.column_stack(([0.0, 0.0], [1.0, 1.0]))   # minimal and maximal value of each axis (x, y)
config['img_path'] = '../Images/First_approach/Poisson/'




############################################################################################
######################################### Model ############################################
############################################################################################
NN_MLP = architectures.Real_MLP(config['seed'], config['layers'])                 
params = NN_MLP.initialize_params()            # Create the MLP
NN_eval = NN_MLP.evaluation            # Evaluation function
solver = pinns.PINN_Poisson_first_approach(NN_eval, config['optimizer'], config['reg'])
opt_state = config['optimizer'].init(params)




############################################################################################
######################################### Data #############################################
############################################################################################
XY_train = jax.numpy.array(utils.gen_inside_square(config['domain_bounds'], config['batch_size']))

# plot
img_path = config['img_path'] + 'domain2d.png'
title = 'Spatial domain - 2d'
__ = utils.plot_2d_domain(XY_train, img_path, title)




############################################################################################
###################################### Training ############################################
############################################################################################
loss_history = []
print("Training start")
if config['options'] == 1:            # start a new training
    # Main loop to solve the PDE
    for ibatch in range(config['maximum_num_epochs']+1):
        loss, params, opt_state = solver.train_step(params, opt_state, XY_train)

        loss_history.append(float(loss))

        if (ibatch%config['report_steps']) == config['report_steps']-1:
            print("Epoch n°{}: ".format(ibatch+1), loss.item())

        if loss <= numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open(config['checkpoint_path'] + "params", "wb"))
            pickle.dump(opt_state, open(config['checkpoint_path'] + "opt_state", "wb"))
            pickle.dump(loss_history, open(config['checkpoint_path'] + "loss_history", "wb"))
        
elif config['options'] == 2:      # continue the last training
    params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
    opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
    loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))
    iepoch = len(loss_history)
    
    # Main loop to solve the PDE
    for ibatch in range(iepoch, config['maximum_num_epochs']+1):
        loss, params, opt_state = solver.train_step(params, opt_state, XY_train)

        loss_history.append(float(loss))

        if (ibatch%config['report_steps']) == config['report_steps']-1:
            print("Epoch n°{}: ".format(ibatch+1), loss.item())

        if loss <= numpy.min(loss_history): # save if the current state is the best 
            pickle.dump(params, open(config['checkpoint_path'] + "params", "wb"))
            pickle.dump(opt_state, open(config['checkpoint_path'] + "opt_state", "wb"))
            pickle.dump(loss_history, open(config['checkpoint_path'] + "loss_history", "wb"))
else:
    params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
    opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
    loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))




############################################################################################
################################ Loss function plot ########################################
############################################################################################
img_path = config['img_path'] + 'loss_function.png'
__ = utils.plot_loss_history(loss_history, img_path, ['loss'])




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
__ = utils.plot_2d_colormesh(params, approximated, config['domain_bounds'], img_path, title)




############################################################################################
############################## Analytical solution plot ####################################
############################################################################################
img_path = config['img_path'] + 'analytical.png'
title = 'Analytical solution'
analytical = lambda params, x, y: solver.analytical_solution(x, y)[:, 0]
__ = utils.plot_2d_colormesh(params, analytical, config['domain_bounds'], img_path, title)




############################################################################################
################################# Squared error plot #######################################
############################################################################################
img_path = config['img_path'] + 'squared_error.png'
title = 'Squared error'
squared_error = lambda params, x, y: (solver.spatial_solution2d(params, x, y)[:, 0] - solver.analytical_solution(x, y)[:, 0])**2
MSE = utils.plot_2d_colormesh(params, squared_error, config['domain_bounds'], img_path, title, color ='turbo')
print("Mean squared error: ", MSE)