############################################################################################
######################################## Libraries #########################################
############################################################################################

import jax, optax
import numpy

import pickle
import pyvista
import sys
sys.path.append('../')  # Add the path to the ascending directory
from pyvista import examples
from Models import architectures, utils, pinns

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
config['n_features'] = 3        # Input dimension (x1, x2)
config['n_targets'] = 1         # Output dimension. It's a complex number (y1 + j*y2)
config['hidden_layers'] = [50, 50, 50, 50, 50]   # Hidden layers structure
config['layers'] = [config['n_features']] + config['hidden_layers'] + [2*config['n_targets']] + [config['n_targets']] 

# Training parameters
config['learning_rate'] = optax.linear_schedule(0.005, 0.00001, transition_steps = 50, transition_begin = 5000)
config['optimizer'] = optax.adam(config['learning_rate'])
config['eta'] = 100
config['maximum_num_epochs'] = 100
config['report_steps'] = 100
config['options'] = 1           # 1: we start a new training. 2: We continue the last training. 
                                # Other cases: We just load the last training

# Data parameters
config['mesh'] = examples.download_woman() # examples: download_horse, download_human, download_action_figure
config['rotate_xy'] = numpy.pi/2
config['checkpoint_path'] = '../Checkpoints/Second_approach/Helmholtz3d/'
config['img_path'] = '../Images/Second_approach/Helmholtz3d/'
config['n_surface_points'] = 100000
config['n_inside_points'] = 40000 # Approximately
config['n_cube_points'] = 10000
config['incident_height'] = 0.05
config['wave_number'] = 0.5


##### Data processing
mesh = config['mesh']
domain, bounds_list, bounds_list_normals = utils.get_box_bounds3d_from_trimesh(mesh, config)
print("Number of points on the surface: ", bounds_list[0].shape[0])
print("Number of points in the domain: ", domain.shape[0])
print("Number of points on the boundary cube: ", numpy.row_stack(bounds_list).shape[0]-bounds_list[0].shape[0])


##### Domain
pl = pyvista.Plotter(off_screen=True)
#pl = pyvista.Plotter()
pl.add_mesh(mesh, show_edges=True, color ='red')
pl.add_points(numpy.row_stack(bounds_list), render_points_as_spheres=True, color ='green')
pl.renderer.background_color = 'white'
pl.camera.position = (3.0, -5.0, 1.5)
pl.camera.zoom(0.9)
#pl.show()
pl.show(screenshot=config['img_path'] + "domain.png")


##### Model initialization
NN_MLP = architectures.Complex_MLP(config['seed'], config['layers'])                 
params = NN_MLP.initialize_params()            # Create the MLP
NN_eval = NN_MLP.evaluation            # Evaluation function
solver = pinns.PINN_Helmholtz_Trimesh(config, NN_eval)
opt_state = config['optimizer'].init(params)


if "__main__" == __name__:
    ##### Training
    loss_history = []
    loss_residual = []               # residual loss
    loss_boundary = []               # boundary loss

    print("Training start")
    if config['options'] == 1:            # start a new training
        # Main loop to solve the PDE
        for ibatch in range(config['maximum_num_epochs']+1):

            loss, params, opt_state, losses = solver.train_step(params,opt_state, domain, bounds_list, bounds_list_normals)

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

            loss, params, opt_state, losses = solver.train_step(params,opt_state, domain, bounds_list, bounds_list_normals)

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


    ##### Plots
    ##### Loss function
    img_path = config['img_path'] + 'loss_function.png'
    losses = [loss_history, loss_residual, loss_boundary]
    __ = utils.plot_loss_history(losses, img_path, ['loss_sum', 'residual', 'boundary'])


    ##### Load best params
    params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))


    ##### Solution
    data = numpy.row_stack(bounds_list[1:])
    result = solver.spatial_3dsolution(params, data[:,0], data[:,1], data[:,2])
    result_real_part = numpy.array(jax.numpy.real(result))
    result_imag_part = numpy.array(jax.numpy.imag(result)) 


    ##### Plot real part
    pl = pyvista.Plotter(off_screen=True)
    #pl = pyvista.Plotter()
    pl.add_points(data, scalars=result_real_part, render_points_as_spheres=True, show_scalar_bar=True, cmap = 'turbo')
    pl.renderer.background_color = 'white'
    #pl.show()
    pl.camera.position = (3.0, -5.0, 1.5)
    pl.camera.zoom(0.9)
    pl.text_color = 'black'
    pl.show(screenshot=config['img_path'] + "real_part_box.png")


    ##### Plot imaginary part
    pl = pyvista.Plotter(off_screen=True)
    #pl = pyvista.Plotter()
    pl.add_points(numpy.row_stack(bounds_list[1:]), scalars=result_imag_part, render_points_as_spheres=True, show_scalar_bar=True, cmap = 'turbo')
    pl.renderer.background_color = 'white'
    #pl.show()
    pl.camera.position = (3.0, -5.0, 1.5)
    pl.camera.zoom(0.9)
    pl.show(screenshot=config['img_path'] + "imag_part_box.png")


    ##### Solution
    data = numpy.row_stack(bounds_list[0])
    result = solver.spatial_3dsolution(params, data[:,0], data[:,1], data[:,2])
    result_real_part = numpy.array(jax.numpy.real(result))
    result_imag_part = numpy.array(jax.numpy.imag(result)) 


    ##### Plot real part
    pl = pyvista.Plotter(off_screen=True)
    #pl = pyvista.Plotter()
    pl.add_points(data, scalars=result_real_part, render_points_as_spheres=True, show_scalar_bar=True, cmap = 'turbo')
    pl.renderer.background_color = 'white'
    #pl.show()
    pl.camera.position = (3.0, -5.0, 1.5)
    pl.camera.zoom(0.9)
    pl.text_color = 'black'
    pl.show(screenshot=config['img_path'] + "real_part_woman.png")


    ##### Plot imaginary part
    pl = pyvista.Plotter(off_screen=True)
    #pl = pyvista.Plotter()
    pl.add_points(numpy.row_stack(bounds_list[1:]), scalars=result_imag_part, render_points_as_spheres=True, show_scalar_bar=True, cmap = 'turbo')
    pl.renderer.background_color = 'white'
    #pl.show()
    pl.camera.position = (3.0, -5.0, 1.5)
    pl.camera.zoom(0.9)
    pl.show(screenshot=config['img_path'] + "imag_part_woman.png")
