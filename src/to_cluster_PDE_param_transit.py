############################################################################################
######################################## Libraries #########################################
############################################################################################
import jax, optax
import pickle
import numpy
import pandas
import os
from pyvista import examples
import time

from Models import architectures, utils, pinns, process_data

# Set and verify device
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)
print("jax.lib.xla_bridge.get_backend().platform:",jax.lib.xla_bridge.get_backend().platform,flush=True)
#jax.config.update('jax_disable_jit', True)



############################################################################################
################################### Mild slope with k fixed ################################
############################################################################################
print("\n=== Mild slope with k fixed ===\n", flush=True)
def run_mild_slope_k(incident_height=0.5,angle_direction=0.0,checkpoint_path='Checkpoints/Second_approach/Mild_slope_k_05_ih_00_ad/',load_params_path=None):
    '''
    Run a full experiment on Arcachon Basin. It solves mild slope with k fixed 
    thanks to a PINN with a loss accounting for boundary conditions.

    Parameters
    ----------
    - incident_height: float
        wave incident height in meter
    - angle_direction: float
        wave incident angle direction in radian

    '''
    # Neural network parameters
    config = {}
    config['seed'] = 351
    config['n_features'] = 2        # Input dimension (x1, x2)
    config['n_targets'] = 1         # Output dimension. It's a complex number (y1 + j*y2)
    config['hidden_layers'] = [50, 50, 50, 50, 50]   # Hidden layers structure
    config['layers'] = [config['n_features']] + config['hidden_layers'] + [2*config['n_targets']] + [config['n_targets']]
    config['eta'] = 100
    config['incident_height'] = incident_height  #In meter
    config['angle_direction'] = angle_direction #In radian
    config['characteristic'] = {'wave_vector':0.005}
    #config['characteristic'] = {'omega':0.00005}       # You can fix the wave_vector or omega

    # Training parameters
    config['learning_rate'] = optax.linear_schedule(0.005, 0.00001, transition_steps = 50, transition_begin = 5000)
    config['optimizer'] = optax.adam(config['learning_rate'])
    if not(os.path.exists(checkpoint_path)):
        os.mkdir(checkpoint_path)
    config['checkpoint_path'] = checkpoint_path
    config['maximum_num_epochs'] = 25000
    config['report_steps'] = 1000
    config['options'] = 3           # 1: we start a new training. 2: We continue the last training. 
                                        # Other cases: We just load the last training

    # Data parameters
    config['img_path'] = 'Images/Second_approach/Mild_slope/'
    config['data_path'] = '../data/Bath_and_topo/'
    config['smooth_knn'] = 7


    ##### Reading 3d data
    maritime_data = pandas.read_csv(config['data_path'] + "maritime_data.csv", sep=",", index_col=False, header=2).to_numpy()
    earth_data = pandas.read_csv(config['data_path'] + "earth_data.csv", sep=",", index_col=False, header=2).to_numpy()


    ##### Pre-process
    pre_processed_data = process_data.pre_process(maritime_data, earth_data, k_neigh = config['smooth_knn'])
    XYZ_shoal = pre_processed_data.get_smooth_data(metters_scalling=True)
    XYZ_shoal = jax.numpy.array(pre_processed_data.normalize(XYZ_shoal[XYZ_shoal[:,2]<0]))

    scale = pre_processed_data.get_scale()
    config['incident_height_normalized'] = config['incident_height']/scale

    data_coast = pre_processed_data.get_smooth_frontier(metters_scalling=True)
    data_coast = jax.numpy.array(pre_processed_data.normalize(data_coast))

    xy_bound_list, xy_bound_normals_list = utils.get_box_bounds2d(XYZ_shoal, data_coast, scale)
    xy_inside = XYZ_shoal[:,:2]


    print("config:\n",config,end="\n\n",flush=True)

    ##### Model initialization
    NN_MLP = architectures.Complex_MLP(config['seed'], config['layers'])
    if load_params_path is None:                 
        params = NN_MLP.initialize_params()            # Create the MLP
    else:
        assert os.path.exists(load_params_path), "Params to load not found"
        params = pickle.load(load_params_path, "rb")
    NN_eval = NN_MLP.evaluation            # Evaluation function
    solver = pinns.PINN_Mild_slope_second_approach(config, NN_eval, XYZ_shoal)
    opt_state = config['optimizer'].init(params)


    ##### Training
    loss_history = []
    loss_residual = []               # residual loss
    loss_boundary = []               # boundary loss
    print("Training start", flush=True)

    if config['options'] == 1:            # start a new training
        # Main loop to solve the PDE
        for ibatch in range(config['maximum_num_epochs']+1):
            print(f"batch {ibatch+1}/{config['maximum_num_epochs']+1}",flush=True)
            start_time = time.time()

            loss, params, opt_state, losses = solver.train_step(params,opt_state, xy_inside, xy_bound_list, xy_bound_normals_list)

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
            
            end_time = time.time()
            print(f"batch training took {end_time-start_time}",flush=True)

    elif config['options'] == 2:      # continue the last training
        params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
        opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
        loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))
        loss_residual = pickle.load(open(config['checkpoint_path'] + "loss_residual", "rb"))
        loss_boundary = pickle.load(open(config['checkpoint_path'] + "loss_boundary", "rb"))
        iepoch = len(loss_history)
        
        # Main loop to solve the PDE
        for ibatch in range(iepoch, config['maximum_num_epochs']+1):
            print(f"batch {ibatch+1}/{config['maximum_num_epochs']+1-iepoch}",flush=True)
            start_time = time.time()

            loss, params, opt_state, losses = solver.train_step(params, opt_state, xy_inside, xy_bound_list, xy_bound_normals_list)

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
            
            end_time = time.time()
            print(f"batch training took {end_time-start_time}",flush=True)


    else:
        params = pickle.load(open(config['checkpoint_path'] + "params", "rb"))
        opt_state = pickle.load(open(config['checkpoint_path'] + "opt_state", "rb"))
        loss_history = pickle.load(open(config['checkpoint_path'] + "loss_history", "rb"))
        loss_residual = pickle.load(open(config['checkpoint_path'] + "loss_residual", "rb"))
        loss_boundary = pickle.load(open(config['checkpoint_path'] + "loss_boundary", "rb"))
        
    print("Training is finished")

def run_transitions():
    # train 1
    checkpoint_path_conf_1 = 'Checkpoints/Second_approach/Mild_slope_k_conf_1/'
    run_mild_slope_k(
        incident_height=0.5,
        angle_direction=0.0,
        checkpoint_path=checkpoint_path_conf_1)
    # train 2
    checkpoint_path_conf_2 = 'Checkpoints/Second_approach/Mild_slope_k_conf_2/'
    run_mild_slope_k(
        incident_height=1.0,
        angle_direction=0.0,
        checkpoint_path=checkpoint_path_conf_2)
    # train 3
    checkpoint_path_conf_3 = 'Checkpoints/Second_approach/Mild_slope_k_conf_3/'
    run_mild_slope_k(
        incident_height=0.5,
        angle_direction=jax.numpy.pi/4,
        checkpoint_path=checkpoint_path_conf_3)
    # train 1 to 2
    checkpoint_path_conf_1_to_2 = 'Checkpoints/Second_approach/Mild_slope_k_conf_1_to_2/'
    run_mild_slope_k(
        incident_height=1.0,
        angle_direction=0.0,
        checkpoint_path=checkpoint_path_conf_1_to_2,
        load_params_path=checkpoint_path_conf_1+'params')
    # train 1 to 3
    checkpoint_path_conf_1_to_3 = 'Checkpoints/Second_approach/Mild_slope_k_conf_1_to_3/'
    run_mild_slope_k(
        incident_height=0.5,
        angle_direction=jax.numpy.pi/4,
        checkpoint_path=checkpoint_path_conf_1_to_3,
        load_params_path=checkpoint_path_conf_1+'params')