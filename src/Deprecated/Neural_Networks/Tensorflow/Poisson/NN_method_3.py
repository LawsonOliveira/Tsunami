#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8

from matplotlib import cm
from sympy import Matrix
import sympy as sm
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from time import time

################### check tensorflow is using GPU ###################
from tensorflow.python.client import device_lib
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(physical_devices))
print(device_lib.list_local_devices())
# what if empty...
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# On windows systems you cannont install NCCL that is required for multi GPU
# So we need to follow hierarchical copy method or reduce to single GPU (less efficient than the former)
strategy = tf.distribute.MirroredStrategy(
    devices=['GPU:0'], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

DTYPE = 'float32'

tf.keras.backend.set_floatx(DTYPE)

################### proper to the computer used ###################
__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\ncwd:', os.getcwd())
os.chdir(__file__)
print('changed to:', os.getcwd(), '\n')


################### Create the model ###################

def generate_model(l_units, noise=False):
    # méthode API Sequential
    n_hidden = len(l_units)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(2))
    ])
    if noise:
        model.add(keras.layers.GaussianNoise(stddev=1e-4))
    for i in range(n_hidden):
        model.add(keras.layers.Dense(
            l_units[i], activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(1, use_bias=False))
    # use_bias=False otherwise returns error after
    # May be the cause of a stagnation in the validation ? (can circumvent by creating a layer class only adding a bias)
    model.summary()
    return model


def test_0():
    model = generate_model()

################### define the EDP to solve ###################
# Given EDO


def f(X):
    return tf.sin(np.pi*X[:, 0])*tf.sin(np.pi*X[:, 1])


def boundary_conditions(X):
    return 0


def residual(du_dxx, du_dyy, f_ind):
    return du_dxx+du_dyy+f_ind


def differentiate(model, x):
    with tf.GradientTape(persistent=True) as tape:
        x1, x2 = x[:, 0:1], x[:, 1:2]
        tape.watch(x1)
        tape.watch(x2)
        u = model(tf.stack([x1[:, 0], x2[:, 0]], axis=1))
        du_dx = tape.gradient(u, x1)
        du_dy = tape.gradient(u, x2)
    du_dxx = tape.gradient(du_dx, x1)
    du_dyy = tape.gradient(du_dy, x2)
    return du_dx, du_dxx, du_dy, du_dyy


################### mesh the domain ###################
# can take a long time with 1000 (2 minutes), can take 100 instead

print('begin generating mesh')
start_mesh = time()
grid_length = 1000

X = np.linspace(0, 1, grid_length, endpoint=True)
Y = np.linspace(0, 1, grid_length, endpoint=True)
tf_coords = tf.convert_to_tensor(
    [tf.constant([x, y], dtype=DTYPE) for x in X for y in Y])
tf_boundary_coords = tf.convert_to_tensor([tf.constant([x, y], dtype=DTYPE) for x in [
                                          0, 1] for y in Y] + [tf.constant([x, y], dtype=DTYPE) for y in [0, 1] for x in X])

end_mesh = time()
duration_mesh = end_mesh-start_mesh
print(f'end generating mesh, duration {duration_mesh}')
#######################################################################################################
################### Method 3: g automatically respects the boundary conditions ########################
# article : 1997_Artificial_neural_networks_for_solving_ordinary_and_partial_differential_equations.pdf

################### Set F here ###################
# Dummy F


x, y = sm.symbols('x,y')


def expr_dummy_F():
    return x*(1-x)*y*(1-y)


expr_F = expr_dummy_F()
dexpr_F_dx = sm.diff(expr_F, x, 1)
dexpr_F_dxx = sm.diff(dexpr_F_dx, x, 1)
dexpr_F_dy = sm.diff(expr_F, y, 1)
dexpr_F_dyy = sm.diff(dexpr_F_dy, y, 1)


# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_F = sm.lambdify([x, y], Matrix([expr_F]), 'numpy')
dexpr_F_dx = sm.lambdify([x, y], Matrix([dexpr_F_dx]), 'numpy')
dexpr_F_dxx = sm.lambdify([x, y], Matrix([dexpr_F_dxx]), 'numpy')
dexpr_F_dy = sm.lambdify([x, y], Matrix([dexpr_F_dy]), 'numpy')
dexpr_F_dyy = sm.lambdify([x, y], Matrix([dexpr_F_dyy]), 'numpy')


def evaluate_F_and_diff(X):
    F = tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)
    dF_dx = tf.expand_dims(dexpr_F_dx(X[:, 0], X[:, 1]), axis=-1)
    dF_dxx = tf.expand_dims(dexpr_F_dxx(X[:, 0], X[:, 1]), axis=-1)
    dF_dy = tf.expand_dims(dexpr_F_dy(X[:, 0], X[:, 1]), axis=-1)
    dF_dyy = tf.expand_dims(dexpr_F_dyy(X[:, 0], X[:, 1]), axis=-1)

    return F, dF_dx, dF_dxx, dF_dy, dF_dyy

 # oddly enough expr_F and dexpr_F_d... do not have the same output


# # #### F of F_functions


# frontier_coords = Pstud._set_coords_rectangle(1, 1, 10)

# l_orders = [(1, 0), (2, 0), (0, 1), (0, 2)]
# strfn = 'sinxpy_real'
# F = F2D(frontier_coords, strfn, l_orders=l_orders)

# # prepare to infer on large matrices :
# F.expr = sm.lambdify(F.variables, Matrix([F.expr]), 'numpy')
# for t_order in l_orders:
#     F.reduced_tab_diff[F.dico_order_to_index[t_order]] = sm.lambdify(
#         F.variables, F.reduced_tab_diff[F.dico_order_to_index[t_order]], 'numpy')


# def evaluate_F_and_diff(X):
#     '''
#     evaluate F and its differentiates get in F.reduced_tab_diff
#     Variables:
#     -X: an array or tensor tf of the coordinates

#     Returns:
#     -l_eval: list of the evaluations. To know which element corresponds to which order, use F.dico_order_to_index and increment the values of 1.

#     remark: to add to F2D class
#     '''
#     l_eval = [tf.squeeze(tf.transpose(F.expr(X[:, 0], X[:, 1])), axis=-1)]

#     for i, t_order in enumerate(F.reduced_tab_diff):
#         l_eval.append(tf.expand_dims(
#             F.reduced_tab_diff[i](X[:, 0], X[:, 1]), axis=-1))

#     return l_eval


################### Set A here ###################
A = 0
dA_dxx = 0
dA_dyy = 0


################### ###################
def try_config(config, id_add):
    '''
    A run of the full algorithm described in the paper of 1997. 
    Variables:
    -config (dict): The hyperparameters used to construct the model and set the training loop are in config.
    -id_add (int): add id_add to the trial_id to avoid overwriting previous trials 
    '''
    print('config:\n', config)
    config_model = config['config_model']
    config_training = config['config_training']

    l_units = config_model['l_units']
    noise = config_model['noise']
    learning_rate = config_model['learning_rate']
    if config_model['optimizer'] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print('optimizer is not known !')

    learning_rate = config_model['learning_rate']

    # generate model
    model = generate_model(l_units, noise=noise)

    # Universal Approximator
    # @tf.function  # TODO: learn to use it to accelerate the computations

    def g_3(X, training=True):
        # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
        N_X = model(X, training=training)
        return tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)*N_X

    # Custom loss function to approximate the derivatives

    def custom_loss_3(tf_sample_coords):
        dN_dx, dN_dxx, dN_dy, dN_dyy = differentiate(
            model, tf_sample_coords)
        f_r = tf.reshape(f(tf_sample_coords), [batch_size, 1])

        F, dF_dx, dF_dxx, dF_dy, dF_dyy = evaluate_F_and_diff(tf_sample_coords)

        dg_dxx = dF_dxx + 2*dF_dx*dN_dx + F*dN_dxx + dA_dxx
        dg_dyy = dF_dyy + 2*dF_dy*dN_dy + F*dN_dyy + dA_dyy
        res = residual(dg_dxx, dg_dyy, f_r)

        loss = tf.reduce_mean(tf.square(res))
        return loss

    # train of method 3:

    def train_step_3(tf_sample_coords):
        with tf.GradientTape() as tape:
            loss = custom_loss_3(tf_sample_coords)
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    mae_metric = tf.keras.metrics.MeanAbsoluteError(
        name="mean_absolute_error", dtype=None)

    def validate(validation_coords):
        _, dg_dxx, _, dg_dyy = differentiate(g_3, validation_coords)
        f_r = tf.reshape(f(validation_coords), [
                         tf.shape(validation_coords)[0], 1])
        res = residual(dg_dxx, dg_dyy, f_r)
        val_mae = mae_metric(res, tf.zeros(tf.shape(res))).numpy()
        return val_mae

    # Training the Model of method 3:

    trial_id = config['trial_id']+id_add
    epochs_max = config_training['epochs_max']
    n_trains = config_training['n_trains']
    batch_size = config_training['batch_size']
    display_step = config_training['display_step']
    tol = config_training['tol']
    patience = config_training['patience']

    # !!! to change according to the way the folders are arranged
    folder_dir = f'differentiate/hypertuning/byHand/trial_{trial_id}/'

    # TODO: learn how to use Yaml instead... (piece of advice from Jules S.)

    if not(os.path.exists(folder_dir)):
        os.mkdir(folder_dir)
    with open(folder_dir+f'config_{trial_id}.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    history = {'mean_loss': [], 'val_mae': []}

    epoch = 0
    val_mae = np.infty
    val_mae_reached = (val_mae <= tol)
    EarlyStopped = False

    # tf.keras.backend.set_learning_phase(1) # 'globally' activate training mode, slightly too strong maybe : check training mode for GaussianNoise layer
    while not(EarlyStopped) and not(val_mae_reached) and epoch < epochs_max:
        epoch += 1
        time_start = time()
        print('epoch:', epoch, end=' ')
        losses = []

        indices = np.random.randint(tf_coords.shape[0], size=batch_size)
        tf_sample_coords = tf.convert_to_tensor(
            [tf_coords[i] for i in indices])
        for k in range(n_trains):
            if k % display_step == display_step-1:
                print('.', end='')
            losses.append(train_step_3(tf_sample_coords))
        loss = np.mean(losses)

        # create validation_coords
        indices = np.random.randint(tf_coords.shape[0], size=100)
        tf_val_coords = tf.convert_to_tensor([tf_coords[i] for i in indices])
        tf_val_coords = tf_val_coords + \
            tf.random.normal(shape=tf.shape(
                tf_val_coords).numpy(), mean=0, stddev=1e-3)
        val_mae = validate(tf_val_coords)

        print("mean_loss:", loss, end=' ')
        print('val_mae:', val_mae, end=' ')
        history['mean_loss'].append(loss)
        history['val_mae'].append(val_mae)

        # time_end_training = time()
        # print('duration training :', time_end_training-time_start, end=' ')

        val_mae_reached = (val_mae <= tol)

        if val_mae_reached:
            print(f'\n tolerance set is reached : {val_mae}<={tol}')

        model.save(
            folder_dir+f'model_poisson_trial_{trial_id}_epoch_{epoch}_val_mae_{val_mae:6f}.h5')

        if (len(history['val_mae']) >= patience+1) and np.argmin(history['val_mae'][-(patience+1):]) == 0:
            print('\n EarlyStopping activated', end=' ')
            EarlyStopped = True

        elif (len(history['val_mae']) >= patience+1):
            # clean the savings folder
            r_val_mae_epoch = epoch-patience
            r_val_mae = history['val_mae'][-(patience+1)]
            file_path = folder_dir + \
                f'model_poisson_trial_{trial_id}_epoch_{r_val_mae_epoch}_val_mae_{r_val_mae:6f}.h5'

            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(file_path)
                print("The system cannot find the file specified")

        time_end_epoch = time()
        duration_epoch = time_end_epoch-time_start
        print('duration epoch:', duration_epoch)
        print()

    # not optimized
    min_val_mae = np.min(history['val_mae'])
    min_val_mae_epoch = np.argmin(history['val_mae'])+1

    model = keras.models.load_model(
        folder_dir+f'model_poisson_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5')
    os.rename(folder_dir+f'model_poisson_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5',
              folder_dir+f'best_model_poisson_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5')
    print('best model loaded and renamed')

    # tf.keras.backend.set_learning_phase(0) # 'globally' disable training mode
    print("val_mae>tol:", val_mae > tol)

    plt.plot(np.arange(0, epoch), history['mean_loss'], label='mean_loss')
    plt.plot(np.arange(0, epoch), history['val_mae'], label='val_mae')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(
        f'epochs_max = {epochs_max},n_trains={n_trains},batch_size={batch_size}')
    plt.legend()
    plt.savefig(folder_dir+f"history_trial_{trial_id}.png", transparent=False)
    print(folder_dir+f"history_trial_{trial_id}.png")

    plt.plot(list(range(0, len(history['val_mae']))), history['val_mae'])
    plt.show()

    print(np.min(history['val_mae']))


# Does not learn with F = F2D(..., 'sinxpy_real') or 'xpy_real'
# - take a look at dF_dx and so on
# - ... at the hyperparameters
# -


# # Questions

# Quelle architecture ?
#
# Comment éviter l'overfitting ?
#
# Choix de l'optimizer + regularizer ? + Implémentation ?
#
# Implémentation de système d'EDP à plusieurs inconnues (étant des fonctions bien sûr) ? (Est-ce que c'est utile ça ? Par curiosité)
#
# Plus rapide ? Comment enlever les boucles `for` ? => mini_batch_gradient_descent ? done
#
# Besoin de batch_normalization ? + autres hyperparamètres ?
#
# bias à la dernière couche : nécessaier ou pas ?

# # Idées

# Ajout de bruit en entrée contre l'overfitting
#
# Une sortie par inconnue


############### ###################
def generate_random_config(trial_id, grid_length):
    n_hidden_layers = np.random.randint(2, 7)
    l_units = [5*np.random.randint(1, 7) for _ in range(n_hidden_layers)]
    noise = np.random.randint(2)

    config_model = {
        'l_units': l_units,
        'noise': noise,
        'learning_rate': 1e-2,
        'optimizer': "Adam"
    }

    n_trains = 50*np.random.randint(2, 5)
    config_training = {
        "epochs_max": 5000,
        "n_trains": n_trains,
        "batch_size": 8192,
        "display_step": 10,
        "tol": 1e-6,
        "patience": 50
    }

    config = {
        "trial_id": trial_id,
        "grid_length": grid_length,  # do not change anything, here to inform
        "config_training": config_training,
        "config_model": config_model
    }

    return config


def randomTuning(max_trials, id_add):
    for trial_id in range(max_trials):
        config = generate_random_config(trial_id, grid_length)
        try_config(config, id_add=id_add)


if __name__ == '__main__':
    run_hyper_tuning = False
    run_spec_config = True

    if run_hyper_tuning:
        max_trials = 100
        id_add = 200
        randomTuning(max_trials, id_add)
    elif run_spec_config:
        config_model = {
            'l_units': [30, 30],
            'noise': 0,
            'learning_rate': 1e-4,
            'optimizer': "Adam"
        }

        config_training = {
            "epochs_max": 5000,
            "n_trains": 10,
            "batch_size": 4096,
            "display_step": 1,
            "tol": 1e-6,
            "patience": 10
        }

        config = {
            "trial_id": 0,
            "grid_length": grid_length,
            'remark': 'just to try a run',
            "config_training": config_training,
            "config_model": config_model
        }

        try_config(config, id_add=1000)
