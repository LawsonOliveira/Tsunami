#!/usr/bin/env python
# coding: utf-8

################### librairies ###################
#!/usr/bin/env python
# coding: utf-8

from tensorflow.python.client import device_lib
import wandb
from time import time
import json
from matplotlib import cm
from sympy import Matrix
import sympy as sm
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os


################### check tensorflow is using GPU ###################
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
################### set constants ###################
pi = tf.constant(np.pi, dtype=DTYPE)
dim = 1
lb = 0  # lower bound
ub = 2  # upper bound
################### proper to the computer used ###################
__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\ncwd:', os.getcwd())
os.chdir(__file__)
print('changed to:', os.getcwd(), '\n')


################### Create the model ###################

def generate_model(l_units, input_shape=dim, noise=False, dropout_rate=0):
    # méthode API Sequential
    n_hidden = len(l_units)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_shape))
    ])
    if noise:
        model.add(keras.layers.GaussianNoise(stddev=1e-4))
    for i in range(n_hidden):
        model.add(keras.layers.Dropout(dropout_rate))  # not so efficient
        model.add(keras.layers.Dense(
            l_units[i], activation='relu', kernel_initializer='he_normal'))  # , kernel_regularizer='l2'
    model.add(keras.layers.Dense(1))
    # use_bias=False otherwise returns error after
    # May be the cause of a stagnation in the validation ? (can circumvent by creating a layer class only adding a bias)
    model.summary()
    return model


################### define the EDP to solve ###################
# Given EDO

@tf.function
def f(x):
    return -1/5*tf.exp(-x/5)*tf.cos(x)


def residual(x, psi, dpsi_dx, dpsi_dxx, f_ev):
    return dpsi_dxx+dpsi_dx/5+psi-f_ev


@tf.function
def differentiate(model, X, training=False):
    with tf.GradientTape(persistent=True) as tape:
        x1 = X[:, 0:1]
        tape.watch(x1)
        u = model(tf.stack([x1[:, 0]], axis=1), training=training)
        du_dx = tape.gradient(u, x1)
    du_dxx = tape.gradient(du_dx, x1)
    del tape
    return u, du_dx, du_dxx


#######################################################################################################
################### Method 3: g automatically respects the boundary conditions ########################
# article : 1997_Artificial_neural_networks_for_solving_ordinary_and_partial_differential_equations.pdf

################### Set F here ###################
# Dummy F

x = sm.symbols('x')


expr_F = x**2
dexpr_F_dx = sm.diff(expr_F, x, 1)
dexpr_F_dxx = sm.diff(dexpr_F_dx, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_F = sm.lambdify([x], Matrix([expr_F]), 'numpy')
dexpr_F_dx = sm.lambdify([x], Matrix([dexpr_F_dx]), 'numpy')
dexpr_F_dxx = sm.lambdify([x], Matrix([dexpr_F_dxx]), 'numpy')


def evaluate_F_and_diff(X):
    F = tf.expand_dims(tf.squeeze(expr_F(X[:, 0])), axis=-1)
    F = tf.cast(F, dtype=DTYPE)
    dF_dx = tf.expand_dims(tf.squeeze(dexpr_F_dx(X[:, 0])), axis=-1)
    dF_dx = tf.cast(dF_dx, dtype=DTYPE)
    dF_dxx = tf.expand_dims(tf.squeeze(dexpr_F_dxx(X[:, 0])), axis=-1)
    dF_dxx = tf.cast(dF_dxx, dtype=DTYPE)
    return F, dF_dx, dF_dxx


################### Set A here ###################

expr_A = x
dexpr_A_dx = sm.diff(expr_A, x, 1)
dexpr_A_dxx = sm.diff(dexpr_A_dx, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_A = sm.lambdify([x], Matrix([expr_A]), 'numpy')
dexpr_A_dx = sm.lambdify([x], Matrix([dexpr_A_dx]), 'numpy')
dexpr_A_dxx = sm.lambdify([x], Matrix([dexpr_A_dxx]), 'numpy')


def evaluate_A_and_diff(X):
    A = tf.expand_dims(tf.squeeze(expr_A(X[:, 0])), axis=-1)
    A = tf.cast(A, dtype=DTYPE)
    dA_dx = tf.expand_dims(tf.squeeze(dexpr_A_dx(X[:, 0])), axis=-1)
    dA_dx = tf.cast(dA_dx, dtype=DTYPE)
    dA_dxx = tf.expand_dims(tf.squeeze(dexpr_A_dxx(X[:, 0])), axis=-1)
    dA_dxx = tf.cast(dA_dxx, dtype=DTYPE)
    return A, dA_dx, dA_dxx


################### Run a config of model and training ###################
def try_config(config, id_add, use_wandb=False):
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
    dropout_rate = config_model['dropout_rate']

    trial_id = config['trial_id']+id_add

    if use_wandb:
        wandb.init(
            project="TF-tsunami",
            name=f'trial_{trial_id}',
            config=config)

    # generate model
    if config_model['save_path'] == '':
        model = generate_model(l_units, noise=noise, dropout_rate=dropout_rate)
    else:
        model = keras.models.load_model(config_model['save_path'])

    lr_max = config_model['learning_rate']['lr_max']
    lr_min = config_model['learning_rate']['lr_min']
    lr_middle = config_model['learning_rate']['lr_middle']
    step_middle = config_model['learning_rate']['step_middle']
    step_min = config_model['learning_rate']['step_min']
    if config_model['optimizer'] == "Adam":
        if config_model['learning_rate']['scheduler']:
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [step_middle, step_min], [lr_max, lr_middle, lr_min])
        else:
            lr = lr_max

        # Choose the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    else:
        print('optimizer is not known !')

    # Universal Approximator
    # @tf.function  # TODO: learn to use it to accelerate the computations

    # Custom loss function to approximate the derivatives

    def custom_loss(tf_sample_coords):
        N, dN_dx, dN_dxx = differentiate(
            model, tf_sample_coords, training=True)
        F, dF_dx, dF_dxx = evaluate_F_and_diff(tf_sample_coords)
        A, dA_dx, dA_dxx = evaluate_A_and_diff(tf_sample_coords)

        psi_ev = A+F*N
        dpsi_dx = dA_dx+dF_dx*N + F*dN_dx
        dpsi_dxx = dA_dxx+dF_dxx*N+2*dF_dx*dN_dx+F*dN_dxx

        f_ev = tf.reshape(f(tf_sample_coords), [
                          tf_sample_coords.shape[0], dim])

        res = residual(tf_sample_coords, psi_ev, dpsi_dx, dpsi_dxx, f_ev)

        loss = tf.reduce_mean(tf.square(res))
        return loss

    def raw_residual(tf_sample_coords):
        N, dN_dx, dN_dxx = differentiate(
            model, tf_sample_coords, training=True)
        F, dF_dx, dF_dxx = evaluate_F_and_diff(tf_sample_coords)
        A, dA_dx, dA_dxx = evaluate_A_and_diff(tf_sample_coords)

        psi_ev = A+F*N
        dpsi_dx = dA_dx+dF_dx*N + F*dN_dx
        dpsi_dxx = dA_dxx+dF_dxx*N+2*dF_dx*dN_dx+F*dN_dxx

        f_ev = tf.reshape(f(tf_sample_coords), [
                          tf_sample_coords.shape[0], dim])

        res = residual(tf_sample_coords, psi_ev, dpsi_dx, dpsi_dxx, f_ev)
        return res

    # train:

    def train_step(tf_sample_coords):
        with tf.GradientTape() as tape:
            loss = custom_loss(tf_sample_coords)
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    mae_metric = tf.keras.metrics.MeanAbsoluteError(
        name="mean_absolute_error", dtype=None)

    def validate(validation_coords):
        res = raw_residual(validation_coords)
        val_mae = mae_metric(res, tf.zeros(tf.shape(res))).numpy()
        return val_mae

    # Training the Model of method 3:

    trial_id = config['trial_id']+id_add
    epochs_max = config_training['epochs_max']
    n_trains = config_training['n_trains']
    batch_size = config_training['batch_size']
    val_size = config_training['val_size']
    display_step = config_training['display_step']
    tol = config_training['tol']
    patience = config_training['patience']

    # !!! to change according to the way the folders are arranged
    over_folder_dir = config['over_folder_dir']
    folder_dir = over_folder_dir+f'trial_{trial_id}/'

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

    total_duration = 0
    while not(EarlyStopped) and not(val_mae_reached) and epoch < epochs_max:
        epoch += 1
        time_start = time()
        print('epoch:', epoch)
        losses = []

        # indices = np.random.randint(tf_coords.shape[0], size=batch_size)
        # tf_sample_coords = tf.convert_to_tensor([tf_coords[i] for i in indices])

        tf_sample_coords = tf.random.uniform((batch_size, dim), lb, ub)
        for k in range(n_trains):
            loss = train_step(tf_sample_coords)
            losses.append(loss)
            if k % display_step == 0:
                print('epoch:', epoch, 'train step:', k, 'loss:', loss.numpy())
                if use_wandb:
                    wandb.log({'train_loss': loss})
        mean_loss = np.mean(losses)

        if use_wandb:
            wandb.log({'mean_loss': mean_loss})

        # create validation_coords
        tf_val_coords = tf.random.uniform((val_size, dim), lb, ub)
        val_mae = validate(tf_val_coords)

        if use_wandb:
            wandb.log({'val_mae': val_mae})

        print("mean_loss:", mean_loss, end=' ')
        print('val_mae:', val_mae, end=' ')
        history['mean_loss'].append(mean_loss)
        history['val_mae'].append(val_mae)

        # time_end_training = time()
        # print('duration training :', time_end_training-time_start, end=' ')

        val_mae_reached = (val_mae <= tol)

        if val_mae_reached:
            print(f'\n tolerance set is reached : {val_mae}<={tol}')

        model.save(
            folder_dir+f'model_trial_{trial_id}_epoch_{epoch}_val_mae_{val_mae:6f}.h5')

        if (len(history['val_mae']) >= patience+1) and np.argmin(history['val_mae'][-(patience+1):]) == 0:
            print('\n EarlyStopping activated', end=' ')
            EarlyStopped = True

        elif (len(history['val_mae']) >= patience+1):
            # clean the savings folder
            r_val_mae_epoch = epoch-patience
            r_val_mae = history['val_mae'][-(patience+1)]
            file_path = folder_dir + \
                f'model_trial_{trial_id}_epoch_{r_val_mae_epoch}_val_mae_{r_val_mae:6f}.h5'

            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(file_path)
                print("The system cannot find the file specified")

        time_end_epoch = time()
        duration_epoch = time_end_epoch-time_start
        total_duration += duration_epoch
        print('duration epoch:', duration_epoch)
        print()

    if use_wandb:
        wandb.finish()

    print('end of the epochs/', end=' ')
    print('total_duration:', total_duration)

    # not optimized
    min_val_mae = np.min(history['val_mae'])
    min_val_mae_epoch = np.argmin(history['val_mae'])+1

    model = keras.models.load_model(
        folder_dir+f'model_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5')
    os.rename(folder_dir+f'model_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5',
              folder_dir+f'best_model_trial_{trial_id}_epoch_{min_val_mae_epoch}_val_mae_{min_val_mae:6f}.h5')
    print('best model loaded and renamed')

    # tf.keras.backend.set_learning_phase(0) # 'globally' disable training mode
    print('minimum validation error:', np.min(history['val_mae']))
    print("minimum val mae > tol:", min_val_mae > tol)

    plt.plot(np.arange(0, epoch), history['mean_loss'], label='mean_loss')
    plt.plot(np.arange(0, epoch), history['val_mae'], label='val_mae')
    # plt.ylabel('mean_loss')
    plt.xlabel('epoch')
    plt.title(
        f'epochs_max = {epochs_max},n_trains={n_trains},batch_size={batch_size}')
    plt.legend()
    plt.savefig(folder_dir+f"history_trial_{trial_id}.png", transparent=False)
    print('figure saved at:', folder_dir+f"history_trial_{trial_id}.png")

    print()


################### Compare to true solution ###################
def compare_truth(model_save_path):
    model = keras.models.load_model(model_save_path)

    grid_length = 100
    X = np.linspace(lb, ub, grid_length, endpoint=True)
    tf_coords = tf.convert_to_tensor(
        [tf.constant([x], dtype=DTYPE) for x in X])

    def fun_psi(X, training=True):
        # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
        N_X = model(X, training=training)
        A = tf.expand_dims(tf.squeeze(expr_A(X[:, 0])), axis=-1)
        A = tf.cast(A, dtype=DTYPE)
        F = tf.expand_dims(tf.squeeze(expr_F(X[:, 0])), axis=-1)
        F = tf.cast(F, dtype=DTYPE)
        return A+F*N_X

    def true_function(x):
        return tf.exp(-x/5)*tf.sin(x)

    # to check that the model is not overfitting
    # Rk: may blur the cmapping then
    noise = (tf.random.uniform((grid_length**dim, dim))-0.5)/grid_length

    tf_noisy_coords = tf_coords+noise

    true_values = tf.reshape(true_function(
        tf_noisy_coords), [grid_length for _ in range(dim)]).numpy()
    appro_values = tf.reshape(
        fun_psi(tf_noisy_coords, training=False), [grid_length for _ in range(dim)]).numpy()
    # change g according to the method applied
    # no @tf.function above g_3...
    error = np.abs(true_values-appro_values)
    print('np.max(error):', np.max(error))
    # print(error.shape)

    plt.plot(X, error, c='r', label='absolute error')
    plt.plot(X, true_values, c='g', label='true values')
    plt.plot(X, appro_values,
             c='b', label='approximate values')
    plt.xlabel('x')
    plt.title('comparison to truth')
    plt.legend()
    plt.show()


################### Set a config and try it ###################
config_model = {
    'l_units': [30, 30],
    'noise': 0,
    'dropout_rate': 0.2,
    'learning_rate': {'scheduler': 0, "lr_max": 1e-5, "lr_middle": 1e-7, "lr_min": 1e-9, 'step_middle': 1300, 'step_min': 3000},
    'optimizer': "Adam",
    'save_path': ''
}


config_training = {
    "epochs_max": 1,
    "n_trains": 1,
    "batch_size": 8192,
    "val_size": 8192,
    "display_step": 1,
    "tol": 1e-6,
    "patience": 100
}

config = {
    "trial_id": 0,
    'problem_id': 3,
    'remark': '',
    'over_folder_dir': 'Neural_Networks/Tensorflow/Problem_3/NN_saves/',
    "config_training": config_training,
    "config_model": config_model
}


if __name__ == '__main__':
    try_config(config, id_add=0, use_wandb=False)
    # compare_truth(
    #     'Neural_Networks/Tensorflow/Problem_3/NN_saves/trial_100/best_model_trial_100_epoch_1_val_mae_0.774451.h5')
    pass
