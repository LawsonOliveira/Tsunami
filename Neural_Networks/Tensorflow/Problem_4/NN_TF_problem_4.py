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
lb = 0
ub = 3
################### proper to the computer used ###################
__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\ncwd:', os.getcwd())
os.chdir(__file__)
print('changed to:', os.getcwd(), '\n')


################### Create the model ###################

def generate_model(l_units, input_shape=dim, noise=False, dropout_rate=0):
    n_hidden = len(l_units)
    assert n_hidden
    input_ = keras.layers.Input(shape=input_shape)
    dropout = keras.layers.Dropout(dropout_rate)(input_)

    if noise:
        gaussian_layer = keras.layers.GaussianNoise(stddev=1e-4)(dropout)
        hidden_v1 = keras.layers.Dense(
            l_units[0], activation='relu', kernel_initializer='he_normal')(gaussian_layer)
        hidden_v2 = keras.layers.Dense(
            l_units[0], activation='relu', kernel_initializer='he_normal')(gaussian_layer)
    else:
        hidden_v1 = keras.layers.Dense(
            l_units[0], activation='relu', kernel_initializer='he_normal')(dropout)
        hidden_v2 = keras.layers.Dense(
            l_units[0], activation='relu', kernel_initializer='he_normal')(dropout)
    for i in range(1, n_hidden):
        hidden_v1 = keras.layers.Dense(
            l_units[i], activation='relu', kernel_initializer='he_normal')(hidden_v1)
        hidden_v2 = keras.layers.Dense(
            l_units[i], activation='relu', kernel_initializer='he_normal')(hidden_v2)
    output_v1 = keras.layers.Dense(1)(hidden_v1)
    output_v2 = keras.layers.Dense(1)(hidden_v2)
    output = keras.layers.Concatenate()([output_v1, output_v2])
    model = keras.Model(inputs=[input_], outputs=[output])
    print('model generated with API Functional')
    model.summary()
    return model


################### define the EDP to solve ###################
# Given EDO

@tf.function
def f_1(x):
    return tf.cos(x)-(1+x**2+tf.sin(x)**2)


@tf.function
def f_2(x):
    return 2*x-(1+x**2)*tf.sin(x)


def residual(x, psi_v1, dpsi_dx_v1, psi_v2, dpsi_dx_v2, f_ev1, f_ev2):
    res1 = dpsi_dx_v1-psi_v1**2+psi_v2-f_ev1
    res2 = dpsi_dx_v2-psi_v1*psi_v2-f_ev2
    return res1+res2


@tf.function
def differentiate(model, X, training=False):
    with tf.GradientTape() as tape:
        x1 = X[:, 0:1]
        tape.watch(x1)
        u = model(tf.stack([x1[:, 0]], axis=1), training=training)
    du_dx = tf.squeeze(tape.batch_jacobian(u, x1))
    del tape
    du_dx = tf.reshape(du_dx, [X.shape[0], u.shape[-1]])
    return u, du_dx


#######################################################################################################
################### Method 3: g automatically respects the boundary conditions ########################
# article : 1997_Artificial_neural_networks_for_solving_ordinary_and_partial_differential_equations.pdf


################### Set F for psi_v1 here ###################
x = sm.symbols('x')


expr_F_v1 = x
dexpr_F_dx_v1 = sm.diff(expr_F_v1, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_F_v1 = sm.lambdify([x], Matrix([expr_F_v1]), 'numpy')
dexpr_F_dx_v1 = sm.lambdify([x], Matrix([dexpr_F_dx_v1]), 'numpy')


def evaluate_F_and_diff_v1(X):
    F = tf.expand_dims(tf.squeeze(expr_F_v1(X[:, 0])), axis=-1)
    F = tf.cast(F, dtype=DTYPE)
    dF_dx = tf.expand_dims(tf.squeeze(dexpr_F_dx_v1(X[:, 0])), axis=-1)
    dF_dx = tf.cast(dF_dx, dtype=DTYPE)
    return F, dF_dx

################### Set F for psi_v2 here ###################


expr_F_v2 = x
dexpr_F_dx_v2 = sm.diff(expr_F_v2, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_F_v2 = sm.lambdify([x], Matrix([expr_F_v2]), 'numpy')
dexpr_F_dx_v2 = sm.lambdify([x], Matrix([dexpr_F_dx_v2]), 'numpy')


def evaluate_F_and_diff_v2(X):
    F = tf.expand_dims(tf.squeeze(expr_F_v2(X[:, 0])), axis=-1)
    F = tf.cast(F, dtype=DTYPE)
    dF_dx = tf.expand_dims(tf.squeeze(dexpr_F_dx_v2(X[:, 0])), axis=-1)
    dF_dx = tf.cast(dF_dx, dtype=DTYPE)
    return F, dF_dx


################### Set A for psi_v1 here ###################
def expr_A_v1():
    return 0


expr_A_v1 = 0
dexpr_A_dx_v1 = sm.diff(expr_A_v1, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_A_v1 = sm.lambdify([x], Matrix([expr_A_v1]), 'numpy')
dexpr_A_dx_v1 = sm.lambdify([x], Matrix([dexpr_A_dx_v1]), 'numpy')


def evaluate_A_and_diff_v1(X):
    A = tf.expand_dims(tf.squeeze(expr_A_v1(X[:, 0])), axis=-1)
    A = tf.cast(A, dtype=DTYPE)
    dA_dx = tf.expand_dims(tf.squeeze(dexpr_A_dx_v1(X[:, 0])), axis=-1)
    dA_dx = tf.cast(dA_dx, dtype=DTYPE)
    return A, dA_dx

################### Set A for psi_v2 here ###################


def expr_A_v2():
    return 1


expr_A_v2 = 1
dexpr_A_dx_v2 = sm.diff(expr_A_v2, x, 1)

# remark: You can forget a no lambdified expression => here we greatly avoid 'for' loops

expr_A_v2 = sm.lambdify([x], Matrix([expr_A_v2]), 'numpy')
dexpr_A_dx_v2 = sm.lambdify([x], Matrix([dexpr_A_dx_v2]), 'numpy')


def evaluate_A_and_diff_v2(X):
    A = tf.expand_dims(tf.squeeze(expr_A_v2(X[:, 0])), axis=-1)
    A = tf.cast(A, dtype=DTYPE)
    dA_dx = tf.expand_dims(tf.squeeze(dexpr_A_dx_v2(X[:, 0])), axis=-1)
    dA_dx = tf.cast(dA_dx, dtype=DTYPE)
    return A, dA_dx

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
    problem_id = config['problem_id']
    if use_wandb:
        wandb.init(
            project="TF-tsunami",
            name=f'prob_{problem_id}_trial_{trial_id}',
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

    # Custom loss function to approximate the derivatives

    def raw_residual(tf_sample_coords):
        N, dN_dx = differentiate(
            model, tf_sample_coords, training=True)

        N_v1 = N[:, 0]
        dN_dx_v1 = dN_dx[:, 0]

        F_v1, dF_dx_v1 = evaluate_F_and_diff_v1(tf_sample_coords)
        A_v1, dA_dx_v1 = evaluate_A_and_diff_v1(tf_sample_coords)

        N_v2 = N[:, 1]
        dN_dx_v2 = dN_dx[:, 1]

        F_v2, dF_dx_v2 = evaluate_F_and_diff_v2(tf_sample_coords)
        A_v2, dA_dx_v2 = evaluate_A_and_diff_v2(tf_sample_coords)

        psi_ev1 = A_v1+F_v1*N_v1
        dpsi_dx_v1 = dA_dx_v1+dF_dx_v1*N_v1 + F_v1*dN_dx_v1

        psi_ev2 = A_v2+F_v2*N_v2
        dpsi_dx_v2 = dA_dx_v2+dF_dx_v2*N_v2 + F_v2*dN_dx_v2

        f_ev1 = tf.reshape(f_1(tf_sample_coords), [
            tf_sample_coords.shape[0], 1])
        f_ev2 = tf.reshape(f_2(tf_sample_coords), [
            tf_sample_coords.shape[0], 1])

        res = residual(tf_sample_coords, psi_ev1, dpsi_dx_v1,
                       psi_ev2, dpsi_dx_v2, f_ev1, f_ev2)

        return res

    def custom_loss(tf_sample_coords):
        N, dN_dx = differentiate(
            model, tf_sample_coords, training=True)

        N_v1 = N[:, 0]
        dN_dx_v1 = dN_dx[:, 0]

        F_v1, dF_dx_v1 = evaluate_F_and_diff_v1(tf_sample_coords)
        A_v1, dA_dx_v1 = evaluate_A_and_diff_v1(tf_sample_coords)

        N_v2 = N[:, 1]
        dN_dx_v2 = dN_dx[:, 1]

        F_v2, dF_dx_v2 = evaluate_F_and_diff_v2(tf_sample_coords)
        A_v2, dA_dx_v2 = evaluate_A_and_diff_v2(tf_sample_coords)

        psi_ev1 = A_v1+F_v1*N_v1
        dpsi_dx_v1 = dA_dx_v1+dF_dx_v1*N_v1 + F_v1*dN_dx_v1

        psi_ev2 = A_v2+F_v2*N_v2
        dpsi_dx_v2 = dA_dx_v2+dF_dx_v2*N_v2 + F_v2*dN_dx_v2

        f_ev1 = tf.reshape(f_1(tf_sample_coords), [
            tf_sample_coords.shape[0], dim])
        f_ev2 = tf.reshape(f_2(tf_sample_coords), [
            tf_sample_coords.shape[0], dim])

        res = residual(tf_sample_coords, psi_ev1, dpsi_dx_v1,
                       psi_ev2, dpsi_dx_v2, f_ev1, f_ev2)

        return tf.reduce_mean(tf.square(res))

    # train:

    def train_step(tf_sample_coords):
        with tf.GradientTape() as tape:
            loss = custom_loss(tf_sample_coords)
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        del tape
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    mae_metric = tf.keras.metrics.MeanAbsoluteError(
        name="mean_absolute_error", dtype=None)

    def validate(validation_coords):
        res = raw_residual(validation_coords)
        val_mae = mae_metric(res, tf.zeros(tf.shape(res)))
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
        val_mae = validate(tf_val_coords).numpy()

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

    def fun_psi_v1(X, training=False):
        # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
        N_X_v1 = model(X, training=training)[:, 0]
        A_v1 = tf.expand_dims(tf.squeeze(expr_A_v1(X[:, 0])), axis=-1)
        A_v1 = tf.cast(A_v1, dtype=DTYPE)
        F_v1 = tf.expand_dims(tf.squeeze(expr_F_v1(X[:, 0])), axis=-1)
        F_v1 = tf.cast(F_v1, dtype=DTYPE)
        return A_v1+F_v1*N_X_v1

    def fun_psi_v2(X, training=False):
        # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
        N_X_v2 = model(X, training=training)[:, 1]
        A_v2 = tf.expand_dims(tf.squeeze(expr_A_v2(X[:, 0])), axis=-1)
        A_v2 = tf.cast(A_v2, dtype=DTYPE)
        F_v2 = tf.expand_dims(tf.squeeze(expr_F_v2(X[:, 0])), axis=-1)
        F_v2 = tf.cast(F_v2, dtype=DTYPE)
        return A_v2+F_v2*N_X_v2

    def true_function_v1(x):
        return tf.sin(x)

    def true_function_v2(x):
        return 1+x**2
    # to check that the model is not overfitting
    # Rk: may blur the cmapping then
    noise = (tf.random.uniform((grid_length**dim, dim))-0.5)/grid_length

    tf_noisy_coords = tf_coords+noise

    true_values_v1 = tf.reshape(true_function_v1(
        tf_noisy_coords), [grid_length for _ in range(dim)]).numpy()
    appro_values_v1 = tf.reshape(
        fun_psi_v1(tf_noisy_coords, training=False), [grid_length for _ in range(dim)]).numpy()
    true_values_v2 = tf.reshape(true_function_v2(
        tf_noisy_coords), [grid_length for _ in range(dim)]).numpy()
    appro_values_v2 = tf.reshape(
        fun_psi_v2(tf_noisy_coords, training=False), [grid_length for _ in range(dim)]).numpy()
    # change g according to the method applied
    # no @tf.function above g_3...
    error_v1 = np.abs(true_values_v1-appro_values_v1)
    error_v2 = np.abs(true_values_v2-appro_values_v2)
    print('np.max(error_v1):', np.max(error_v1))
    print('np.max(error_v2):', np.max(error_v2))
    # print(error.shape)

    fig, axs = plt.subplots(1, 2)
    axs[0, 0].plot(X, error_v1, c='r', label='absolute error')
    axs[0, 0].plot(X, true_values_v1, c='g', label='true values')
    axs[0, 0].plot(X, appro_values_v1,
                   c='b', label='approximate values')
    axs[0, 0].set_title('comparison to truth for psi 1')
    axs[0, 1].plot(X, error_v2, c='r', label='absolute error')
    axs[0, 1].plot(X, true_values_v2, c='g', label='true values')
    axs[0, 1].plot(X, appro_values_v2,
                   c='b', label='approximate values')
    axs[0, 1].set_title('comparison to truth for psi 2')

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


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
    "n_trains": 200,
    "batch_size": 8192,
    "val_size": 8192,
    "display_step": 50,
    "tol": 1e-6,
    "patience": 100
}

config = {
    "trial_id": 0,
    'problem_id': 4,
    'remark': '',
    'over_folder_dir': 'Neural_Networks/Tensorflow/Problem_4/NN_saves/',
    "config_training": config_training,
    "config_model": config_model
}


if __name__ == '__main__':
    try_config(config, id_add=0, use_wandb=False)
    # compare_truth(
    #     'Neural_Networks/Tensorflow/Problem_3/NN_saves/trial_100/best_model_trial_100_epoch_1_val_mae_0.774451.h5')
    pass
