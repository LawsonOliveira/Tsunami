import pandas as pd
from sympy import Matrix
import os
from random import sample
import tensorflow as tf
import keras_tuner as kt
import keras
import sympy as sm
import numpy as np
import json

from tensorflow.python.client import device_lib
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(physical_devices))
print(device_lib.list_local_devices())
# what if empty...
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # On windows systems you cannont install NCCL that is required for multi GPU
# # So we need to follow hierarchical copy method or reduce to single GPU (less efficient than the former)
# strategy = tf.distribute.MirroredStrategy(
#     devices=['GPU:0'], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

DTYPE = 'float32'

tf.keras.backend.set_floatx(DTYPE)

__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\ncwd:\n', os.getcwd())
os.chdir(__file__)
print('changed to:\n', os.getcwd(), '\n')

# where we'll put the results and save the models
directory = "differentiate/my_dir"
project_name = "tune_hypermodel"
path_to_trials = __file__+'/'+directory+'/' + project_name

#############################################################################
####################### config of the tuner #################################

overwrite_checkpoint = True  # overwrite_model_saved

overwrite = True  # true = destroy previous results, false = resume search

n_epoch_max = 100


#############################################################################
# Set F here

x, y = sm.symbols('x,y')


def expr_dummy_F():
    return x*(1-x)*y*(1-y)


expr_F = expr_dummy_F()
dexpr_F_dx = sm.diff(expr_F, x, 1)
dexpr_F_dxx = sm.diff(dexpr_F_dx, x, 1)
dexpr_F_dy = sm.diff(expr_F, y, 1)
dexpr_F_dyy = sm.diff(dexpr_F_dy, y, 1)

# print(dexpr_F_dx)
# print(dexpr_F_dxx)

# You can forget a no lambdified expression => here we greatly avoid 'for' loops

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

#############################################################################
# Set A here


A = 0
dA_dxx = 0
dA_dyy = 0
#############################################################################
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


grid_length = 100


X = np.linspace(0, 1, grid_length, endpoint=True)
Y = np.linspace(0, 1, grid_length, endpoint=True)
tf_coords = tf.convert_to_tensor(
    [tf.constant([x, y], dtype=DTYPE) for x in X for y in Y])
tf_boundary_coords = tf.convert_to_tensor([tf.constant([x, y], dtype=DTYPE) for x in [
                                          0, 1] for y in Y] + [tf.constant([x, y], dtype=DTYPE) for y in [0, 1] for x in X])


#############################################################################
class MyTuner(kt.Hyperband):
    def __init__(self, hypermodel=None, objective=None, max_trials=10, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, **kwargs):
        super().__init__(hypermodel, objective, max_trials, seed,
                         hyperparameters, tune_new_entries, allow_new_entries, **kwargs)
        self.l_names_hp = []

    def build(self, hp, dim=2):
        model = keras.models.Sequential([
            keras.layers.Input(shape=(dim))
        ])

        if hp.Boolean('noise_enabled'):
            model.add(keras.layers.GaussianNoise(stddev=hp.Float(
                "stddev", min_value=1e-4, max_value=1e-2, sampling="log")))

        for i in range(hp.Int("num_layers", 2, 10)):
            model.add(
                keras.layers.Dense(units=hp.Int(f"units_{i}", min_value=5, max_value=30, step=5),
                                   activation='elu', kernel_initializer='he_normal'
                                   )
            )
        model.add(keras.layers.Dense(1, use_bias=False))

        optimizer = tf.optimizers.Adam(
            learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log'))

        model.compile(
            optimizer=optimizer, loss="mse", metrics=["mae"],
        )

        self.l_names_hp += ['noise_enabled', 'num_layers', 'lr']
        return model

    def fit(self, hp, model, tf_coords, tf_boundary_coords, validation_coords=[], patience=10):
        def save_history(history):
            # print('\nhistory:\n', history, '\n')
            last_trial_folder = [f for f in os.listdir(
                path_to_trials) if f[:5] == 'trial'][-1]
            path_to_last_trial = path_to_trials+'/'+last_trial_folder
            with open(path_to_last_trial+'/history.json', 'w') as fp:
                json.dump(history, fp, indent=2)

        def save_model(model):
            # checkpoint to save trained model
            last_trial_folder = [f for f in os.listdir(
                path_to_trials) if f[:5] == 'trial'][-1]
            path_to_last_trial = path_to_trials+'/'+last_trial_folder
            if not('checkpoint.h5' in os.listdir(path_to_last_trial)):
                model.save(path_to_trials+'/' +
                           last_trial_folder+'/checkpoint.h5')

        # @tf.function
        def train_step(model, tf_sample_coords, tf_boundary_coords, batch_size):
            def g(X):
                # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
                N_X = model(X)
                return tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)*N_X

            def custom_loss():
                dN_dx, dN_dxx, dN_dy, dN_dyy = differentiate(
                    g, tf_sample_coords)
                f_r = tf.reshape(f(tf_sample_coords), [batch_size, 1])

                F, dF_dx, dF_dxx, dF_dy, dF_dyy = evaluate_F_and_diff(
                    tf_sample_coords)

                dg_dxx = dF_dxx + 2*dF_dx*dN_dx + F*dN_dxx + dA_dxx
                dg_dyy = dF_dyy + 2*dF_dy*dN_dy + F*dN_dyy + dA_dyy
                res = residual(dg_dxx, dg_dyy, f_r)

                loss = tf.reduce_mean(tf.square(res))
                return loss, res

            with tf.GradientTape() as tape:
                # this custom_loss function shall be added to keras.losses and use the loss defined in compile
                loss, res = custom_loss()

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            # Update metrics (includes the metric that tracks the loss)
            model.compiled_metrics.update_state(res, tf.zeros(tf.shape(res)))
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in model.metrics}, loss

        # @tf.function
        def validate(model, validation_coords):
            def g_3(x):
                N_x = model(x, training=False)
                return N_x

            _, dg_dxx, _, dg_dyy = differentiate(g_3, validation_coords)
            f_r = tf.reshape(f(validation_coords), [
                tf.shape(validation_coords)[0], 1])
            res = residual(dg_dxx, dg_dyy, f_r)
            model.compiled_metrics.update_state(res, tf.zeros(tf.shape(res)))
            return {m.name: m.result() for m in model.metrics}

        history = {'train_loss': [],
                   'val_mae': []}
        n_epoch = n_epoch_max
        for epoch in range(1, n_epoch+1):
            EarlyStopped = False

            print(f'Epoch: {epoch}/{n_epoch}', end=' ')
            train_losses, val_losses, val_maes = [], [], []
            batch_size = hp.Choice(
                'batch_size', [2**i for i in range(4, 13)])
            indices = np.random.randint(
                tf_coords.shape[0], size=batch_size)
            tf_sample_coords = tf.convert_to_tensor(
                [tf_coords[i] for i in indices])
            for _ in range(hp.Int('n_train', 10, 500)):
                metrics, train_loss = train_step(
                    model, tf_sample_coords, tf_boundary_coords, batch_size)
                train_losses.append(train_loss)
            mean_train_loss = np.mean(train_losses)
            history['train_loss'].append(
                float(mean_train_loss))
            # .3f pour 3 chiffres après la virgule
            print(f'train-loss: {mean_train_loss:.8f}', end=' ')

            if tf.shape(validation_coords).numpy()[0]:
                val_metrics = validate(model, validation_coords)
                print(val_metrics)
                val_mae = val_metrics['mae']
                history['val_mae'].append(
                    float(val_mae))
                print(
                    f'val_mae: {val_mae:.3f}')
            else:
                # create validation set of size batch_size from tf_coords modified
                # random => no chance to be the same as the training set
                # remark: points of validation set could exceed the domain of definition
                indices = np.random.randint(
                    tf_coords.shape[0], size=batch_size)
                tf_val_coords = tf.convert_to_tensor(
                    [tf_coords[i] for i in indices])
                tf_val_coords = tf_val_coords + tf.random.normal(shape=tf.shape(
                    tf_val_coords).numpy(), mean=0, stddev=1)
                val_metrics = validate(model, tf_val_coords)
                val_mae = val_metrics['mae']
                history['val_mae'].append(
                    float(val_mae))
                print(
                    f'val_mae: {val_mae:.8f}')

            self.l_names_hp += ['batch_size', 'n_train']

            save_history(history)
            # EarlyStopping implemented :
            if (len(history['val_mae']) > (patience+1)) and np.argmin(history['val_mae'][-(patience+1):]) == 0:
                EarlyStopped = True
                print('## early stopped ## ')
                if overwrite_checkpoint:
                    save_model(model)
                break

        if not(EarlyStopped) and overwrite_checkpoint:
            save_model(model)
        return history

    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        model = self.build(hp)
        history = self.fit(hp, model, tf_coords, tf_boundary_coords,
                           validation_coords=[], patience=10)

        # Return the objective value to minimize.
        return np.min(history["val_mae"])


if __name__ == '__main__':
    tuner = MyTuner(
        # No hypermodel or objective specified.
        factor=3,
        hyperband_iterations=1,
        overwrite=overwrite,
        directory=directory,
        project_name=project_name,
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search()
    # print(tuner.get_best_hyperparameters()[0].get('units_4'))
    # print(tuner.get_best_hyperparameters()[0].get('noise_enabled'))
    # print(tuner.get_best_hyperparameters()[0])

    # best_model = tuner.get_best_models(num_models=1)

    def print_best_hyperparameters(tuner, l_names_hp):
        print('\n         best hyperparameters :')
        for name_hp in l_names_hp:
            print(name_hp+':', tuner.get_best_hyperparameters()
                  [0].get(name_hp))
        for i in range(tuner.get_best_hyperparameters()
                       [0].get('num_layers')):
            print(f'units_{i} :', tuner.get_best_hyperparameters()
                  [0].get(f'units_{i}'))

    print_best_hyperparameters(tuner, tuner.l_names_hp)
