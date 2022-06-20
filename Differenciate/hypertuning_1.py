import os
from random import sample
import tensorflow as tf
import keras_tuner as kt
import keras
import sympy as sm
import numpy as np

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

__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\n cwd:', os.getcwd())
os.chdir(__file__)
print('changed to:', os.getcwd(), '\n')

# where we'll put the results and save the models
directory = "differentiate/my_dir"
project_name = "tune_hypermodel"
path_to_trials = __file__+'/'+directory+'/' + project_name

first_run = True  # wait that it creates the folder trial_O... first and nothing else first

overwrite = True  # true = destroy previous results, false = resume search
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


class MyHyperModel(kt.HyperModel):
    def build(self, hp, dim=2):
        model = keras.models.Sequential([
            keras.layers.Input(shape=(dim))
        ])

        if hp.Boolean('noise_enabled'):
            model.add(keras.layers.GaussianNoise(stddev=hp.Float(
                "stddev", min_value=1e-4, max_value=1e-2, sampling="log")))

        for i in range(hp.Int("num_layers", 1, 10)):
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
        return model

    def fit(self, hp, model, tf_coords, tf_boundary_coords, validation_coords=[], patience=1, *args, **kwargs):
        def save_model(model):
            # checkpoint to save trained model
            last_trial_folder = [f for f in os.listdir(
                path_to_trials) if f[:5] == 'trial'][-1]
            path_to_last_trial = path_to_trials+'/'+last_trial_folder
            if not('checkpoint.h5' in os.listdir(path_to_last_trial)):
                model.save(path_to_trials+'/' +
                           last_trial_folder+'/checkpoint.h5')

        # @tf.function
        def train_step(hp, model, tf_sample_coords, tf_boundary_coords, batch_size):
            def g_1(x):
                N_x = model(x, training=True)
                return N_x

            def custom_loss():
                _, dg_dxx, _, dg_dyy = differentiate(g_1, tf_sample_coords)
                f_r = tf.reshape(f(tf_sample_coords), [batch_size, 1])
                res = residual(dg_dxx, dg_dyy, f_r)

                alpha = hp.Float('alpha', min_value=1e-6,
                                 max_value=3, sampling='log')
                loss = tf.reduce_mean(tf.square(res)) + alpha*tf.reduce_mean(
                    tf.square(g_1(tf_boundary_coords)-boundary_conditions(tf_boundary_coords)))
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
            def g_1(x):
                N_x = model(x, training=False)
                return N_x

            _, dg_dxx, _, dg_dyy = differentiate(g_1, validation_coords)
            f_r = tf.reshape(f(validation_coords), [
                tf.shape(validation_coords)[0], 1])
            res = residual(dg_dxx, dg_dyy, f_r)
            model.compiled_metrics.update_state(res, tf.zeros(tf.shape(res)))
            return {m.name: m.result() for m in model.metrics}

        history = {'train_loss': [],
                   'val_mae': []}
        n_epoch = 1000
        for epoch in range(1, n_epoch+1):
            EarlyStopped = False

            print(f'Epoch: {epoch}/{n_epoch}', end=' ')
            train_losses, val_losses, val_maes = [], [], []
            batch_size = hp.Int(
                'batch_size', 10, (tf.shape(tf_coords).numpy()[0]-1)//2)
            indices = np.random.randint(
                tf_coords.shape[0], size=batch_size)
            tf_sample_coords = tf.convert_to_tensor(
                [tf_coords[i] for i in indices])
            for _ in range(hp.Int('n_train', 10, 3000)):
                metrics, train_loss = train_step(
                    hp, model, tf_sample_coords, tf_boundary_coords, batch_size)
                train_losses.append(train_loss)
            mean_train_loss = np.mean(train_losses)
            history['train_loss'].append(mean_train_loss)
            # .3f pour 3 chiffres après la virgule
            print(f'train loss: {mean_train_loss:.3f}', end=' ')

            if tf.shape(validation_coords).numpy()[0]:
                val_metrics = validate(model, validation_coords)
                print(val_metrics)
                val_mae = val_metrics['mae']
                history['val_mae'].append(
                    val_mae)
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
                    val_mae)
                print(
                    f'val_mae: {val_mae:.3f}')

            # EarlyStopping implemented :
            if (len(history['val_mae']) > (patience+1)) and np.argmin(history['val_mae'][-(patience+1):]) == 0:
                EarlyStopped = True
                if not(first_run):
                    save_model(model)
                break

        if not(EarlyStopped) and not(first_run):
            save_model(model)
        return history


hp = kt.HyperParameters()
hypermodel = MyHyperModel()
with strategy.scope():
    model = hypermodel.build(hp)
history = hypermodel.fit(hp, model, tf_coords=tf_coords,
                         tf_boundary_coords=tf_boundary_coords)


# keep exuctions_per_trial to 10 ! to overwrite with same sample size
# patience à 30

tuner = kt.RandomSearch(
    MyHyperModel(),
    objective="val_mae",
    max_trials=500,
    executions_per_trial=10,
    overwrite=overwrite,
    directory=directory,
    project_name=project_name,
)

# tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
tuner.search(tf_coords=tf_coords,
             tf_boundary_coords=tf_boundary_coords)

# print('search_space_summary:\n', tuner.search_space_summary(), end='\n')
# Get the top 2 models.
# models = tuner.get_best_models(num_models=1)
# best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
# best_model.build(dim=2)
# best_model.summary()

# hypermodel = MyHyperModel()
# best_hp = tuner.get_best_hyperparameters()[0]
# # model = hypermodel.build(best_hp)
# model = keras.models.load_model(
#     'differentiate/my_dir/tune_hypermodel/trial_0/checkpoint.h5')
# hypermodel.fit(best_hp, model, tf_coords=tf_coords,
#                tf_boundary_coords=tf_boundary_coords)

# print(best_hp)

# tuner.results_summary()
