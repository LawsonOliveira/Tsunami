import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np


x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, (1000, 1))
x_val = np.random.rand(1000, 28, 28, 1)
y_val = np.random.randint(0, 10, (1000, 1))


class MyHyperModel(keras_tuner.HyperModel):
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

    def fit(self, hp, model, tf_coords, tf_boundary_coords, validation_coords=[], callbacks=[], patience=10, *args, **kwargs):
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
            def g_3(X):
                # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
                N_X = model(X)
                return tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)*N_X

            def custom_loss():
                _, dg_dxx, _, dg_dyy = differentiate(g_3, tf_sample_coords)
                f_r = tf.reshape(f(tf_sample_coords), [batch_size, 1])
                res = residual(dg_dxx, dg_dyy, f_r)

                alpha = hp.Float('alpha', min_value=1e-6,
                                 max_value=3, sampling='log')
                loss = tf.reduce_mean(tf.square(res)) + alpha*tf.reduce_mean(
                    tf.square(g_3(tf_boundary_coords)-boundary_conditions(tf_boundary_coords)))
                return loss, res

            def custom_loss_3():
                dN_dx, dN_dxx, dN_dy, dN_dyy = differentiate(
                    model, tf_sample_coords)
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
        n_epoch = 10
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
                    hp, model, tf_sample_coords, tf_boundary_coords, batch_size)
                train_losses.append(train_loss)
            mean_train_loss = np.mean(train_losses)
            history['train_loss'].append(mean_train_loss)
            # .3f pour 3 chiffres après la virgule
            print(f'train loss: {mean_train_loss:.8f}', end=' ')

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
                    f'val_mae: {val_mae:.8f}')

            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"val_mae": val_mae})

        if not(EarlyStopped) and not(first_run):
            save_model(model)
        return history


cb_EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae", min_delta=1e-9, patience=10)

tuner = keras_tuner.RandomSearch(
    objective=keras_tuner.Objective("val_mae", "min"),
    max_trials=2,
    hypermodel=MyHyperModel(),
    directory="results",
    project_name="custom_training",
    overwrite=True,
)
