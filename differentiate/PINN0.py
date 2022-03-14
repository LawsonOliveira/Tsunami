# %%
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np


# %%
import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))


# %%
f0 = 1
learning_rate = 0.01
training_steps = 5000
batch_size = 100
display_step = 500


# %%
# méthode API Sequential
multilayer_perceptron = keras.models.Sequential([
    keras.layers.Input(shape=[1], name='input layer'),
    keras.layers.GaussianNoise(stddev=1e-3),
    keras.layers.Dense(10, activation='selu'),
    keras.layers.Dense(10, activation='selu'),
    keras.layers.Dense(1)
])


optimizer = tf.optimizers.SGD(learning_rate)

# %%
# Universal Approximator


def g(x):
    return x * multilayer_perceptron(x) + f0

# Given EDO


def f(x):
    return 2*x


def differentiate(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        # print(x)
        # sans la reshape a une shape () invalide pour model
        u = model(tf.reshape(x, [1]))
    du_dx = tape.gradient(u, x)
    return du_dx


# Custom loss function to approximate the derivatives
def custom_loss():
    summation = []
    for x in tf.constant(np.linspace(-1, 1, 10), dtype='float32'):  # in mesh
        dNN = differentiate(g, x)
        summation.append((dNN - f(x))**2)
    return tf.sqrt(tf.reduce_mean(tf.abs(summation)))


# %%


def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables = multilayer_perceptron.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Training the Model:


for i in range(training_steps):
    print('epoch:', i)
    train_step()
    if i % display_step == 0:
        print("loss: %f " % (custom_loss()))

# %%
# Save for API Sequential or Functional
multilayer_perceptron.save('my_keras_model_0.h5')

# %%
# load for API Sequential or Functional
multilayer_perceptron = keras.models.load_model('my_keras_model_0.h5')

# %%
# Save pour API Subclassing
multilayer_perceptron.save_weights('weights_API_Subclassing_model')

# %%
# API Subclassing
multilayer_perceptron = PINN()  # nom de la classe qui crée le modèle
multilayer_perceptron.load_weights('weights_API_Subclassing_model')

# %%
# True Solution (found analitically)


def true_solution(x):
    return x**2 + 1


X = tf.constant(np.linspace(-1, 1, 200), dtype='float32')
result = []
for x in X:
    # print(g(tf.reshape(x,[1])).numpy()[0])
    result.append(g(tf.reshape(x, [1])).numpy()[0])

S = true_solution(X)

plt.plot(X, result, label='resultat')
plt.plot(X, S, label='true result')
plt.legend()
plt.show()
