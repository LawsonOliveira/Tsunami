# Import TensorFlow and NumPy
import sympy as sm
from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)

# # Define initial condition
# def fun_u_0(x):
#     return -tf.sin(pi * x)

# # Define boundary condition
# def fun_u_b(t, x):
#     n = x.shape[0]
#     return tf.zeros((n, 1), dtype=DTYPE)

# Define right member function


def f(X):
    return tf.sin(np.pi*X[:, 0])*tf.sin(np.pi*X[:, 1])


# Define residual of the PDE


def residual(X, du_dxx, du_dyy):
    return f(X)+du_dxx+du_dyy


#############################
# Set number of data points
N_r = 10000

# Set boundary
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.

# Lower bounds
lb = tf.constant([xmin, ymin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, ymax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)


# Draw uniformly sampled collocation points
x_r = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype=DTYPE)
y_r = tf.random.uniform((N_r, 1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([x_r, y_r], axis=1)


#############################

fig = plt.figure(figsize=(9, 6))
plt.scatter(x_r, y_r, c='r', marker='.', alpha=0.1)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.title('Positions of collocation points and boundary data')
# plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)


#############################
def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
        lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation=tf.keras.activations.get(
                                            'tanh'),
                                        kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1, use_bias=False))

    return model


def generate_model(l_units):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
        lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Append hidden layers
    n_hidden = len(l_units)
    for i_hid in range(n_hidden):
        model.add(tf.keras.layers.Dense(l_units[i_hid],
                                        activation=tf.keras.activations.get(
                                            'tanh'),
                                        kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))

    return model


################### Set A here ###################
dA_dxx = 0
dA_dyy = 0

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

expr_F = sm.lambdify([x, y], sm.Matrix([expr_F]), 'numpy')
dexpr_F_dx = sm.lambdify([x, y], sm.Matrix([dexpr_F_dx]), 'numpy')
dexpr_F_dxx = sm.lambdify([x, y], sm.Matrix([dexpr_F_dxx]), 'numpy')
dexpr_F_dy = sm.lambdify([x, y], sm.Matrix([dexpr_F_dy]), 'numpy')
dexpr_F_dyy = sm.lambdify([x, y], sm.Matrix([dexpr_F_dyy]), 'numpy')


def evaluate_F_and_diff(X):
    F = tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)
    dF_dx = tf.expand_dims(dexpr_F_dx(X[:, 0], X[:, 1]), axis=-1)
    dF_dxx = tf.expand_dims(dexpr_F_dxx(X[:, 0], X[:, 1]), axis=-1)
    dF_dy = tf.expand_dims(dexpr_F_dy(X[:, 0], X[:, 1]), axis=-1)
    dF_dyy = tf.expand_dims(dexpr_F_dyy(X[:, 0], X[:, 1]), axis=-1)

    return F, dF_dx, dF_dxx, dF_dy, dF_dyy

#############################


@tf.function
def get_residual(model, X_r):

    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        x, y = X_r[:, 0:1], X_r[:, 1:2]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(x)
        tape.watch(y)

        # Determine residual
        u = model(tf.stack([x[:, 0], y[:, 0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        du_dx = tape.gradient(u, x)
        du_dy = tape.gradient(u, y)

    du_dxx = tape.gradient(du_dx, x)
    du_dyy = tape.gradient(du_dy, y)

    del tape

    # F, dF_dx, dF_dxx, dF_dy, dF_dyy = evaluate_F_and_diff(X_r)

    # dg_dxx = dF_dxx + 2*dF_dx*du_dx + F*du_dxx + dA_dxx
    # dg_dyy = dF_dyy + 2*dF_dy*du_dy + F*du_dyy + dA_dyy

    dg_dxx = du_dxx
    dg_dyy = du_dyy

    return residual(X_r, dg_dxx, dg_dyy)


#############################


def compute_loss(model, X_r):
    res = get_residual(model, X_r)
    loss = tf.reduce_mean(tf.square(res))
    return loss


#############################
def get_grad(model, X_r):

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, X_r)

    grad_theta = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, grad_theta


#############################
# Initialize model aka u_\theta
# model = generate_model([20, 20, 20])
model = init_model()

# We choose a piecewise decay of the learning rate, i.e., the
# step size in the gradient descent type algorithm
# the first 1000 steps use a learning rate of 0.01
# from 1000 - 3000: learning rate = 0.001
# from 3000 onwards: learning rate = 0.0005

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [1500, 3000], [1e-2, 5e-3, 5e-4])

# Choose the optimizer
opt = tf.keras.optimizers.Adam(learning_rate=lr)


#############################

# Define one training step as a TensorFlow function to increase speed of training

# @tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(model, X_r)

    # Perform gradient descent step
    opt.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss


# Number of training epochs
N = 5000
hist = []

# Start timer
t0 = time()

for i in range(N+1):

    loss = train_step()

    # Append current loss to hist
    hist.append(loss.numpy())

    # Output current loss after 50 iterates
    if i % 50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss))

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

#############################
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$')
