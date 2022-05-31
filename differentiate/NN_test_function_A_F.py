#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sympy import Matrix
import sympy as sm
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


__file__ = 'C:/Users/antie/Documents/Pole_recherche/Tsunami'


# In[2]:


__file__ = 'C:/Users/Gilles/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'


# In[3]:


import os

print(os.getcwd())
os.chdir(__file__)
print(os.getcwd())


# In[4]:


DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


# In[5]:


def generate_model(n_hidden=2, noise=False):
    # méthode API Sequential
    multilayer_perceptron = keras.models.Sequential([
        keras.layers.Input(shape=(2))
    ])
    if noise:
        multilayer_perceptron.add(keras.layers.GaussianNoise(stddev=1e-3))
    for _ in range(n_hidden):
        multilayer_perceptron.add(keras.layers.Dense(
            20, activation='elu', kernel_initializer='he_normal'))
    multilayer_perceptron.add(keras.layers.Dense(1, use_bias=False))
    multilayer_perceptron.summary()
    return multilayer_perceptron


# In[34]:


# to check :
multilayer_perceptron = generate_model()


# In[6]:


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

# print(boundary_coords.shape)


# ## Method 3: g automatically respects the boundary conditions

# article : 1997_Artificial_neural_networks_for_solving_ordinary_and_partial_differential_equations.pdf

# What is the fastest way to differentiate a sympy expr ?
#
# - see dynamic_diff.ipynb

# diff_gradually is better. Should be kept to differentiate F and A.
#
# --- done

# Other libraries than sympy? autograd,...

# ##### TODO: compare different methods according to speed of differentiation
#
# --- done in dynamic_diff.ipynb

# We need to compute :
# $\Delta g = \frac{\partial^2 g}{\partial x^2} + \frac{\partial^2 g}{\partial y^2}$
#
# Since we set $ g(X) := A(X)+F(X)\times N(x)$ with $ F_{|\delta \Omega} = 0 $ and $ A_{|\delta \Omega} = 0 $ to respect the homogeneous Dirichlet boundary condition, and $ N $ estimated by the model.
#
# We need to know :
#
# $ \frac{\partial^2 g}{\partial x^2} = \frac{\partial^2 F}{\partial x^2}N + 2\frac{\partial F}{\partial x}\frac{\partial N}{\partial x} + F\frac{\partial^2 N}{\partial x^2} + \frac{\partial^2 A}{\partial x^2} $
#
# and the same goes for the variable $ y $

# #### Dummy F

# In[8]:


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


# #### F of F_functions

# # In[58]:


# from Polynomials.F_functions import F2D
# import sympy as sm
# from sympy import Matrix

# frontier_coords = Pstud._set_coords_rectangle(1,1,10)

# l_orders = [(1,0),(2,0),(0,1),(0, 2)]
# strfn = 'sinxpy_real'
# F = F2D(frontier_coords,strfn,l_orders=l_orders)

# # prepare to infer on large matrices :
# F.expr = sm.lambdify(F.variables, Matrix([F.expr]), 'numpy')
# for t_order in l_orders:
#     F.reduced_tab_diff[F.dico_order_to_index[t_order]] = sm.lambdify(F.variables, F.reduced_tab_diff[F.dico_order_to_index[t_order]], 'numpy')


# def evaluate_F_and_diff(X):
#     '''
#     evaluate F and its differentiates get in F.reduced_tab_diff
#     Variables:
#     -X: an array or tensor tf of the coordinates

#     Returns:
#     -l_eval: list of the evaluations. To know which element corresponds to which order, use F.dico_order_to_index and increment the values of 1.

#     remark: to add to F2D class
#     '''
#     l_eval = [tf.squeeze(tf.transpose(F.expr(X[:,0],X[:,1])),axis=-1)]

#     for i,t_order in enumerate(F.reduced_tab_diff):
#         l_eval.append(tf.expand_dims(F.reduced_tab_diff[i](X[:,0],X[:,1]),axis=-1))

#     return l_eval

# remark : you can assign like this : a,b,c = [0,1,2]


# In[11]:


# Set A here

A = 0
dA_dxx = 0
dA_dyy = 0


# In[18]:


# method 1: mini-batch gradient descent + (loss = MSE + MSE on all boundary)

# a few train parameters to adjust
# use learning_rate = 1e-2, batch_size = 1000 for dummy F
learning_rate = 1e-2
training_steps = 100
batch_size = 1000
display_step = 10

optimizer = tf.optimizers.Adam(lr=learning_rate)

# generate model
multilayer_perceptron = generate_model(n_hidden=2)


# Universal Approximator
# @tf.function
def g_3(X):
    # F_x = Pstud._eval_polynome_numpy(F_xpy_real,x[0,0],x[0,1])
    N_X = multilayer_perceptron(X)
    return tf.squeeze(tf.transpose(expr_F(X[:, 0], X[:, 1])), axis=-1)*N_X


# Custom loss function to approximate the derivatives

def custom_loss_3():
    indices = np.random.randint(tf_coords.shape[0], size=batch_size)
    tf_sample_coords = tf.convert_to_tensor([tf_coords[i] for i in indices])

    dN_dx, dN_dxx, dN_dy, dN_dyy = differentiate(
        multilayer_perceptron, tf_sample_coords)
    f_r = tf.reshape(f(tf_sample_coords), [batch_size, 1])

    F, dF_dx, dF_dxx, dF_dy, dF_dyy = evaluate_F_and_diff(tf_sample_coords)

    dg_dxx = dF_dxx + 2*dF_dx*dN_dx + F*dN_dxx + dA_dxx
    dg_dyy = dF_dyy + 2*dF_dy*dN_dy + F*dN_dyy + dA_dyy
    res = residual(dg_dxx, dg_dyy, f_r)

    loss = tf.reduce_mean(tf.square(res))
    return loss

# train of method 1:


def train_step_3():
    with tf.GradientTape() as tape:
        loss = custom_loss_3()
    trainable_variables = multilayer_perceptron.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


# In[13]:


# Training the Model of method 3:

all_losses = []

for i in range(training_steps):
    print('epoch:', i)
    loss = train_step_3()
    if i % display_step == display_step-1:
        print("loss:", loss)
    all_losses.append(loss)

plt.plot(np.arange(0, training_steps), all_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# Does not learn with F = F2D(..., 'sinxpy_real') or 'xpy_real'
# - take a look at dF_dx and so on
# - ... at the hyperparameters
# -

# ## Compare to the true solution

# Common to all the methods

# In[14]:


# from matplotlib import cm
# from matplotlib.colors import ListedColormap,LinearSegmentedColormap


# In[77]:


# def true_function(X):
#     return tf.sin(np.pi*X[:,0])*tf.sin(np.pi*X[:,1])/(2*np.pi**2)

# # to check that the model is not overfitting
# # Rk: may blur the cmapping then
# noise = (tf.random.uniform((grid_length**2,2))-0.5)/grid_length

# tf_noisy_coords = tf_coords+noise
# true_values = tf.reshape(true_function(tf_noisy_coords),[100,100]).numpy()
# appro_values = tf.reshape(g_3(tf_noisy_coords),[100,100]).numpy()
# # change g according to the method applied
# # no @tf.function above g_3...
# error = np.abs(true_values-appro_values)


# print('np.max(error):',np.max(error))
# #print(error.shape)

# combined_data = [error, appro_values, true_values]
# _min,_max = np.min(combined_data), np.max(combined_data)


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,12))

# seismic = cm.get_cmap('seismic', 1024)

# plt.subplot(221)
# plt.pcolormesh(error, cmap = seismic,vmin=_min,vmax=_max)
# plt.title('Graph of the error')

# ax = axes.flat[1]
# ax.set_axis_off()

# plt.subplot(223)
# plt.pcolormesh(appro_values, cmap = seismic,vmin=_min,vmax=_max)
# plt.title('Graph of the estimated values')

# plt.subplot(224)
# im = plt.pcolormesh(true_values, cmap = seismic, vmin=_min,vmax=_max)
# plt.title('Graph of the true values')

# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

# cbar.set_ticks(np.arange(_min,_max+1e-10, 0.5e-2))


# Les valeurs sont très petites aux bords. L'écart avec les conditions aux frontières doit être négligé.

# ## Save the model

# In[1]:


# multilayer_perceptron.save('differentiate/savings/model_EDP_2D.h5')


# ## Load the model

# In[18]:


# multilayer_perceptron = keras.models.load_model('differentiate/savings/model_EDP_2D.h5')


# # Questions

# Quelle architecture ?
#
# Comment éviter l'overfitting ?
#
# Comment exploiter les avantages de l'IA ?
#
# Choix de l'optimizer + regularizer ? + Implémentation ?
#
# Implémentation de système d'EDP à plusieurs inconnues (étant des fonctions bien sûr) ? (Est-ce que c'est utile ça ? Par curiosité)
#
# Plus rapide ? Comment enlever les boucles `for` ? => mini_batch_gradient_descent ? done
#
# Besoin de batch_normalization ? + autres hyperparamètres ?

# # Idées

# Ajout de bruit en entrée contre l'overfitting
#
# Une sortie par inconnue
