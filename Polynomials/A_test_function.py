from cmath import tanh
from math import dist
from cv2 import sqrt
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def distance(x, x_i):
    # Returns the distance between two vectors x and x_i
    return np.linalg.norm(x - x_i)

def dist_min(x, x_list):
    #Returns the minimal distance from x to a list of points
    min_dist = np.inf
    for y in x_list:
        if y.all() != x.all():
            dist = distance(x, y)
            min_dist = min(min_dist, dist)
    return min_dist

def psi_i(x, x_i, l_square, width=1):
    dist_square = distance(x, x_i)**2
    if dist_square < l_square:
        return np.exp(-1/(width*(l_square - dist_square)))
    return 0

def A_alt(x, sample, sample_values, l_square, width):
    # A function with hyperbolic tangents
    res = 0
    l = l_square**0.5
    n=1/l
    for i in range(len(sample)):
        res += sample_values[i]*(1-tanh(n*distance(x, sample[i]))**2)
    return 0.5*res

def A(x, sample, sample_values, l_square, width):
    # Returns the value of the A function with samples from the boundary
    res = 0
    for i in range(len(sample)):
        res += sample_values[i]*psi_i(x, sample[i], l_square, width)
    return res*np.exp(1/(width*l_square))

def Z_calc(X, Y, sample, sample_values, l_square, width):
    mesh = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            mesh[i][j] = A_alt((X[i][j], Y[i][j]), sample, sample_values, l_square, width)
    return mesh

"""
Test on a circular boundary, uniform sampling, following a sinusoid boundary condition
"""
list_theta = np.linspace(0, 2*np.pi, 201, endpoint=False)
sample = np.array([np.array([np.cos(list_theta[i]), np.sin(list_theta[i])]) for i in range(len(list_theta))])
sample_values = np.sin(10*list_theta)

#Computing l:
l = min([dist_min(sample[i], sample) for i in range(len(sample))])
l_square = l**2
width = 4

"""
Trying to break the function
"""
def boundary(sample):
    results = np.zeros(len(sample))


"""
Plotting the resulting A
"""
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = Z_calc(X, Y, sample, sample_values, l_square, width)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.scatter3D(sample[:,0], sample[:,1], sample_values, cmap='Greens')
plt.show()