import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import random as rd

# Define some points:
theta = np.linspace(-3, 2, 40)
points = np.vstack( (np.cos(theta), np.sin(theta)) ).T

# add some noise:
points = points + 0.05*np.random.randn(*points.shape)

# P=[]
# for i in range(points.shape[0]) :
#     j=rd.randrange(points.shape[0])
#     P.append(points[j])

# points=np.array(P)

plt.scatter(points[:,0],points[:,1],c='y')

# Linear length along the line:

distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Build a list of the spline function, one for each dimension:
splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

# Computed the spline for the asked distances:
alpha = np.linspace(0, 1, 2*75)
# points_fitted = np.vstack( spl(alpha) for spl in splines ).T
points_fitted = np.vstack( list(spl(alpha) for spl in splines) ).T

plt.scatter(points_fitted[:,0],points_fitted[:,1],s=100,c='r')
plt.plot(points_fitted[:,0],points_fitted[:,1])


# Graph:
# plt.plot(*points.T, 'ok', label='original points')
# plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=.2')
plt.show()
