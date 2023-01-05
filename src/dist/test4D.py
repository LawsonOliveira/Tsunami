from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.standard_normal(1000)
y = np.random.standard_normal(1000)
z = np.random.standard_normal(1000)


img = ax.scatter(x, y, z, c=2*z+2*y, cmap=plt.hot())
fig.colorbar(img)
plt.show()