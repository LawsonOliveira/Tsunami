import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
mainpath = Path(__file__).parents[0]
Z = np.load(mainpath/'arcachon_points.npy')
print(Z.shape)

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
# data = pd.read_csv(mainpath/"Approaches/Lagaris_Paper2/Mild_slope_bassin/Data/normalized_domain.csv").to_numpy()
lat = data[:,0]
long = data[:,1]

# drawing the function :

# levels = np.arange(0,0.005,0.0003)
# cset = plt.tricontourf(lat,long,Z,levels=levels,cmap='hot')
# cset2 = plt.tricontour(lat,long,Z,levels=levels,colors='k')
# plt.colorbar(cset)

# plt.show()


# drawing the function 2 :

# fig = plt.figure()
# ax = fig.add_subplot()
# img = ax.scatter(lat, long, c=Z, cmap=plt.hot())
# fig.colorbar(img)

# plt.show()

# 3D render :

Zmax = 0.005

index = []

for i in range(Z.shape[0]) :
    if Z[i] < Zmax :
        index.append(i)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lat[index],long[index],Z[index],c=Z[index], cmap=plt.hot())

plt.show()
