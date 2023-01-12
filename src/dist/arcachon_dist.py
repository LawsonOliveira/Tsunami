import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
from pathlib import Path
from dist3D import plan,dist,eval_function,get_xy,fonct,phi3D,finaldist

print("reading data...")

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"data/arcachon_bathymetry/dataset.xyz", sep=",", header=2)
data = data.rename(columns={"long(DD)": "long", "lat(DD)": "lat", "depth(m - down positive - LAT)": "depth"})
depth = data["depth"].to_numpy()
long = data["long"].to_numpy()
lat = data["lat"].to_numpy()


print("done")

points=np.array(list(zip(lat, long)))

print("triangle start...")
tri = Delaunay(points)
print("done")

H = depth[tri.simplices]

Lim = 0.01

def longchara(tria) :
    return (np.linalg.norm(tria[0]-tria[1]) + np.linalg.norm(tria[2]-tria[1]) + np.linalg.norm(tria[0]-tria[2]))/3

trineo = []
finaltri = []
for e in tri.simplices :
    if longchara(points[e]) < Lim :
        trineo.append(e)
        finaltri.append([[points[e[i]][0],points[e[i]][1],depth[e][i]] for i in range(3)])

finaltri = np.array(finaltri)
trineo = np.array(trineo)


## Domain :

# number of points :
n = 1000

# boundary :
Xmin = -1.35
Xmax = -1
Ymin = 44.5
Ymax = 44.8
Zmin = -10
Zmax = 10

x = np.random.uniform(Xmin,Xmax,n)
y = np.random.uniform(Ymin,Ymax,n)
z = np.random.uniform(Zmin,Zmax,n)

############################################# 3D RENDER ##############################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm

## plot the 2D mesh : 

# plt.triplot(lat, long, trineo)
# plt.show()

## plot the 3D mesh : 

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot_trisurf(lat, long, depth, triangles = trineo)
# plt.show()

## plot the 3D distance function :

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# c=[]

# for i in range(n) :
#     print('progress : ',i,'/',n)
#     c.append(finaldist(finaltri,np.array([x[i],y[i],z[i]])))

# c=np.array(c)
# img = ax.scatter(x, y, z, c=c*100, cmap=plt.hot())