import numpy as np
import pandas as pd
from pathlib import Path

## define the functions :

def dist(A,B) :
    """
    Compute the norm L1 of the vectore B-A
        Input:
            A : a 3-dimensional array
            B : a 3-dimensional array
    """
    return np.linalg.norm(B-A)

def linseg(X,A,B) :
    """
    compute the distance function, evaluated in X to the segment [A,B]
        Input:
            X : 3D array
            A : 3D array
            B : 3D array
    """
    L=dist(A,B)
    C=(A+B)/2
    U=X-A
    V=B-A
    W=U-np.dot(U,V)*V/(np.linalg.norm(V)**2)
    f = (1/L)*np.dot((X-A),W)
    t = (1/L)*((L/2.)**2-dist(X,C)**2)
    varphi = np.sqrt(t**2+f**4)
    phi = np.sqrt(f**2 + (1/4.)*(varphi - t)**2)
    return  phi

def phi(X,segments):
    """
    compute the distance function evaluated in (x,y) to the set of segments listed in segments
        Input:
            x : real
            y : real
            segments : list of segments (x1,y1,x2,y2)
    """
    m = 1.
    R = 0.
    for i in range(segments.shape[0]):
        phi = linseg(X,segments[i,0],segments[i,1])
        R = R + 1./phi**m
    R = 1/R**(1/m)
    return R

def segment_gen(points) :
    """
    returns a list of segments [A,B] joining the points
    Input:
        points : a list of points A (3D array)
    """
    segments=[]
    for i in range(points.shape[0]) :
        segments.append([points[i-1],points[i]])
    return np.array(segments)

## process the data

print("reading data...")

mainpath = Path(__file__).parents[1]
data_frontier = pd.read_csv(mainpath/"Approaches/Lagaris_Paper2/Mild_slope_bassin/Data/normalized_frontier.csv").to_numpy()
print("done")
print(data_frontier.shape)
print('segment generation...')
segments = segment_gen(data_frontier)
print("done")

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"data/arcachon_bathymetry/dataset.xyz", sep=",", header=2)
data = data.rename(columns={"long(DD)": "long", "lat(DD)": "lat", "depth(m - down positive - LAT)": "depth"})
data = data.rename(columns={"long(DD)": "long", "lat(DD)": "lat", "depth(m - down positive - LAT)": "depth"})
depth = data["depth"].to_numpy()
long = data["long"].to_numpy()
lat = data["lat"].to_numpy()



# number of points :
n = 1000

# boundary :
Xmin = -1.35
Xmax = -1
Ymin = 44.5
Ymax = 44.8
Zmin = -1
Zmax = 1

############################################# 3D RENDER ##############################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm

## plot the 2D frontier : 

# plt.scatter(*zip(*data_frontier))
# plt.show()


## plot the 3D distance function :

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

c=[]

for i in range(depth.shape[0]) :
    if i%10 == 0 :
        print('progress : ',i,'/',depth.shape[0])
    X=np.array([lat[i],long[i],depth[i]])
    c.append(phi(X,segments))

c=np.array(c)
img = ax.scatter(lat, long, depth, c=c, cmap=plt.hot())

fig.colorbar(img)
plt.show()