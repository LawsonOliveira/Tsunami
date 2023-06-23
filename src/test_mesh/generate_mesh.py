import numpy as np
import matplotlib.pyplot as plt
import random as rd
from perso_simple_env import SimpleEnvironment as SE

########################################## VARIABLES ############################################

# number of points :
n=10000

# depth of reference :
h=-10

# dimentions of the square :
xmin=0
xmax=1
ymin=0
ymax=1

# X slope coefficient :
alpha=0.3

# Y slope coefficient :
beta=0

## circular obstacle (empty list if no obstacle) :

# center of the circles :
Lc=[[0.5,0.5],[0.5,1]]

# corresponding radii :
Lr=[0.2,0.1]


## polygon cuts (empty list if no cuts) :

# lists of the vertices' coordinates for the polygons
points1=[[0.25,0.05],[0.05,0.15],[0.05,0.3],[0.25,0.45],[0.45,0.3],[0.45,0.15]]
points2=[[0,1],[0.2,1],[0.1,0.7],[0,0.8]]
points3=[[1,1],[0.8,1],[0.8,0.8],[1,0.8]]

polygons=[points1,points2,points3]


## noise generation :

# seed number :
seed=0

# maximum deviation from value :
epsilon=0.1

#################################################################################################

np.random.seed(seed)
data=SE([xmin,ymin,xmax,ymax])
x=np.linspace(xmin,xmax,int(np.sqrt(n)))
y=np.linspace(ymin,ymax,int(np.sqrt(n)))
X,Y=np.meshgrid(x,y)
H=h*np.ones(X.shape)
Z=alpha*X+beta*Y+H

for i in range(len(Lc)) :
    (xc,yc)=Lc[i]
    r=Lr[i]
    d=np.sqrt(abs(X-xc)**2+abs(Y-yc)**2)
    inside=d<=r
    Xinside=X[inside]
    Yinside=Y[inside]
    d=np.sqrt(abs(Xinside-xc)**2+abs(Yinside-yc)**2)
    Z[inside]+=np.sqrt(r**2-d**2)


for i in range(len(polygons)) :
    points=polygons[i]
    data.add_convex_polygon(np.array(points))

    inside=np.invert(data._is_in_this_polygon(i,X,Y))

    X=X[inside]
    Y=Y[inside]
    Z=Z[inside]

X=X.flatten()
Y=Y.flatten()
Z=Z.flatten()

Z+=np.random.rand(Z.shape[0])*epsilon

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z)
plt.show()