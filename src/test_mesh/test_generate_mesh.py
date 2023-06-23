import numpy as np
import matplotlib.pyplot as plt
import random as rd
from perso_simple_env import SimpleEnvironment as SE

np.random.seed(0)
print(np.random.rand(4))

# number of points
n=10000

# depth of reference
h=-10

# dimentions of the square :
xmin=0
xmax=1
ymin=0
ymax=1

 

data=SE([xmin,ymin,xmax,ymax])
x=np.linspace(xmin,xmax,int(np.sqrt(n)))
y=np.linspace(ymin,ymax,int(np.sqrt(n)))
X,Y=np.meshgrid(x,y)
H=h*np.ones(X.shape)

########################################## RAMP ############################################

# slope coefficient :
alpha=0.3



Z=alpha*X+H
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z)
plt.show()

################################### CIRCULAR OBSTACLE #####################################

# radius of the circle
r=0.2

# coordinates of the center:
(xc,yc)=(0.5,0.5)



d=np.sqrt(abs(X-xc)**2+abs(Y-yc)**2)
inside=d<=r
Z=np.zeros(X.shape)+H
Xinside=X[inside]
Yinside=Y[inside]
d=np.sqrt(abs(Xinside-xc)**2+abs(Yinside-yc)**2)
Z[inside]+=np.sqrt(r**2-d**2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z)
plt.show()


################################### SQUARE CUT FRONTIER #####################################

# list of centers of zones to remove :
Lc=[(0,0),(0.5,1),(0.5,0.5)]

# corresponding sizes of the squares :
Lr=[0.5,0.3,0.2]


Xplot=X
Yplot=Y

for i in range(len(Lc)) :
    (xc,yc)=Lc[i]
    r=Lr[i]
    d=(abs(abs(Xplot-xc)-abs(Yplot-yc))+abs(Xplot-xc)+abs(Yplot-yc))/2
    outside = d>=r/2
    Xplot=Xplot[outside]
    Yplot=Yplot[outside]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xplot,Yplot,0*Xplot)
plt.show()

################################### POLYGON CUT FRONTIER #####################################



# list of the vertices' coordinates for the polygon
points1=[[0.5,0.1],[0.1,0.3],[0.1,0.6],[0.5,0.9],[0.9,0.6],[0.9,0.3]]
points2=[[0,0],[0.2,0],[0.1,0.3],[0,0.2]]

polygons=[points1,points2]
Xplot=X
Yplot=Y

for i in range(len(polygons)) :
    points=polygons[i]
    data.add_convex_polygon(np.array(points))

    inside=np.invert(data._is_in_this_polygon(i,Xplot,Yplot))

    Xplot=Xplot[inside]
    Yplot=Yplot[inside]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xplot,Yplot,0*Xplot)
plt.show()