import numpy as np
import matplotlib.pyplot as plt

######################################## 2D RENDER #################################################

def dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def linseg(x,y,x1,y1,x2,y2):
    L = dist(x1,y1,x2,y2)
    xc = (x1+x2)/2.
    yc = (y1+y2)/2.
    f = (1/L)*((x-x1)*(y2-y1) - (y-y1)*(x2-x1))
    t = (1/L)*((L/2.)**2-dist(x,y,xc,yc)**2)
    varphi = np.sqrt(t**2+f**4)
    phi = np.sqrt(f**2 + (1/4.)*(varphi - t)**2)
    return phi

def phi(x,y,segments):
    m = 1.
    R = 0.
    for i in range(len(segments[:,0])):
        phi = linseg(x,y,segments[i,0],segments[i,1],segments[i,2],segments[i,3])
        R = R + 1./phi**m
    R = 1/R**(1/m)
    return R

def segment_gen(points) :
    segments=[]
    Xdraw=[]
    Ydraw=[]
    for i in range(len(points)) :
        segments.append(points[i-1]+points[i])
        Xdraw.append(points[i][0])
        Ydraw.append(points[i][1])
    Xdraw.append(points[0][0])
    Ydraw.append(points[0][1])
    return np.array(segments),Xdraw,Ydraw

######################################## 2D RENDER ##################################################
import random as rd


# chose dimention of the window :

# xmin=-1
# xmax=3
# ymin=-1
# ymax=3

# # choose points :

# a=[0,0]
# b=[-1,2]
# c=[2,3]
# d=[2,0]
# e=[1.5,1]
# f=[0.5,-1]

# points=[a,b,c,d,e,f]

# # different segments & the distance function :

# x=np.linspace(xmin,xmax)
# y=np.linspace(ymin,ymax)[::-1]
# X,Y = np.meshgrid(x, y)

# segments,Xdraw,Ydraw = segment_gen(points)
# plt.plot(Xdraw,Ydraw,linewidth=3)
# Z = phi(X,Y,segments)


# # drawing the function :

# im = plt.imshow(Z,interpolation='none', extent=[xmin,xmax,ymin,ymax]) 
# levels = np.arange(0,1,0.1)
# cset = plt.contour(X,Y,Z,levels=levels)
# plt.colorbar(im)

# plt.show()

# ############################################# 3D RENDER ##############################################

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
#                       cmap=cm.RdBu,linewidth=0, antialiased=False)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()