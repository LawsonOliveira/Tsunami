import numpy as np
import sympy as sm
from dist import dist,linseg,phi,segment_gen

def plan(A,B,C) :
    x, y, z = sm.symbols("x, y, z")
    X=np.array([x,y,z])
    AB=B-A
    AC=C-A
    W=np.cross(AB,AC)
    f = np.dot(X-A,W.T)
    U=AB-np.dot(AB,AC)*AC/(np.linalg.norm(AC)**2)
    V=AC
    U=U/np.linalg.norm(U)
    V=V/np.linalg.norm(V)
    return (f,U,V)

def dist(A,B) :
    return np.linalg.norm(B-A)

def eval_function(func, xi, yi, zi):
    x, y, z = sm.symbols("x, y, z")
    temp = func.evalf(subs={x: xi, y: yi, z:zi})
    return temp

def get_xy(A,U,V,X) :
    AX = X-A
    x=np.dot(AX,U)
    y=np.dot(AX,V)
    return (x,y)

def phi3D(f,t) :
    f=float(f)
    t=float(t)
    varphi = np.sqrt(t**2+f**4)
    return np.sqrt(f**2 + (1/4.)*(varphi - t)**2)

############################################# 3D PLOTTING ###########################################
print('3D plotting...')


# choisir les trois points du plan
A=np.array([0,0,1])
B=np.array([1,0,0])
C=np.array([1,1,0])

# nombre de points
n=2000

# bornes du domaine
min=-2
max=3

f,U,V =plan(A,B,C)

xa,ya =get_xy(A,U,V,A)
xb,yb =get_xy(A,U,V,B)
xc,yc =get_xy(A,U,V,C)
a = [xa,ya]
b = [xb,yb]
c = [xc,yc]

points=[a,b,c]
segments,Xdraw,Ydraw = segment_gen(points)

############################################# 3D RENDER ##############################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.uniform(min,max,n)
y = np.random.uniform(min,max,n)
z = np.random.uniform(min,max,n)

######### VISUALISER LA FONCTION F

# c=[]
# for i in range(n) :
#     if (i%500==0) :
#         print('progress : ',i,'/',n)
#     c.append(eval_function(f,x[i],y[i],z[i]))

# c=np.array(c)
# img = ax.scatter(x, y, z, c=c, cmap=cm.RdBu, norm=colors.CenteredNorm())

######### VISUALISER LA FONCTION PHI

# c=[]
# for i in range(n) :
#     if (i%500==0) :
#         print('progress : ',i,'/',n)
#     E=np.array([x[i],y[i],z[i]])
#     xe,ye = get_xy(A,U,V,E)
#     E=A+xe*U+ye*V
#     Test1 = np.dot(np.cross(E-A,B-A),np.cross(C-A,E-A))
#     Test2 = np.dot(np.cross(E-C,A-C),np.cross(B-C,E-C))
#     if Test1>0 and Test2 > 0 :
#         c.append(phi(xe,ye,segments))
#     else : 
#         c.append(-phi(xe,ye,segments))

# c=np.array(c)
# img = ax.scatter(x, y, z, c=c, cmap=cm.RdBu, norm=colors.CenteredNorm())

######### VISUALISER LA FONCTION DIST 

c=[]
for i in range(n) :
    if (i%500==0) :
        print('progress : ',i,'/',n)
    E=np.array([x[i],y[i],z[i]])
    xe,ye = get_xy(A,U,V,E)
    E=A+xe*U+ye*V
    Test1 = np.dot(np.cross(E-A,B-A),np.cross(C-A,E-A))
    Test2 = np.dot(np.cross(E-C,A-C),np.cross(B-C,E-C))
    if Test1>0 and Test2 > 0 :
        t= phi(xe,ye,segments)
    else : 
        t= -phi(xe,ye,segments)
    fonc = eval_function(f,x[i],y[i],z[i])
    c.append(phi3D(fonc,t))

c=np.array(c)
img = ax.scatter(x, y, z, c=c*100, cmap=plt.hot())

ax.scatter(A[0], A[1], A[2], c='blue', marker='*', s=200)
ax.scatter(B[0], B[1], B[2], c='blue', marker='*', s=200)
ax.scatter(C[0], C[1], C[2], c='blue', marker='*', s=200)
fig.colorbar(img)
plt.show()