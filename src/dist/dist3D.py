import numpy as np
import sympy as sm
from dist import dist,linseg,phi,segment_gen

def plan(A,B,C) :
    """
    Returns f(x,y,z), a C1 function whose kernel is the plane defined by A, B, C.
    Also returns (U,V) two vectors in this plane.
        Input:
            A : a 3-dimensional array
            B : a 3-dimensional array
            C : a 3-dimensional array
    """
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
    """
    Compute the norm L1 of the vectore B-A
        Input:
            A : a 3-dimensional array
            B : a 3-dimensional array
    """
    return np.linalg.norm(B-A)

def eval_function(func, xi, yi, zi):
    """
    returns func(xi,yi,zi)
        Input:
            func : a scimpy function
            xi : the x-coordinate (real)
            yi : the y-coordinate (real)
            zi : the y-coordinate (real)
    """
    x, y, z = sm.symbols("x, y, z")
    temp = func.evalf(subs={x: xi, y: yi, z:zi})
    return temp

def get_xy(A,U,V,X) :
    """
    Returns the coordinates (x,y) on the plane (A,U,V) of the projection of the point X on the plane (A,U,V) 
        Input:
            A : a 3-dimensional array (a point in the space)
            U : a 3-dimensional array (a vector)
            V : a 3-dimensional array (a vector)
            X : a 3-dimensional array (a point in the space)
    """
    AX = X-A
    x=np.dot(AX,U)
    y=np.dot(AX,V)
    return (x,y)

def phi3D(f,t) :
    """
    compute the funcion distance with the value of f(X) and t(X)
        Input :
            f : the value of f(X) (real)
            t : the value of t(X) (real)
    """
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