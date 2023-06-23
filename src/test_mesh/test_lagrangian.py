import numpy as np

def laplacian_F_function(x,y,seg):
    """
    compute the Laplacian of the distance function phi at coordinate (x,y)
    Input:
        x: the x-coordinate
        y: the y-coordinate
        seg: a list of segments (x1,y1,x2,y2)
    Output:
        PHI: the distance function
        GRAD: the gradiant of the distance function
        LAP: the laplacian of the distance function
    """

    def dist(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    def linseg(x,y,x1,y1,x2,y2):
        L = dist(x1,y1,x2,y2)
        xc = (x1+x2)/2.
        yc = (y1+y2)/2.
        f = (1/L)*((x-x1)*(y2-y1) - (y-y1)*(x2-x1))
        f_dx=(y2-y1)/L
        f_dxx=0
        f_dy=(x1-x2)/L
        f_dyy=0
        t = (1/L)*((L/2.)**2-dist(x,y,xc,yc)**2)
        t_dx=-2*(x-xc)/L
        t_dxx=2/L
        t_dy=-2*(y-yc)/L
        t_dyy=-2/L
        varphi = np.sqrt(t**2+f**4)
        varphi_dx=(2*t_dx*t + 4*f_dx*f**3)/(2*varphi)
        varphi_dxx=(2*t_dxx*t+2*t_dx**2+4*f_dxx*f**3+12*f**2*f_dx**2-2*varphi_dx**2)/(2*varphi)
        varphi_dy=(2*t_dy*t + 4*f_dy*f**3)/(2*varphi)
        varphi_dyy=(2*t_dyy*t+2*t_dy**2+4*f_dyy*f**3+12*f**2*f_dy**2-2*varphi_dy**2)/(2*varphi)
        phi = np.sqrt(f**2 + (1/4.)*(varphi - t)**2)
        phi_dx=(2*f_dx*f+0.5*(varphi_dx-t_dx)*(varphi-t))/(2*phi)
        phi_dxx=(2*f_dxx*f+2*f_dx**2+0.5*(varphi_dxx-t_dxx)*(varphi-t)+0.5*(varphi_dx-t_dx)**2-2*phi_dx**2)/(2*phi)
        phi_dy=(2*f_dy*f+0.5*(varphi_dy-t_dy)*(varphi-t))/(2*phi)
        phi_dyy=(2*f_dyy*f+2*f_dy**2+0.5*(varphi_dyy-t_dyy)*(varphi-t)+0.5*(varphi_dy-t_dy)**2-2*phi_dy**2)/(2*phi)
        return phi,phi_dx,phi_dy,phi_dxx,phi_dyy

    def PHI(x,y,seg) :
        m = 1
        rep=0
        phi_list=[]
        phi_dx_list=[]
        phi_dy_list=[]
        phi_dxx_list=[]
        phi_dyy_list=[]
        for i in range(len(seg)) :
            (x1,y1,x2,y2)=seg[i]
            phi,phi_dx,phi_dy,phi_dxx,phi_dyy=linseg(x,y,x1,y1,x2,y2)
            phi_list.append(phi)
            phi_dx_list.append(phi_dx)
            phi_dy_list.append(phi_dy)
            phi_dxx_list.append(phi_dxx)
            phi_dyy_list.append(phi_dyy)
            rep = rep + 1./phi**m
        rep = 1/rep**(1/m)
        return rep,phi_list,phi_dx_list,phi_dy_list,phi_dxx_list,phi_dyy_list


    def PHI_dx(PHI,phi_list,phi_dx_list) :
        rep=0
        for i in range(len(phi_list)) :
            rep+=(PHI/phi_list[i])**2 * phi_dx_list[i]
        return rep
    
    def PHI_dxx(PHI,phi_list,phi_dx_list,phi_dxx_list):
        rep=0
        for i in range(len(phi_list)) :
            rep+=2*PHI*PHI_dx(PHI,phi_list,phi_dx_list)*phi_dx_list[i]
            rep-=2*(PHI*phi_dx_list[i])**2/(phi_list[i])**3
            rep+=(PHI/phi_list[i])**2 *phi_dxx_list[i]
        return rep

    def PHI_dy(PHI,phi_list,phi_dy_list) :
        rep=0
        for i in range(len(phi_list)) :
            rep+=(PHI/phi_list[i])**2 * phi_dy_list[i]
        return rep
    
    def PHI_dyy(PHI,phi_list,phi_dy_list,phi_dyy_list):
        rep=0
        for i in range(len(phi_list)) :
            rep+=2*PHI*PHI_dy(PHI,phi_list,phi_dy_list)*phi_dy_list[i]
            rep-=2*(PHI*phi_dy_list[i])**2/(phi_list[i])**3
            rep+=(PHI/phi_list[i])**2 *phi_dyy_list[i]
        return rep
    
    f_dist,phi_list,phi_dx_list,phi_dy_list,phi_dxx_list,phi_dyy_list=PHI(x,y,seg)
    

    return f_dist**2, 2*f_dist*PHI_dx(f_dist,phi_list,phi_dx_list)+2*f_dist*PHI_dy(f_dist,phi_list,phi_dy_list), 2*f_dist*PHI_dxx(f_dist,phi_list,phi_dx_list,phi_dxx_list)+2*PHI_dx(f_dist,phi_list,phi_dx_list)**2+2*f_dist*PHI_dyy(f_dist,phi_list,phi_dy_list,phi_dyy_list)+2*PHI_dy(f_dist,phi_list,phi_dy_list)**2, np.array([2*f_dist*PHI_dx(f_dist,phi_list,phi_dx_list),2*f_dist*PHI_dy(f_dist,phi_list,phi_dy_list)])


a=[0,0]
b=[0,0.75]
c=[1,1]
d=[0.6,0.5]
e=[0.5,0.5]
f=[1,0]

seg=[a+b,b+c,c+d,d+f,f+a]



graph=2
# F : 0
# divF : 1
# lapF : 2

x=np.linspace(0,1,500)
y=np.linspace(0,1,500)[::-1]
X,Y = np.meshgrid(x, y)
Z = laplacian_F_function(X,Y,seg)[graph]


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point
from tqdm import tqdm

poly = Polygon([a,b,c,d,f])

Xcut=X.copy()
Ycut=Y.copy()
Zcut=Z.copy()
for i in tqdm(range(X.shape[0])) :
    for j in range(X.shape[1]) :
        x=X[i,j]
        y=Y[i,j]
        if (Point(x,y).within(poly)) :
            Xcut[i,j]=X[i,j]
            Ycut[i,j]=Y[i,j]
            Zcut[i,j]=Z[i,j]
        else :
            Xcut[i,j]=None
            Ycut[i,j]=None
            Zcut[i,j]=None


im = plt.imshow(Zcut,interpolation='none', extent=[0,1,0,1],cmap='hot') 
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X.flatten()[cond], Y.flatten()[cond], Z.flatten()[cond], cmap='hot', linewidth=0.1)
ax.plot_surface(Xcut, Ycut, Zcut,rstride=4,cstride=4, cmap='hot', linewidth=0)
plt.show()

