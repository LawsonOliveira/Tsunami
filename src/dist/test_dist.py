import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dist import phi,segment_gen2
import random as rd
from tqdm import tqdm
from scipy import interpolate as int




def gen_test(points,N_points=100) :
    """
    generates an example of a frontier passing through the points given
        Input
            points : an array of points
            N_points : int
    """
    # pbar = tqdm(total=len(points)*N_points)
    nuageX=[]
    nuageY=[]
    n=len(points)
    for i in range(n) :
        X1=points[i,0]
        Y1=points[i,1]
        X2=points[(i+1)%n,0]
        Y2=points[(i+1)%n,1]
        for j in range(N_points) :
            l=rd.random()
            Rx=X1+l*(X2-X1)+(0.5-rd.random())/200
            Ry=Y1+l*(Y2-Y1)+(0.5-rd.random())/200
            nuageX.append(Rx)
            nuageY.append(Ry)
            # pbar.update(1)
    # pbar.close()
    return (np.array(nuageX),np.array(nuageY))

def orga_point(X,Y,margin) :
    pbar = tqdm(total=X.shape[0])
    chosen=[False]*(X.shape[0])
    x0=X[0]
    y0=Y[0]
    chosen[0]=True
    dist=[0]
    Xd=[x0]
    Yd=[y0]
    done=False
    Left=True
    while not done :
        V0=np.array([x0,y0])
        done=True
        d0=np.Inf
        n_neig=0
        n_visited=0
        for i in range(X.shape[0]) :
            x=X[i]
            y=Y[i]
            V=np.array([x,y])
            d=np.linalg.norm(V-V0)
            if d<margin :
                n_neig +=1
                if (not chosen[i]) and d<d0 :
                    n_visited+=1
                    d0=d
                    j=i
        if d0 < np.Inf :
            if n_visited/n_neig < 0.1 :
                pbar.update(1)
                plt.plot([x0,X[j]],[y0,Y[j]])
                if Left :
                    dist.append(d0)
                    Xd.append(X[j])
                    Yd.append(Y[j])
                    
                else :
                    dist.insert(0, d0)
                    Xd.insert(0,X[j])
                    Yd.insert(0,Y[j])
                x0=X[j]
                y0=Y[j]
            done=False
            chosen[j]=True
        else :
            pbar.update(1)
            if Left :
                Left=False
                done=False
                if n_visited/n_neig < 0.1 :
                    x0=X[0]
                    y0=Y[0]
            chosen[j]=True
    pbar.close()
    return np.array(Xd),np.array(Yd),np.array(dist)

def orga_point_2(X,Y,margin,ind,chosen,m) :
    pbar = tqdm(total=X.shape[0]-m)
    x0=X[ind]
    y0=Y[ind]
    chosen[ind]=True
    dist=[0]
    Xd=[x0]
    Yd=[y0]
    done=False
    while not done :
        j=ind
        V0=np.array([x0,y0])
        done=True
        d0=np.Inf
        n_neig=0
        n_visited=0
        for i in range(X.shape[0]) :
            x=X[i]
            y=Y[i]
            V=np.array([x,y])
            d=np.linalg.norm(V-V0)
            if d<margin :
                n_neig +=1
                if (not chosen[i]) and d<d0 :
                    n_visited+=1
                    d0=d
                    j=i
        if d0 < np.Inf :
            if n_visited/n_neig < 0.1 :
                pbar.update(1)
                dist.append(d0)
                Xd.append(X[j])
                Yd.append(Y[j])
                x0=X[j]
                y0=Y[j]
            done=False
            chosen[j]=True
        else :
            pbar.update(1)
            chosen[j]=True
    pbar.close()
    return np.array(Xd),np.array(Yd),np.array(dist)

def orga_final(X,Y,margin,marginseg) :
    m=0
    XF=[]
    YF=[]
    DIST=[]
    chosen=[False]*(X.shape[0])
    for i in range(X.shape[0]) :
        if not chosen[i] :
            take=True
            for j in range(X.shape[0]) :
                V1=np.array([X[i],Y[i]])
                V2=np.array([X[j],Y[j]])
                if chosen[j] and np.linalg.norm(V1-V2)< marginseg :
                    take=False
                    chosen[i] = True
            if take :
                print('points visited : ',m,'/',X.shape[0])
                Xd,Yd,dist=orga_point_2(X,Y,margin,i,chosen,m)
                print('computing taken points...')
                if Xd.shape[0] > 10 :
                    Xinside=[]
                    Xoutside=[]
                    Yinside=[]
                    Youtside=[]
                    m=0
                    for i1 in tqdm(range(X.shape[0])) :
                        for j1 in range(Xd.shape[0]) :
                            V1=np.array([X[i1],Y[i1]])
                            V2=np.array([Xd[j1],Yd[j1]])
                            if np.linalg.norm(V1-V2)< marginseg :
                                chosen[i1] = True
                        if chosen[i1] :
                            m+=1
                            Xinside.append(X[i1])
                            Yinside.append(Y[i1])
                        else :
                            Xoutside.append(X[i1])
                            Youtside.append(Y[i1])
                    plt.scatter(Xinside,Yinside,c='r')
                    plt.scatter(Xoutside,Youtside,c='b')
                    plt.show()
                    XF.append(Xd)
                    YF.append(Yd)
                    DIST.append(dist)
            
    return XF,YF,DIST


def orga_final_bis(X,Y,margin,marginseg) :
    m=0
    XF=[]
    YF=[]
    DIST=[]
    chosen=[False]*(X.shape[0])
    for i in range(X.shape[0]) :
        if not chosen[i] :
            take=True
            for j in range(X.shape[0]) :
                V1=np.array([X[i],Y[i]])
                V2=np.array([X[j],Y[j]])
                if chosen[j] and np.linalg.norm(V1-V2)< marginseg :
                    take=False
                    chosen[i] = True
            if take :
                print('points remaining : ',X.shape[0]-m,' left ...')
                Xd,Yd,dist=orga_point_2(X,Y,margin,i,chosen,m)
                print('computing taken points...')
                if Xd.shape[0] > 10 :
                    # Xinside=[]
                    # Xoutside=[]
                    # Yinside=[]
                    # Youtside=[]
                    m=0
                    for i1 in range(X.shape[0]) :
                        if not chosen[i1] :
                            for j1 in range(Xd.shape[0]) :
                                V1=np.array([X[i1],Y[i1]])
                                V2=np.array([Xd[j1],Yd[j1]])
                                if np.linalg.norm(V1-V2)< marginseg :
                                    chosen[i1] = True
                        if chosen[i1] :
                            m+=1
                    #         Xinside.append(X[i1])
                    #         Yinside.append(Y[i1])
                    #     else :
                    #         Xoutside.append(X[i1])
                    #         Youtside.append(Y[i1])
                    # plt.scatter(Xinside,Yinside,c='r')
                    # plt.scatter(Xoutside,Youtside,c='b')
                    # plt.show()
                    XF.append(Xd)
                    YF.append(Yd)
                    DIST.append(dist)

    return XF,YF,DIST

def cleaning_points(points,X,Y,margin) :
    Xs=[]
    Ys=[]
    Xd=points[:,0]
    Yd=points[:,1]
    for i in range(Xd.shape[0]) :
        Inside=False
        for j in range(X.shape[0]) :
            V=np.array([X[j],Y[j]])
            Vd=np.array([Xd[i],Yd[i]])
            if np.linalg.norm(Vd-V) < margin :
                Inside=True
        if Inside :
            Xs.append(Xd[i])
            Ys.append(Yd[i])
        else :
            return np.array(Xs),np.array(Ys)


A=[0,0]
B=[1,0]
C=[1,1]
D=[0,1]
points1=np.array([A,B,C,D])

A=[0.3,0.3]
B=[0.6,0.3]
C=[0.6,0.6]
D=[0.3,0.6]
points2=np.array([A,B,C,D])

A=[0.85,0.75]
B=[0.85,0.8]
C=[0.9,0.8]
D=[0.9,0.75]
points3=np.array([A,B,C,D])

A=[0.15,0.75]
B=[0.15,0.8]
C=[0.1,0.8]
D=[0.1,0.75]
points4=np.array([A,B,C,D])

N_point=500
print('generating points...')
X1,Y1=gen_test(points1,N_point)
X2,Y2=gen_test(points2,N_point)
X3,Y3=gen_test(points3,N_point)
X4,Y4=gen_test(points4,N_point)
X=np.concatenate((X1,X2,X3,X4))
print(X.shape)
Y=np.concatenate((Y1,Y2,Y3,Y4))
print('done')

mainpath = Path(__file__).parents[2]
data_frontier = pd.read_csv(mainpath/"processed_data/Domain_height_2/normalized_frontier.csv").to_numpy()
X=data_frontier[:,0]
Y=data_frontier[:,1]

plt.scatter(X,Y)
plt.show()
margin=0.01
print('organizing the points...')
# Xd,Yd,dist=orga_point(X,Y,margin)
# plt.scatter(Xd,Yd,c='y')


XD,YD,DIST=orga_final_bis(X,Y,margin,margin/10)
print('done')


Seg_coords=[]

for i in range(len(XD)) :
    print(i+1,'/',len(XD))
    Xd=XD[i]
    Yd=YD[i]
    dist=DIST[i]

    points_seg=[]

    for j in range(dist.shape[0]) :
        if dist[j] == 0 and j!=0 :
            print(j)

    distance=np.cumsum(dist)
    modified=False
    if distance.shape != np.unique(distance).shape :
        print('oups...')
        modified=True
        distance = np.unique(distance)
        Xd=Xd[1:Xd.shape[0]]
        Yd=Yd[1:Yd.shape[0]]

    points=np.vstack( (Xd, Yd) ).T


    # Build a list of the spline function, one for each dimension:

    splines = [int.make_interp_spline(distance, coords, k=3) for coords in points.T]
    # splines = [int.UnivariateSpline(distance, coords, k=3, s=0.2) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 10, 1000)
    points_fitted = np.vstack( list(spl(alpha) for spl in splines) ).T

    print('cleaning points...')
    Xs,Ys=cleaning_points(points_fitted,X,Y,margin)
    print('done')


    plt.scatter(Xd,Yd,c='b')
    plt.scatter(Xs,Ys,s=10,c='r')
    plt.plot(Xs,Ys,c='r')
    if modified :
        print(points_fitted.shape)
        plt.scatter(points_fitted[:,0],points_fitted[:,1],s=5,c='g')
        
    for j in range(Xs.shape[0]) :
        x=Xs[j]
        y=Ys[j]
        points_seg.append([x,y])
    Seg_coords.append(points_seg)
plt.show()

seg_final=[]
for L in Seg_coords :
    seg=segment_gen2(np.array(L))[0]
    seg_final.append(np.array(seg))
n=500

x=np.linspace(0,1.4,n)
y=np.linspace(0,1,n)[::-1]
X,Y = np.meshgrid(x, y)

Z=1/phi(X,Y,seg_final[0])
for index in range(1,len(seg_final)) :
    seg=seg_final[index]
    Z += 1/phi(X,Y,seg)

Z=1/Z



levels = np.arange(0,0.01,0.0008)
cset = plt.contourf(X,Y,Z,levels=levels,cmap='hot')
cset2 = plt.contour(X,Y,Z,levels=levels,colors='k',linewidths=0.5)
plt.colorbar(cset)

plt.show()