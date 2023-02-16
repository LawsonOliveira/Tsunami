import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from dist import phi

def is_inside_triangle(A,B,C,X) :
    xa=A[0]
    ya=A[1]
    xb=B[0]
    yb=B[1]
    xc=C[0]
    yc=C[1]
    x=X[0]
    y=X[1]
    c1 = (xb-xa)*(y-ya)-(yb-ya)*(x-xa)
    c2 = (xc-xb)*(y-yb)-(yc-yb)*(x-xb)
    c3 = (xa-xc)*(y-yc)-(ya-yc)*(x-xc)
    return (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0)


mainpath = Path(__file__).parents[2]
# data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
# frontier = pd.read_csv(mainpath/"processed_data/normalized_frontier.csv").to_numpy()
data = pd.read_csv(mainpath/"processed_data/Domain_height_3/normalized_domain.csv").to_numpy()
frontier = pd.read_csv(mainpath/"processed_data/Domain_height_3/normalized_frontier.csv").to_numpy()
lat = frontier[:,0].tolist()
long = frontier[:,1].tolist()

# plt.scatter(frontier[:,0],frontier[:,1])
# plt.show()

margin = 0.003
segmargin = 0.008
niter = 4
decrate = 0.85


# print("choosing points...")

# for k in range(niter) :
#     print('iteration : ',k+1,'out of ',niter)
#     seglat = []
#     seglong = []
#     pbar = tqdm(total = len(lat))
#     while(len(lat) >=1) :
#         B=np.zeros(2)
#         n=0
#         A=np.array([lat[0],long[0]])
#         i=0
#         while i < len(lat) :
#             X = np.array([lat[i],long[i]])
#             if np.linalg.norm(A-X) < margin :
#                 lat.pop(i)
#                 long.pop(i)
#                 B+=X
#                 n+=1
#                 pbar.update(1)
#             i+=1
#         seglat.append(B[0]/n)
#         seglong.append(B[1]/n)
#     pbar.close()
#     lat = seglat
#     long = seglong
#     margin*= decrate

# print("done")
# seglat=np.array(seglat)
# seglong=np.array(seglong)

# plt.scatter(seglat,seglong)




# plt.show()


# np.save('src/dist/saves/arcachon_frontier_lat',seglat)
# np.save('src/dist/saves/arcachon_frontier_long',seglong)

mainpath = Path(__file__).parents[0]
seglat=np.load(mainpath/'saves/arcachon_frontier_lat.npy')
seglong=np.load(mainpath/'saves/arcachon_frontier_long.npy')

# plt.scatter(seglat,seglong)



segdic={}
seg_interdit={}
print("building segments")
for i in tqdm(range(seglat.shape[0])) :
    if i not in segdic :
        segdic[i]=[]
    n=0
    for j in range(i,seglat.shape[0]) :
            A = np.array([seglat[i],seglong[i]])
            B = np.array([seglat[j],seglong[j]])
            if i!=j and np.linalg.norm(A-B) < segmargin :
                seg_interdit[(i,j)]=False
                seg_interdit[(j,i)]=False

                segdic[i].append(j)
                if j not in segdic :
                   segdic[j]=[]
                segdic[j].append(i)
                # plt.plot([seglat[i],seglat[j]],[seglong[i],seglong[j]],'b')
print("done")


# plt.show()



seg_final=[]


print("segment optimisation...")
for i in range(seglong.shape[0]) :
    if len(segdic)>=3 :
        neig=segdic[i]
        for j in neig :
            for k in neig :
                if k>j and j>i :
                    A = np.array([seglat[i],seglong[i]])
                    B = np.array([seglat[j],seglong[j]])
                    C = np.array([seglat[k],seglong[k]])
                    cycle = np.array([i,j,k])
                    points=np.array([A,B,C])
                    if np.linalg.norm(B-C) < segmargin :
                        Order = np.array([len(segdic[i]),len(segdic[j]),len(segdic[k])]).argsort()[::-1]
                        if Order[1] != Order[2] and len(segdic[Order[1]])>=3 :
                            cycle = cycle[Order]
                            points = points[Order]
                            if not seg_interdit[(cycle[0],cycle[1])] :
                                seg_interdit[(cycle[0],cycle[1])]=True
                                seg_interdit[(cycle[1],cycle[0])]=True
                                # plt.plot([seglat[cycle[0]],seglat[cycle[1]]],[seglong[cycle[0]],seglong[cycle[1]]],'r')

print("done")
    


print("drawing segment...")
for i in tqdm(range(seglong.shape[0])) :
    for j in range(i,seglat.shape[0]) :
        A = np.array([seglat[i],seglong[i]])
        B = np.array([seglat[j],seglong[j]])
        if i!=j and np.linalg.norm(A-B) < segmargin and not seg_interdit[(i,j)] :
            plt.plot([seglat[i],seglat[j]],[seglong[i],seglong[j]],'b')
            seg_final.append(np.concatenate((B,A)))
plt.show()
print("done")

np.save('src/dist/saves/arcachon_segment',seg_final)


# seg = np.array(seg_final)
# print('compute the distance function')
# La = data[:,0]
# Lo = data[:,1]
# Z = phi(La,Lo,seg)
# np.save('src/dist/arcachon_points',Z)
# print("done")

# fig = plt.figure()
# ax = fig.add_subplot()
# img = ax.scatter(La, Lo, c=Z, cmap=plt.hot())
# fig.colorbar(img)

# plt.show()