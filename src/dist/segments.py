import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from dist import phi


mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
frontier = pd.read_csv(mainpath/"processed_data/normalized_frontier.csv").to_numpy()
lat = frontier[:,0].tolist()
long = frontier[:,1].tolist()

# plt.scatter(lat,long)

# margin = 0.0035
# segmargin = 0.0085
# niter = 4

margin = 0.004
segmargin = 0.0095
niter = 4

print("choosing points...")

for k in range(niter) :
    print('iteration : ',k+1,'out of ',niter)
    seglat = []
    seglong = []
    pbar = tqdm(total = len(lat))
    while(len(lat) >=1) :
        B=np.zeros(2)
        n=0
        A=np.array([lat[0],long[0]])
        i=0
        while i < len(lat) :
            X = np.array([lat[i],long[i]])
            if np.linalg.norm(A-X) < margin :
                lat.pop(i)
                long.pop(i)
                B+=X
                n+=1
                pbar.update(1)
            i+=1
        seglat.append(B[0]/n)
        seglong.append(B[1]/n)
    pbar.close()
    lat = seglat
    long = seglong

print("done")
seglat=np.array(seglat)
seglong=np.array(seglong)

plt.scatter(seglat,seglong)


seg=[]
print("building segments")
for i in tqdm(range(seglat.shape[0])) :
    n=0
    for j in range(i,seglat.shape[0]) :
            A = np.array([seglat[i],seglong[i]])
            B = np.array([seglat[j],seglong[j]])
            if i!=j and np.linalg.norm(A-B) < segmargin :
                seg.append(np.concatenate((B,A)))
                plt.plot([seglat[i],seglat[j]],[seglong[i],seglong[j]])
print("done")

# print("computing segments...")
# newseg = []

# sus_index = []
# for i in tqdm(range(seglat.shape[0])) :
#     n=0
#     neig = []
#     A = np.array([seglat[i],seglong[i]])
#     for j in range(seglat.shape[0]) :
#             B = np.array([seglat[j],seglong[j]])
#             if i!=j and np.linalg.norm(A-B) < segmargin :
#                 n+=1
#                 neig.append(j)
#     if n<=2 :
#         for j in neig :
#             B = np.array([seglat[j],seglong[j]])
#             newseg.append(np.concatenate((B,A)))
#             plt.plot([seglat[i],seglat[j]],[seglong[i],seglong[j]])
#     else :
#         j=neig[0]
#         Bmin = np.array([seglat[j],seglong[j]])
#         Cmin = Bmin
#         Nmin = np.inf
#         for j in neig :
#             B = np.array([seglat[j],seglong[j]])
#             N = np.linalg.norm(B-A)
#             if Nmin > N :
#                 Bmin = B
#                 Nmin = N
#         Nmin = np.inf
#         for j in neig :
#             C = np.array([seglat[j],seglong[j]])
#             N = np.linalg.norm(C-A)
#             if Nmin > N and C.all() != Bmin.all() :
#                 Cmin = C
#                 Nmin = N       
#         if np.linalg.norm(Cmin-Bmin) > segmargin :
#             newseg.append(np.concatenate((Cmin,A)))
#             plt.plot([A[0],Cmin[0]],[A[1],Cmin[1]])    
#         newseg.append(np.concatenate((Bmin,A)))
#         plt.plot([A[0],Bmin[0]],[A[1],Bmin[1]]) 
        
# print("done")



plt.show()


seg = np.array(seg)
print('compute the distance function')
data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
La = data[:,0]
Lo = data[:,1]
Z = phi(La,Lo,seg)
np.save('src/dist/arcachon_points',Z)
print("done")

fig = plt.figure()
ax = fig.add_subplot()
img = ax.scatter(La, Lo, c=Z, cmap=plt.hot())
fig.colorbar(img)

plt.show()