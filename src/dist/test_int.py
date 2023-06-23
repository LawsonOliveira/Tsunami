import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dist import phi
from tqdm import tqdm

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"processed_data/Domain_height_2/normalized_domain.csv").to_numpy()
seg=np.load(mainpath/'src/dist/saves/arcachon_segment_alpha_1000.npy')
lat = data[:,0]
long = data[:,1]
depth=data[:,2]

for i in range(len(seg)) :
    segment=seg[i]
    x1=segment[0]
    y1=segment[1]
    x2=segment[2]
    y2=segment[3]
    plt.plot([x1,x2],[y1,y2],c='b')


plt.scatter(lat,long,c='r')
plt.show()

# margin=0.001

# for i in tqdm(range(lat.shape[0])) :
#     if depth[i] < 2 :
#         take = True
#         for j in range(lat.shape[0]) :
#             V=np.array([lat[i],long[i]])
#             W=np.array([lat[j],long[j]])
#             if i!=j and np.linalg.norm(V-W)<margin :
#                 if depth[j] > 2 :
#                     take = False
#         if take :
#             X.append(lat[i])
#             Y.append(long[i])
#             Z.append(depth[i])
    
# plt.scatter(X,Y)
# plt.show()


# # 3D RENDER :
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X, Y, Z, c='b', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()