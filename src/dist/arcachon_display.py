import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dist import phi
from tqdm import tqdm
# from deprecated.test_dist import orga_final_bis


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mainpath = Path(__file__).parents[0]
seg=np.load(mainpath/'saves/arcachon_segment_alpha_1000.npy')

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
lat = data[:,0]
long = data[:,1]
data_frontier = pd.read_csv(mainpath/"processed_data/Domain_height_2/normalized_frontier.csv").to_numpy()
X=data_frontier[:,0]
Y=data_frontier[:,1]

# visualize the frontier :

# fig, ax = plt.subplots(figsize=[5,4])
# ax.scatter(X,Y,s=1)
# axins = zoomed_inset_axes(ax, 20, loc=1) 
# axins.scatter(X,Y,s=5)
# x1, x2, y1, y2 = 0.265, 0.275, 0.51, 0.53
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()

# visualize the linking of the points together :

# margin=0.01
# XF,YF,DIST = orga_final_bis(X,Y,margin,margin/20)
# fig, ax = plt.subplots(figsize=[5,4])
# axins = zoomed_inset_axes(ax, 20, loc=1) 


# for i in range(len(XF)) :
#     Xd=XF[i]
#     Yd=YF[i]
#     for j in range(1,Xd.shape[0]) :
#         ax.plot([Xd[j-1],Xd[j]],[Yd[j-1],Yd[j]])
#         axins.plot([Xd[j-1],Xd[j]],[Yd[j-1],Yd[j]])

# x1, x2, y1, y2 = 0.265, 0.275, 0.51, 0.53
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()

# visualize the segments obtained by interpolation :

# margin=0.01
# fig, ax = plt.subplots(figsize=[5,4])
# axins = zoomed_inset_axes(ax, 20, loc=1) 
# ax.scatter(X,Y,s=2)
# axins.scatter(X,Y,s=2)
# Xseg=[]
# Yseg=[]
# for e in seg :
#     [x1,y1,x2,y2] = e
#     ax.plot([x1,x2],[y1,y2],c='r',linewidth=1)
#     axins.plot([x1,x2],[y1,y2],c='r',linewidth=1)
#     Xseg.append(x1)
#     Xseg.append(x2)
#     Yseg.append(y1)
#     Yseg.append(y2)

# ax.scatter(Xseg,Yseg,c='r',s=5)
# axins.scatter(Xseg,Yseg,c='r',s=20)

# x1, x2, y1, y2 = 0.265, 0.275, 0.51, 0.53
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()


# drawing the function :

n=500

x=np.linspace(0,1.4,n)
y=np.linspace(0,1,n)[::-1]
X,Y = np.meshgrid(x, y)

Z = phi(X,Y,seg)

levels = np.arange(0,0.01,0.0008)
cset = plt.contourf(X,Y,Z,levels=levels,cmap='hot')
cset2 = plt.contour(X,Y,Z,levels=levels,colors='k',linewidths=0.5)
plt.colorbar(cset)

plt.show()


# 3D render :


# Z=phi(lat,long,seg)
# Zmax = 0.01

# index = []

# for i in range(Z.shape[0]) :
#     if Z[i] < Zmax :
#         index.append(i)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(lat[index],long[index],Z[index],c=Z[index], cmap=plt.hot())

# plt.show()
