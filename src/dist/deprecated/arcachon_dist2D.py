import numpy as np
import pandas as pd
from pathlib import Path
from dist import phi
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rd

def segment_gen_brute_force(points,epsilon) :
    """
    returns a list of segments [A,B] joining the points
    Input:
        points : a list of points A (2D array)
    """
    segments=[]
    for i in tqdm(range(points.shape[0])) :
        for j in range(i,points.shape[0]) :
            if i!=j and np.linalg.norm(points[i]-points[j]) < epsilon :
                segments.append(np.concatenate((points[j],points[i])))
    return np.array(segments)

def segment_gen_naive(points) :
    """
    returns a list of segments [A,B] joining the points
    Input:
        points : a list of points A (2D array)
    """
    segments=[]
    for i in tqdm(range(points.shape[0])) :
        segments.append(np.concatenate((points[i-1],points[i])))
    return np.array(segments)

print("reading data...")
mainpath = Path(__file__).parents[2]
data_frontier = pd.read_csv(mainpath/"processed_data/normalized_frontier.csv").to_numpy()
# data_frontier = pd.read_csv(mainpath/"Approaches/Lagaris_Paper2/Mild_slope_bassin/Data/normalized_frontier.csv").to_numpy()
print("done")

mainpath = Path(__file__).parents[2]
data = pd.read_csv(mainpath/"processed_data/normalized_domain.csv").to_numpy()
# data = pd.read_csv(mainpath/"Approaches/Lagaris_Paper2/Mild_slope_bassin/Data/normalized_domain.csv").to_numpy()
lat = data[:,0]
long = data[:,1]

# boundary :

xmin = 0
xmax = 1
ymin = 0
ymax = 1

La = lat
Lo = long

epsilon = 0.005

fig = plt.figure()
ax = fig.add_subplot()

print('segment generation...')
segments = segment_gen_brute_force(data_frontier[:,:2],epsilon)
print("done")

print(segments.shape)
# plt.scatter(*zip(*data_frontier[:,:2]))
print('compute the points')
Z = phi(La,Lo,segments)
print('done')

np.save('src/dist/arcachon_points',Z)

img = ax.scatter(La, Lo, c=Z,vmin=0, vmax=0.0003, cmap=plt.hot())
fig.colorbar(img)

plt.show()

