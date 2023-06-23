import pyvista as pv
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Reading files
data_ocean_earth = pv.read("./Mesh/mesh_arcachon.vtk")
data_ocean_earth = np.array(data_ocean_earth.GetPoints().GetData())

data_frontier = pd.read_csv("./Boundary/frontier.csv").to_numpy()
data_ocean = data_ocean_earth[data_ocean_earth[:,2]<0]

domain_bounds = np.column_stack(([np.min(data_ocean[:,0]),np.min(data_ocean[:,1]),np.min(data_ocean[:,2])],[np.max(data_ocean[:,0]),np.max(data_ocean[:,1]),0]))



# Remove duplicates
delta = set(map(tuple, data_frontier))
data_ocean = np.array([x for x in data_ocean if tuple(x) not in delta])


# Normalize inside domain
min_value, max_value = domain_bounds[:,0], domain_bounds[:,1]
XYZ_shoal = (data_ocean-min_value)/(max_value-min_value)        # Normalize
XYZ_shoal = np.column_stack((XYZ_shoal[:,0], XYZ_shoal[:,1], XYZ_shoal[:,2]))
np.savetxt("./normalized_domain.csv", XYZ_shoal, delimiter=",")


print(f'scale:{max_value-min_value}')

# Smooth boundary
def smooth_boundary(points,k_neigh=7):

    new_long=[]
    new_lat=[]
    new_depth=[]

    long=points[:,0]
    lat=points[:,1]
    matrix=np.array([long,lat]).T

    neigh = NearestNeighbors(n_neighbors=k_neigh)
    neigh.fit(matrix) 
    NearestNeighbors(algorithm='auto', leaf_size=30)

    for i in range(len(matrix)):
        if i%1000==999:
            print(f'{i+1} out of {len(matrix)}')
        neighbors=neigh.kneighbors([[matrix[i,0],matrix[i,1]]])[1][0]
        average_long=(np.sum([points[k,0] for k in neighbors])+points[i,0])/(k_neigh+1)
        average_lat=(np.sum([points[k,1] for k in neighbors])+points[i,1])/(k_neigh+1)
        average_depth=(np.sum([points[k,2] for k in neighbors])+points[i,2])/(k_neigh+1)
        new_long.append(average_long)
        new_lat.append(average_lat)
        new_depth.append(average_depth)
    
    return np.array([new_long, new_lat, new_depth]).T



xy_right = smooth_boundary(data_frontier, k_neigh=50)


# Normalize boundary
#min_value, max_value = domain_bounds[:,0], domain_bounds[:,1]
xy_right = (xy_right - min_value)/(max_value - min_value)
np.savetxt("./normalized_frontier.csv", xy_right, delimiter=",")