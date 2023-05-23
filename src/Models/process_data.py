from sklearn.neighbors import NearestNeighbors
import numpy
import jax

class pre_process():
    def __init__(self, data_sea, data_earth, k_neigh = 7):

        self.data = numpy.row_stack((data_sea, data_earth))
        self.data[:,:2] = self.data[:,:2]*111000

        self.smooth_data = self.smooth_grid(self.data, k_neigh = k_neigh)
        self.smooth_frontier = self.get_frontier(self.smooth_data, k_neigh = 7)
        self.smooth_maritime_data = self.smooth_data[self.smooth_data[:,2]<0]

    def smooth_grid(self, matrix, k_neigh):

        coordinates = []

        neigh = NearestNeighbors(n_neighbors=k_neigh)
        neigh.fit(matrix) 
        NearestNeighbors(algorithm='auto', leaf_size=30)

        for i in range(len(matrix)):
            if i%10000==9999:
                print(f'Smoothing {i+1} out of {len(matrix)}')
            neighbors=neigh.kneighbors([[matrix[i,0],matrix[i,1],matrix[i,2]]])[1][0]
            average_long=(numpy.sum([matrix[k,0] for k in neighbors])+matrix[i,0])/(k_neigh+1)
            average_lat=(numpy.sum([matrix[k,1] for k in neighbors])+matrix[i,1])/(k_neigh+1)
            average_depth=(numpy.sum([matrix[k,2] for k in neighbors])+matrix[i,2])/(k_neigh+1)-1e-2
            coordinates.append(numpy.array([average_long, average_lat, average_depth]))
        
        return numpy.row_stack(coordinates)


    def get_frontier(self, matrix, k_neigh):
        coordinates = []
        neigh = NearestNeighbors(n_neighbors=k_neigh)
        neigh.fit(matrix)

        for i in range(matrix.shape[0]):
            neighbors = neigh.kneighbors([[matrix[i, 0], matrix[i, 1], matrix[i, 2]]])[1][0]
            if (matrix[neighbors,2]>0).any() and (matrix[neighbors,2]<0).any():
                coordinates.append(matrix[neighbors])
            if i%10000==9999:
                print(f'getting frontier {i+1} out of {matrix.shape[0]}')

        coordinates = numpy.unique(numpy.row_stack(coordinates), axis=0)
        coordinates = coordinates[coordinates[:,2]<0]
        return coordinates
    
    
    def get_smooth_data(self, metters_scalling = True):
        if metters_scalling:
            return self.smooth_data
        else:
            return numpy.column_stack((self.smooth_data[:,0]/111000,
                                    self.smooth_data[:,1]/111000,
                                    self.smooth_data[:,2]))
        
    def get_smooth_frontier(self, metters_scalling=True): 
        if metters_scalling:
            return self.smooth_frontier
        else:
            return numpy.column_stack((self.smooth_frontier[:,0]/111000,
                                    self.smooth_frontier[:,1]/111000,
                                    self.smooth_frontier[:,2]))

    def normalize(self, data):
        domain_bounds = numpy.column_stack(([numpy.min(self.smooth_maritime_data[:,0]),numpy.min(self.smooth_maritime_data[:,1]),numpy.min(self.smooth_maritime_data[:,2])],[numpy.max(self.smooth_maritime_data[:,0]),numpy.max(self.smooth_maritime_data[:,1]),0.0]))
        min_value, max_value = domain_bounds[:,0], domain_bounds[:,1]
        scale = numpy.max(max_value-min_value)

        XY_shoal = (data[:,:2]-min_value[:2])/scale        # Normalize
        Z_shoal = data[:,2]/scale
        normalized_data = numpy.column_stack((XY_shoal[:,0], XY_shoal[:,1], Z_shoal))
        
        return normalized_data
    
    
    def get_scale(self):
        domain_bounds = numpy.column_stack(([numpy.min(self.smooth_maritime_data[:,0]),numpy.min(self.smooth_maritime_data[:,1]),numpy.min(self.smooth_maritime_data[:,2])],[numpy.max(self.smooth_maritime_data[:,0]),numpy.max(self.smooth_maritime_data[:,1]),0.0]))
        min_value, max_value = domain_bounds[:,0], domain_bounds[:,1]
        scale = numpy.max(max_value-min_value)
        return scale
    

    def get_metters_domain_bounds(self):
        domain_bounds = numpy.column_stack(([numpy.min(self.smooth_maritime_data[:,0]),numpy.min(self.smooth_maritime_data[:,1]),numpy.min(self.smooth_maritime_data[:,2])],[numpy.max(self.smooth_maritime_data[:,0]),numpy.max(self.smooth_maritime_data[:,1]),0.0]))
        return domain_bounds


class post_process():
    def __init__(self, function, long_lat_depth):
        self.function = function
        self.long_lat_depth = long_lat_depth
    

    def long_lat_to_normalized(self, x, y):
        metters_scaled_data = numpy.column_stack((self.long_lat_depth[:,0]*111000, self.long_lat_depth[:,1]*111000, self.long_lat_depth[:,2]))
        domain_bounds = numpy.column_stack(([numpy.min(metters_scaled_data[:,0]),numpy.min(metters_scaled_data[:,1]),numpy.min(metters_scaled_data[:,2])],[numpy.max(metters_scaled_data[:,0]),numpy.max(metters_scaled_data[:,1]),0.0]))
        min_value, max_value = domain_bounds[:,0], domain_bounds[:,1]
        scale = numpy.max(max_value-min_value)

        xy_normalized = (jax.numpy.column_stack((x*111000, y*111000))-min_value[:2])/scale        # Normalize
        return xy_normalized, scale

    def convert_to_output(self, params, x, y, t):
        xy_normalized_data, scale = self.long_lat_to_normalized(x, y)
        output = self.function(params, xy_normalized_data[:,0], xy_normalized_data[:,1], t)
        return output*scale

