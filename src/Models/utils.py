import numpy
import open3d
import jax
import functools
import matplotlib.pyplot
import pyvista, trimesh
from matplotlib.animation import FuncAnimation

############################################################################################
###################################### Plot functions ######################################
############################################################################################
def plot_normals_2d(points, normals, img_path):
    """
    Plot the normals of each point

    Parameters
    ----------
    points : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the point
    normals : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the normals
    img_path : string
        -- path to save the image

    Returns
    -------
    """
    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(14.0, 14.0)
    matplotlib.pyplot.quiver(points[:,0], points[:,1], normals[:,0], normals[:,1], width = 0.0005)
    fig.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()

    return 




def plot_2d_domain(points, img_path, title, color = 'red', legend = ['Training set']):
    """
    Plot the two dimensional domain

    Parameters
    ----------
    points : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the point
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    color : string
        -- color of the plot
    legend : list[]
        -- legend of the plot


    Returns
    -------
    """
    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(18, 8.0)
    ax.set_title(title)
    graph = matplotlib.pyplot.scatter(points[:,0], points[:,1], color = color, s = 5)
    __ = ax.legend(legend)
    matplotlib.pyplot.savefig(img_path, facecolor = 'white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return




def plot_2d_domain_with_boundary(inside, boundaries, img_path, title, colors = ['red'], legend = ['Inside', 'Boundary']):
    """
    Plot the two dimensional domain with the boundaries

    Parameters
    ----------
    inside : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) inside the domain
    boundaries : numpy.ndarray[[N_points, 2]] or a list of numpy.ndarray[[N_points, 2]]
        -- boundaries coordinates (x, y)
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    colors : [string]
        -- colors of the boundary
    legend : list[]
        -- legend of the plot

    Returns
    -------
    """
    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(14.0, 14.0)
    ax.set_title(title)
    if isinstance(boundaries, list):
        matplotlib.pyplot.scatter(inside[:,0], inside[:,1], color='green', s=0.05)
        for i in range(len(boundaries)):
            matplotlib.pyplot.scatter(boundaries[i][:,0], boundaries[i][:,1], color=colors[i], s=0.05)
    else:
        matplotlib.pyplot.scatter(inside[:,0], inside[:,1], color='green', s=5.0)
        matplotlib.pyplot.scatter(boundaries[:,0], boundaries[:,1], color='red', s=5.0)
    ax.legend(legend)
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return




def plot_2d_colormesh(params, function, domain_bound, img_path, title, color='rainbow', npoints = 300):
    """
    Plot the solution using a pcolormesh 2d

    Parameters
    ----------
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    domain_bounds : numpy.ndarray[[2, 2]]
        -- minimum and maximum coordinates on each axis
    img_path : string
        -- path to save the image
    function : object
        -- fuction to plot
    title : string
        -- title of the plot
    color : string
        -- cmap
    npoints : int
        -- number of the points on each axis

    Returns
    -------
    mean_values : float
        -- mean value of the function in the domain
    """
    x, y = numpy.meshgrid(numpy.linspace(domain_bound[0,0], domain_bound[0,1], npoints),numpy.linspace(domain_bound[1,0], domain_bound[1,1], npoints))
    values = numpy.zeros((npoints, npoints))

    fig, ax = matplotlib.pyplot.subplots(facecolor='white')
    fig.set_size_inches(18, 7.2)
    fig.tight_layout(h_pad=3)
    ax.set_title(title)

    for i in range(npoints):
        values[i,:] = function(params, x[i,:], y[i,:])
    graph = matplotlib.pyplot.pcolormesh(x, y, values, cmap = color)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  
    mean_values = numpy.mean(values)

    return mean_values




def plot_2d_colormesh_complex(params, function, domain_bounds, img_path, title, npoints=200):
    """
    Plot the complex function using a pcolormesh 2d (real and imag part)

    Parameters
    ----------
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    domain_bounds : numpy.ndarray[[2, 2]]
        -- minimum and maximum coordinates on each axis
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    npoints : int
        -- number of the points on each axis

    Returns
    -------
    """
    real_values = numpy.zeros((npoints, npoints))
    imag_values = numpy.zeros((npoints, npoints))

    x, y = numpy.meshgrid(numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], npoints), numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], npoints))

    fig, ax = matplotlib.pyplot.subplots(1,2)
    fig.set_size_inches(18, 7.2)
    fig.suptitle(title)

    for i in range(npoints):
        print("Plotting: {} out of {}".format(i+1, npoints), end='\r')
        real_values[i,:] = jax.numpy.real(functools.partial(function, params)(x[i,:], y[i,:])) 
        imag_values[i,:] = jax.numpy.imag(functools.partial(function, params)(x[i,:], y[i,:]))

    ax[0].set_title('Real')
    graph = ax[0].pcolormesh(x, y, real_values, cmap = 'rainbow')
    matplotlib.pyplot.colorbar(graph, ax=ax[0])

    ax[1].set_title('Imaginary')
    graph = ax[1].pcolormesh(x, y, imag_values, cmap = 'rainbow')
    matplotlib.pyplot.colorbar(graph, ax=ax[1])

    matplotlib.pyplot.savefig(img_path, facecolor = 'white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return 




def plot_frames2d2(data, params, function, times, img_path, title, scale = 1):
    """
    Plot several frames 2d of the solution

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the point
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    scale : float
        -- scale of the plot

    Returns
    -------
    """
    def plot_one_frame2d(data, params, function, time, scale, ax):
        npoints = data.shape[0]
        x, y = data[:,0], data[:,1]
        t = jax.numpy.ones(npoints)*time


        ax.set_title(f'time(s) = {time}')

        values = function(params, x, y, t)
        min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))

        graph = ax.scatter(x, y, c=scale*values, s=1.0, cmap = 'rainbow', vmin = min_values, vmax = max_values)

        return graph


    fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=len(times), sharey=True, sharex=True, figsize=(22, 4), constrained_layout=True)
    fig.suptitle(title)
    fig.set_facecolor('white')
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")
    for i, time in enumerate(times):
        graph = plot_one_frame2d(data, params, function, time, scale, ax=axs[i])

    cbar = fig.colorbar(graph, ax=axs.ravel().tolist())
    cbar.set_label('Elevation(m)', rotation=90)

    fig.savefig(img_path, bbox_inches='tight')

    return 

def plot_frames2d(data, params, function, times, img_path, title, scale = 1):
    """
    Plot several frames 2d of the solution

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the point
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    scale : float
        -- scale of the plot

    Returns
    -------
    """
    def plot_one_frame2d(data, params, function, time, scale, ax):
        npoints = data.shape[0]
        x, y = data[:,0], data[:,1]
        t = jax.numpy.ones(npoints)*time


        ax.set_title(f'time(s) = {time}')

        values = function(params, x, y, t)
        min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))

        graph = ax.scatter(x, y, c=scale*values, s=0.01, cmap = 'rainbow', vmin = min_values, vmax = max_values)

        return graph


    fig, axs = matplotlib.pyplot.subplots(nrows=times.shape[0], ncols=times.shape[1], sharey=True, sharex=True, figsize=(int(times.shape[1]*4), int(times.shape[0]*4)), constrained_layout=True)
    fig.suptitle(title)
    fig.set_facecolor('white')
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")


    for i in range(times.shape[0]):
      for j in range(times.shape[1]):
        if times.shape[0]<2:
          graph = plot_one_frame2d(data, params, function, times[i,j], scale, axs[j])
        else:
          graph = plot_one_frame2d(data, params, function, times[i,j], scale, axs[i, j])
 

    cbar = fig.colorbar(graph, ax=axs.ravel().tolist())
    cbar.set_label('Elevation(m)', rotation=90)

    fig.savefig(img_path, bbox_inches='tight')

    return 


def plot_2d_animation(data, params, function, end_time, frames, img_path, scale = 1):
    """
    Plot an animation 2d of the solution

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 2]]
        -- coordinates (x, y) of the point
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    end_time : float
        -- end time of the animation
    frames : int
        -- number of time steps
    img_path : string
        -- path to save the image
    scale : float
        -- scale of the plot

    Returns
    -------
    """
    npoints = data.shape[0]
    step = end_time/frames
    x, y = data[:,0], data[:,1]
    t = jax.numpy.ones(npoints)

    def update_graph(num):
        print("Animating: {} out of {}".format(num+1, frames), end='\r')
        time = t*step*(num+1)
        values = function(params, x, y, time)
        title.set_text('Displacement, time step = {}'.format(num+1))
        graph.set_array(scale*values.flatten())
    
        return fig,

    fig, ax = matplotlib.pyplot.subplots(facecolor='white')
    fig.set_size_inches(14.0, 14.0)
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")
    title = ax.set_title('Displacement, time step = 1')
    values = function(params, x, y, jax.numpy.zeros(npoints))
    min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))

    graph = matplotlib.pyplot.scatter(x, y, c=scale*values, s=5.0, cmap = 'rainbow', vmin = min_values, vmax = max_values)
    matplotlib.pyplot.clim(vmin = min_values, vmax = max_values)
    matplotlib.pyplot.colorbar(graph, ax=ax)

    ani = FuncAnimation(fig, update_graph, frames = frames,  blit = False)
    ani.save(img_path, writer='pillow', fps = 10)
    matplotlib.pyplot.show()  

    return 




def plot_3d_domain(data, img_path):
    """
    Plot the domain 3d

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 3]]
        -- coordinates (x, y, z) of the point
    img_path : string
        -- path to save the image

    Returns
    -------
    """
    fig = matplotlib.pyplot.figure()
    fig.set_size_inches(14.0, 14.0)
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title('Spatial domain - 3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], color='green', s=1)
    ax.view_init(45,225)
    __ = ax.legend(['Ground'])
    fig.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return
    



def plot_3d_animation(data, params, function, end_time, frames, img_path, scale = 1):
    
    """
    Plot an animation 3d of the solution

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 3]]
        -- coordinates (x, y, z) of the point
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    end_time : float
        -- end time of the animation
    frames : int
        -- number of time steps
    img_path : string
        -- path to save the image
    scale : float
        -- scale of the plot

    Returns
    -------
    """
    npoints = data.shape[0]
    step = end_time/frames
    x, y, z = data[:,0], data[:,1], data[:,2]
    t = jax.numpy.ones(npoints)

    def update_graph(num):
        ax.cla()

        
        print("Animating: {} out of {}".format(num+1, frames), end='\r')
        
        time = t*step*(num+1)
        ax.scatter(x, y, z, color='green', s=10)

        values = function(params, x, y, time)
        values = function(params, x, y, time)
        graph = ax.scatter(x, y, scale*values, c=scale*values, s=10, cmap = 'rainbow')
        ax.set_zlim([-1, max_values])
        ax.set_title('Displacement, time step = {}'.format(num+1))

        graph.set_clim(vmin = min_values, vmax = max_values)
        __ = ax.legend(['Shoal', "Wave"])
        return fig,

    fig = matplotlib.pyplot.figure(facecolor='white')
    fig.set_size_inches(14.0, 14.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,225)

    ax.scatter(x, y, z, color='green', s=10)

    values = function(params, x, y, jax.numpy.zeros(npoints))
    min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))
    graph = ax.scatter(x, y, scale*values, c=scale*values, s=10, cmap = 'rainbow')
    ax.set_zlim([-1, max_values])

    graph.set_clim(vmin = min_values, vmax = max_values)
    __ = ax.legend(["Wave"])
    matplotlib.pyplot.colorbar(graph, ax=ax)
    ani = FuncAnimation(fig, update_graph, frames = frames,  blit = False)
    ani.save(img_path, writer='pillow', fps = 10)
    matplotlib.pyplot.show()  

    return 





def plot_frames3d(data, params, function, times, img_path, title, scale = 1):
    """
    Plot several frames 3d of the solution

    Parameters
    ----------
    data : numpy.ndarray[[batch_size, 3]]
        -- coordinates (x, y, z) of the point
    params : list of parameters[[w1,b1],...,[wn,bn]]
        -- weights and bias
    function : object
        -- fuction to plot
    img_path : string
        -- path to save the image
    title : string
        -- title of the plot
    scale : float
        -- scale of the plot

    Returns
    -------
    """
    def plot_one_frame3d(data, params, function, time, scale, ax):
        npoints = data.shape[0]
        x, y = data[:,0], data[:,1]
        t = jax.numpy.ones(npoints)*time


        ax.set_title(f'time(s) = {time}')

        values = function(params, x, y, t)
        min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))

        graph = ax.scatter(x, y, scale*values, c=scale*values,  s=10, vmin = min_values, vmax = max_values, cmap = 'rainbow')
        ax.set_zlim([-1.0, max_values])
        ax.view_init(45,225)

        return graph


    fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=len(times), subplot_kw=dict(projection='3d'), figsize=(22, 4), constrained_layout=True)
    fig.suptitle(title)
    fig.set_facecolor('white')
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")

    for i, time in enumerate(times):
        graph = plot_one_frame3d(data, params, function, time, scale, ax=axs[i])


    cbar = fig.colorbar(graph, ax=axs.ravel().tolist())
    cbar.set_label('Elevation(m)', rotation=90)

    fig.savefig(img_path, bbox_inches='tight')

    return




def plot_loss_history(losses, img_path, legend):
    """
    Plot the loss function

    Parameters
    ----------
    losses : numpy.ndarray[[batch_size, ]] or a list of numpy.ndarray[[batch_size, ]]
        -- loss 
    img_path : string
        -- path to save the image
    Returns
    -------
    """
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    fig.set_size_inches(18.0, 14.0)
    if isinstance(losses[0], list):
        for loss_values in losses:
            __ = ax.plot(numpy.log10(loss_values))
    else:
        __ = ax.plot(numpy.log10(losses))
    ax.set_xlabel(r'${\rm Epoch}$')
    ax.set_ylabel(r'$\log_{10}{\rm (loss)}$')
    ax.set_title(r'${\rm Training}$')
    ax.legend(legend)
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()

    return 




############################################################################################
################################ Generate domain functions #################################
############################################################################################
def gen_inside_square(domain_bounds, n_inside):
    """
    Generate a square domain

    Parameters
    ----------
    domain_bounds : numpy.ndarray[[2, 2]]
        -- minimum and maximum coordinates on each axis
    n_inside : int
        -- number of points inside the domain

    Returns
    -------
    inside_points : numpy.ndarray[[n_inside, 2]]
        -- coordinates (x, y) of the domain
    """
    ### Inside data
    x = jax.numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], int(jax.numpy.sqrt(n_inside))+1, endpoint=False)[1:]
    y = jax.numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], int(jax.numpy.sqrt(n_inside))+1, endpoint=False)[1:]
    x, y = jax.numpy.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    inside_points = jax.numpy.column_stack((x, y))

    return inside_points




def gen_boundary_square(domain_bounds, n_bound):
    """
    Generate a square boundary

    Parameters
    ----------
    domain_bounds : numpy.ndarray[[2, 2]]
        -- minimum and maximum coordinates on each axis
    n_bound : int
        -- number of points on the boundary
    Returns
    -------
    boundary_points : numpy.ndarray[[n_inside, 2]]
        -- coordinates (x, y) of the boundary
    """
    #### Boundary data ---- 4 edges
    ### Left
    x = jax.numpy.linspace(domain_bounds[0,0],domain_bounds[0,0],1, endpoint = False)
    y = jax.numpy.linspace(domain_bounds[1,0],domain_bounds[1,1],n_bound//4, endpoint = False)
    x, y = jax.numpy.meshgrid(x,y)
    x, y = x.flatten(), y.flatten()
    xy_left = jax.numpy.column_stack((x, y))


    ### Behind
    x = jax.numpy.linspace(domain_bounds[0,0],domain_bounds[0,1],n_bound//4, endpoint = False)
    y = jax.numpy.linspace(domain_bounds[1,1],domain_bounds[1,1],1, endpoint = False)
    x, y = jax.numpy.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    xy_behind = jax.numpy.column_stack((x, y))


    ### Right
    x = jax.numpy.linspace(domain_bounds[0,1],domain_bounds[0,1],1, endpoint = False)
    y = jax.numpy.linspace(domain_bounds[1,1],domain_bounds[1,0],n_bound//4, endpoint = False)
    x, y = jax.numpy.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    xy_right = jax.numpy.column_stack((x, y))


    ### Front
    x = jax.numpy.linspace(domain_bounds[0,1],domain_bounds[0,0],n_bound//4, endpoint = False)
    y = jax.numpy.linspace(domain_bounds[1,0],domain_bounds[1,0],1, endpoint = False)
    x, y = jax.numpy.meshgrid(x,y)
    x, y = x.flatten(), y.flatten()
    xy_front = jax.numpy.column_stack((x, y))

    boundary_points = jax.numpy.concatenate((xy_left, xy_behind, xy_right, xy_front))

    return boundary_points




############################################################################################
################################## Get normals functions ###################################
############################################################################################
def map_from_vtk_polydata_file(file_name):
    """For UNSTRUCTURED GRID VTK and POLYDATA"""

    # file_name is a string, for example, "mesh.vtk"
    pvmesh = pyvista.read(file_name).cast_to_unstructured_grid()
    nelems = pvmesh.number_of_cells

    # faces and p_elem2nodes_shifted
    faces = pvmesh.cells
    p_elem2nodes_shifted = pvmesh.offset + range(nelems + 1)

    # Now we will find p_elem2nodes, elem2nodes, node_coords and elem2subdoms
    elem2nodes = numpy.delete(faces, p_elem2nodes_shifted[:-1])
    p_elem2nodes = p_elem2nodes_shifted - range(nelems + 1)
    node_coords = pvmesh.points
    if pvmesh.GetCellData().GetScalars() != None:
        elem2subdoms = numpy.array(pvmesh.GetCellData().GetScalars(), dtype=numpy.int64)
        #elem2subdoms = pvmesh.active_scalars  # Second option, but the line above is more flexible
    else:
        elem2subdoms = numpy.zeros(nelems, dtype=numpy.int64)

    elem_type = pvmesh.cast_to_unstructured_grid().celltypes

    return node_coords, p_elem2nodes, elem2nodes, elem_type, elem2subdoms


def normalize_coords(coords):
    max_coord, min_coord = numpy.max(coords, axis=0), numpy.min(coords, axis=0)
    node_coords = (coords-min_coord)/numpy.max((max_coord - min_coord)) 
    node_coords = node_coords - numpy.mean(node_coords, axis = 0)
    return node_coords


def sample_from_triangle_mesh_polydata(pvmesh, npoints ):
    faces_as_array = pvmesh.faces.reshape((pvmesh.n_faces, 4))[:, 1:] 
    tmesh = trimesh.Trimesh(pvmesh.points, faces_as_array) 
    point_cloud, face_index = trimesh.sample.sample_surface(tmesh, npoints)
    return point_cloud


def point_is_inside_to_mesh_polydata(pvmesh, point):
    end_point = numpy.array([10, 10, 10])
    intersections = pvmesh.ray_trace(point, end_point)
    cond = False if intersections[1].shape[0]%2==0 else True
    return cond


def rotation_matrix(theta, axis=[0, 0, 1]):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    axis = axis / numpy.sqrt(numpy.dot(axis, axis))
    a = numpy.cos(theta / 2.0)
    b, c, d = -axis * numpy.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_random_sample(data, sample_size):
    index = numpy.random.randint(0, data.shape[0], size=(sample_size, ))
    return jax.numpy.array(data[index]), index

def get_random_sample_boundary(bounds_list, bounds_values, sample_size):
    random_bound = []
    random_bound_values = []
    for i in range(len(bounds_list)):
        data, index = get_random_sample(bounds_list[i], sample_size[i])
        random_bound.append(data), random_bound_values.append(bounds_values[i][index])
    return random_bound, random_bound_values




def alpha_shape(data, alpha=0.0025, only_outer=True):
    import numpy as np
    from scipy.spatial import Delaunay
    points = data[:,:2]
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    contour_points = []
    for i, j in edges:
        contour_points.append([points[[i, j]], points[[i, j]]])
    contour_points = np.array(contour_points).reshape(-1,2)
    get_nearest_point = lambda xy: np.argmin(np.linalg.norm(xy-data[:,:2], axis = 1))
    contour_index =  np.array([get_nearest_point(contour_points[i]) for i in range(contour_points.shape[0])])
    #contour_points = data[contour_index]
    return  contour_index




def get_box_bounds2d(data, right, scale):
    def get_unitary_2dnormals_pcd(frontier_3d, k_neighbors=50):
        """
        Get the normals vector of an boundary 3d

        Parameters
        ----------
        frontier_3d : numpy.ndarray[[batch_size, 3]]
            -- coordinates (x, y, z) of the boundary

        Returns
        -------
        unit_normal_vectors : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y) of the normals
        """
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(frontier_3d)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
        pcd.orient_normals_to_align_with_direction(numpy.array([0.0, 0.0, -1.0]))

        normals = numpy.asarray(pcd.normals)
        normals2d = normals[:,:2]
        unit_normal_vectors = normals2d/numpy.linalg.norm(normals2d, axis=1)[:, numpy.newaxis]
        unit_normal_vectors = jax.numpy.array(unit_normal_vectors)

        return unit_normal_vectors

    def nearests_point_to_line(point, line_point1, line_point2, threshold=0.002):
        """Calculate the distance between a point and a line."""
        x0, y0 = point[:,0], point[:,1]
        x1, y1 = line_point1
        x2, y2 = line_point2
        distances = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / numpy.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return data[distances<threshold]

    top = nearests_point_to_line(data, [0,1], [1,1]) 

    left = nearests_point_to_line(data, [0,0], [0,1]) 

    down = nearests_point_to_line(data, [0,0], [1,0]) 

    bounds = [left[:,:2], top[:,:2], right[:,:2], down[:,:2]]

    new_data = numpy.concatenate((data, right))
    data_in_metters = numpy.column_stack((scale*new_data[:,0], scale*new_data[:,1], scale*new_data[:,2]))

    normals = [jax.numpy.column_stack((numpy.ones(left.shape[0]), jax.numpy.zeros(left.shape[0]))),
               jax.numpy.column_stack((numpy.zeros(top.shape[0]), jax.numpy.ones(top.shape[0]))),
               get_unitary_2dnormals_pcd(data_in_metters, k_neighbors=7)[-right.shape[0]:],
               jax.numpy.column_stack((numpy.zeros(down.shape[0]), -jax.numpy.ones(down.shape[0])))]
    
    return bounds, normals


def get_box_bounds3d_from_trimesh(mesh, config):
    def get_unitary_3dnormals_pcd(frontier_3d, k_neighbors=50):
        """
        Get the normals vector of an boundary 3d

        Parameters
        ----------
        frontier_3d : numpy.ndarray[[batch_size, 3]]
            -- coordinates (x, y, z) of the boundary

        Returns
        -------
        unit_normal_vectors : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y) of the normals
        """
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(frontier_3d)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
        pcd.orient_normals_to_align_with_direction(numpy.array([0.0, 0.0, -1.0]))
        normals = numpy.asarray(pcd.normals)
        unit_normal_vectors = normals/numpy.linalg.norm(normals, axis=1)[:, numpy.newaxis]
        unit_normal_vectors = jax.numpy.array(unit_normal_vectors)

        return unit_normal_vectors

    if config['rotate_xy'] != 0.0:
        angle = config['rotate_xy']
        M = rotation_matrix(angle)
        points = numpy.array(mesh.points)
        rotated_points = numpy.array([M @ points[i,:] for i in range(points.shape[0])])
        mesh.points = rotated_points


    n_surface_points = config['n_surface_points']
    n_inside_points = round(config['n_inside_points']**(1/3))
    n_cube_points = round((config['n_cube_points']/6)**(1/2))
    

    mesh.points = normalize_coords(coords = mesh.points)
    point_cloud = jax.numpy.array(sample_from_triangle_mesh_polydata(mesh, n_surface_points))
    point_cloud_normals = jax.numpy.array(get_unitary_3dnormals_pcd(point_cloud))


    x, y, z = numpy.meshgrid(numpy.linspace(-1, 1, n_inside_points+1, endpoint=False)[1:], numpy.linspace(-1, 1, n_inside_points+1, endpoint=False)[1:], numpy.linspace(-1, 1, n_inside_points+1, endpoint=False)[1:])
    grid_points = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    inside_surface_mask = numpy.array([point_is_inside_to_mesh_polydata(mesh, grid_points[i]) for  i in range(grid_points.shape[0])])
    domain_mask = numpy.logical_not(inside_surface_mask)
    domain = jax.numpy.array(grid_points[domain_mask])


    x, y, z = numpy.meshgrid(-1.0, numpy.linspace(-1, 1, n_cube_points), numpy.linspace(-1, 1, n_cube_points))
    front_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    front_face = jax.numpy.array(front_face)
    front_normals = -numpy.broadcast_to(numpy.array([1.0, 0.0, 0.0]), front_face.shape)
    front_normals = jax.numpy.array(front_normals)


    x, y, z = numpy.meshgrid(1.0, numpy.linspace(-1, 1, n_cube_points), numpy.linspace(-1, 1, n_cube_points))
    behind_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    behind_face = jax.numpy.array(behind_face)
    behind_normals = numpy.broadcast_to(numpy.array([1.0, 0.0, 0.0]), behind_face.shape)
    behind_normals = jax.numpy.array(behind_normals)


    x, y, z = numpy.meshgrid(numpy.linspace(-1, 1, n_cube_points), -1.0, numpy.linspace(-1, 1, n_cube_points))
    left_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    left_face = jax.numpy.array(left_face)
    left_normals = numpy.broadcast_to(numpy.array([0.0, -1.0, 0.0]), left_face.shape)
    left_normals = jax.numpy.array(left_normals)


    x, y, z = numpy.meshgrid(numpy.linspace(-1, 1, n_cube_points), 1.0, numpy.linspace(-1, 1, n_cube_points))
    right_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    right_face = jax.numpy.array(right_face)
    right_normals = numpy.broadcast_to(numpy.array([0.0, 1.0, 0.0]), right_face.shape)
    right_normals = jax.numpy.array(right_normals)


    x, y, z = numpy.meshgrid(numpy.linspace(-1, 1, n_cube_points), numpy.linspace(-1, 1, n_cube_points), 1.0)
    top_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    top_face = jax.numpy.array(top_face)
    top_normals = numpy.broadcast_to(numpy.array([0.0, 0.0, 1.0]), top_face.shape)
    top_normals = jax.numpy.array(top_normals)


    x, y, z = numpy.meshgrid(numpy.linspace(-1, 1, n_cube_points), numpy.linspace(-1, 1, n_cube_points), -1.0)
    down_face = numpy.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    top_face = jax.numpy.array(top_face)
    down_normals = numpy.broadcast_to(numpy.array([0.0, 0.0, -1.0]), down_face.shape)
    down_normals = jax.numpy.array(down_normals)


    bounds_list = [point_cloud, front_face, behind_face, left_face, right_face, top_face, down_face]
    bounds_list_normals = [point_cloud_normals, front_normals, behind_normals, left_normals, right_normals, top_normals, down_normals]

    return domain, bounds_list, bounds_list_normals
