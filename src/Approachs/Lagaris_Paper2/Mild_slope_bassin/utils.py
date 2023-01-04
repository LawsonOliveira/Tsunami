import numpy
import open3d
import jax
import functools
import matplotlib.pyplot

def get_flow_arrows(data, flow_direction):
    # Flow
    xdatamin, xdatamax = numpy.min(data[:,0]), numpy.max(data[:,0])
    ydatamin, ydatamax = numpy.min(data[:,1]), numpy.max(data[:,1])
    zdatamin, zdatamax = numpy.min(data[:,2]), 0.0

    xflow = numpy.linspace(xdatamin, xdatamax, int((1-flow_direction[0])*5) + 1)
    yflow = numpy.linspace(ydatamin, ydatamax, int((1-flow_direction[1])*5) + 1)
    zflow = numpy.linspace(zdatamin, zdatamax, int((1-flow_direction[2])*5) + 1)

    xflow, yflow, zflow = numpy.meshgrid(xflow, yflow, zflow)
    xflow, yflow, zflow = xflow.flatten(), yflow.flatten(), zflow.flatten()
    flow = numpy.column_stack((xflow, yflow, zflow))

    return flow


def plot_3d_domain_and_arrows(data, flow_direction, img_path):
    # Flow
    flow = get_flow_arrows(data, flow_direction)

    # Plot
    fig = matplotlib.pyplot.figure()
    fig.set_size_inches(14.0, 14.0)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title('Spatial domain - 3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], color='green', s=1)
    ax.quiver(flow[:,0],flow[:,1],flow[:,2],flow_direction[0],flow_direction[1],flow_direction[2],length=0.1,color='black',normalize=True)
    ax.view_init(45,225)
    __ = ax.legend(['Ground','Flow'])
    fig.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return
    
def plot_normals_2d(points, normals, img_path):
    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(14.0, 14.0)
    matplotlib.pyplot.quiver(points[:,0], points[:,1], normals[:,0], normals[:,1])
    fig.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()
    
    return 

def plot_2d_domain(inside, boundaries, img_path):

    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(14.0, 14.0)
    ax.set_title('Spatial domain - 2d')
    matplotlib.pyplot.scatter(boundaries[:,0], boundaries[:,1], color='red', s=1)
    matplotlib.pyplot.scatter(inside[:,0], inside[:,1], color='green', s=0.01)
    __ = ax.legend(['Boundary', 'Inside'])
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return


def get_unitary_normals_pcd3d(frontier_3d, k_neighbors=50):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(frontier_3d)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
    pcd.orient_normals_to_align_with_direction(numpy.array([0.0, 0.0, -1.0]))
    normals = numpy.asarray(pcd.normals)
    normals2d = normals[:,:2]
    unit_normal_vectors = normals2d/numpy.linalg.norm(normals2d, axis=1)[:, numpy.newaxis]
    
    return jax.numpy.array(unit_normal_vectors)


def get_unitary_normals_regular2d(bound_points):
    point_nplus1 = 2*bound_points[-1]-bound_points[-2]
    extended_bound_points = jax.numpy.concatenate((bound_points, point_nplus1.reshape((1,2))))

    deltax_deltay = extended_bound_points[1:]-bound_points
    normal_vector = jax.numpy.column_stack((-deltax_deltay[:,1], deltax_deltay[:,0]))
    unitary_normal_vector = normal_vector/jax.numpy.linalg.norm(normal_vector, axis=1)[:, jax.numpy.newaxis]

    return unitary_normal_vector



def plot_loss_history(losses, path, legend = ['loss_sum','residual','boundary']):
    # losses = [los_history, loss_res, loss_bound]
    nloss = len(legend)
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    fig.set_size_inches(18.0, 14.0)

    for loss_values in losses:
        __ = ax.plot(numpy.log10(loss_values))
    ax.set_xlabel(r'${\rm Epoch}$')
    ax.set_ylabel(r'$\log_{10}{\rm (loss)}$')
    ax.set_title(r'${\rm Training}$')
    ax.legend(legend)
    matplotlib.pyplot.savefig(path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()

    return 



def plot_2d_animation(data, params, function, end_time, frames, path, scale = 1):
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
    title = ax.set_title('Displacement, time step = 1')

    values = function(params, x, y, jax.numpy.zeros(npoints))
    min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))

    graph = matplotlib.pyplot.scatter(x, y, c=scale*values, s=5, cmap = 'rainbow', vmin = min_values, vmax = max_values)
    matplotlib.pyplot.clim(vmin = min_values, vmax = max_values)
    matplotlib.pyplot.colorbar(graph, ax=ax)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames = frames,  blit = False)
    ani.save(path, writer='pillow', fps = 10)
    matplotlib.pyplot.show()  

    return 


def plot_3d_animation_and_arrows(data, params, function, flow_direction, end_time, frames, path, scale = 1):
    npoints = data.shape[0]
    step = end_time/frames
    x, y, z = data[:,0], data[:,1], data[:,2]
    t = jax.numpy.ones(npoints)

    def update_graph(num):
        ax.cla()

        
        print("Animating: {} out of {}".format(num+1, frames), end='\r')
        
        time = t*step*(num+1)
        ax.scatter(x, y, z, color='green', s=10)
        ax.quiver(flow[:,0],flow[:,1],flow[:,2],1,0,0,length=0.1,color='black',normalize=True)

        values = function(params, x, y, time)
        values = function(params, x, y, time)
        graph = ax.scatter(x, y, scale*values, c=scale*values, s=10, cmap = 'rainbow')
        ax.set_zlim([-1, max_values])
        ax.set_title('Displacement, time step = {}'.format(num+1))

        graph.set_clim(vmin = min_values, vmax = max_values)
        __ = ax.legend(['Shoal', "Flow direction", "Wave"])
        return fig,

    fig = matplotlib.pyplot.figure(facecolor='white')
    fig.set_size_inches(14.0, 14.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,225)

    ax.scatter(x, y, z, color='green', s=10)
    flow = get_flow_arrows(data, flow_direction)

    ax.quiver(flow[:,0],flow[:,1],flow[:,2],1,0,0,length=0.1,color='black',normalize=True)
    values = function(params, x, y, jax.numpy.zeros(npoints))
    min_values, max_values = -scale*jax.numpy.max(abs(values)), scale*jax.numpy.max(abs(values))
    graph = ax.scatter(x, y, scale*values, c=scale*values, s=10, cmap = 'rainbow')
    ax.set_zlim([-1, max_values])

    graph.set_clim(vmin = min_values, vmax = max_values)
    __ = ax.legend(['Flow', "Wave"])
    matplotlib.pyplot.colorbar(graph, ax=ax)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames = frames,  blit = False)
    ani.save(path, writer='pillow', fps = 10)
    matplotlib.pyplot.show()  

    return 
