import numpy
import jax
import matplotlib.pyplot
    
def gen_square(domain_bounds, n_inside, n_bound):
    ### Inside data
    x = jax.numpy.linspace(domain_bounds[0,0], domain_bounds[0,1], int(jax.numpy.sqrt(n_inside))+1, endpoint=False)[1:]
    y = jax.numpy.linspace(domain_bounds[1,0], domain_bounds[1,1], int(jax.numpy.sqrt(n_inside))+1, endpoint=False)[1:]
    x, y = jax.numpy.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    inside_points = jax.numpy.column_stack((x, y))

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

    return inside_points, boundary_points


def plot_2d_domain(inside, boundaries, img_path):

    fig, ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(14.0, 14.0)
    ax.set_title('Spatial domain - 2d')
    matplotlib.pyplot.scatter(boundaries[:,0], boundaries[:,1], color='red', s=5)
    matplotlib.pyplot.scatter(inside[:,0], inside[:,1], color='green', s=5)
    __ = ax.legend(['Boundary', 'Inside'])
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return



def plot_loss_history(losses, path, legend = ['loss_sum','residual','boundary']):
    # losses = [los_history, loss_res, loss_bound]
    nloss = len(legend)
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    fig.set_size_inches(18.0, 14.0)

    for loss_values in losses:
        __ = ax.plot(numpy.log10(loss_values))
    xlabel = ax.set_xlabel(r'${\rm Epoch}$')
    ylabel = ax.set_ylabel(r'$\log_{10}{\rm (loss)}$')
    title = ax.set_title(r'${\rm Training}$')
    ax.legend(legend)
    matplotlib.pyplot.savefig(path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()

    return 


def plot_2d_colormesh(params, domain_bound, function, img_path, title_name, color='rainbow', npoints = 300):

    x, y = numpy.meshgrid(numpy.linspace(domain_bound[0,0], domain_bound[0,1], npoints),numpy.linspace(domain_bound[1,0], domain_bound[1,1], npoints))
    values = numpy.zeros((npoints, npoints))

    fig, ax = matplotlib.pyplot.subplots(facecolor='white')
    fig.set_size_inches(18, 7.2)
    fig.tight_layout(h_pad=3)
    title = ax.set_title(title_name)

    for i in range(npoints):
        values[i,:] = function(params, x[i,:], y[i,:])

    graph = matplotlib.pyplot.pcolormesh(x, y, values, cmap = color)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig(img_path, facecolor='white', bbox_inches = 'tight')
    matplotlib.pyplot.show()  

    return numpy.mean(values)