import numpy
import jax
import matplotlib.pyplot


def get_sub_domain(data, residuals):
    size = data.shape[0]//2
    dim = data.shape[1]
    residuals_index = numpy.argpartition(numpy.array(residuals).flatten(), -size)[-size:]
    return data[residuals_index].reshape(size, dim)


def scheduler_wave_lenght_cossinus_decay_warmup(k_max, index, maximum_num_epochs):
    warmup_epochs = maximum_num_epochs//5
    if index < warmup_epochs:
        new_k = k_max * index / warmup_epochs
    else:
        cosine_decay_warmup = (1 + numpy.cos((index - warmup_epochs) / (maximum_num_epochs - warmup_epochs) * numpy.pi)) / 2
        new_k = k_max * (1-cosine_decay_warmup)
        if index == maximum_num_epochs:
            new_k = k_max
    return new_k



def scheduler_wave_lenght_cossinus_decay(k_max, index, maximum_num_epochs):
    if index >= maximum_num_epochs//2:
        new_k = k_max
    else:
        cosine_decay = 0.5 * (1 + numpy.cos(numpy.pi * index / (maximum_num_epochs//2)))
        new_k = k_max * (1-cosine_decay)
    return  new_k
    
    
def scheduler_wave_lenght_linear_decay(k_max, index, maximum_num_epochs):
    if index >= maximum_num_epochs//2:
        new_k = k_max
    else:
        linear_decay = index / (maximum_num_epochs//2)
        new_k = k_max * (1-linear_decay)
    return  new_k
    
    
def GALE(scheduler, data, solver, params, config, iter):
        
    wave_number_ibatch = scheduler(config['wave_number'], iter, config['maximum_num_epochs'])
    solver.update_wave_length(wave_number_ibatch)
    residuals = abs(solver.compute_residual(params, data))
    sub_dom = get_sub_domain(data, residuals)

    return sub_dom


def plot_scheduler(scheduler, k_max, maximum_num_epochs, path=None):

    epochs = numpy.arange(maximum_num_epochs)
    new_k = [scheduler(k_max, epoch, maximum_num_epochs) for epoch in epochs]

    matplotlib.pyplot.plot(epochs, new_k)
    matplotlib.pyplot.title("Cosine Decay Schedule")
    matplotlib.pyplot.xlabel("Epochs")
    matplotlib.pyplot.ylabel("Wave length Multiplier")
    
    if path!=None:
        matplotlib.pyplot.savefig(path, facecolor='white', bbox_inches = 'tight')
        return
    matplotlib.pyplot.show()

    return 

def plot_clusters(data, residuals, path=None):
    residuals_mean = numpy.mean(residuals)

    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    graph = ax.scatter(data[:, 0], data[:, 1], c=residuals, cmap='cool', s=5)
    ax.scatter(data[residuals.flatten()>residuals_mean][:, 0], data[residuals.flatten()>residuals_mean][:, 1], color='black', s=0.1)
    matplotlib.pyplot.colorbar(graph)
    if path!=None:
        matplotlib.pyplot.savefig(path, facecolor='white', bbox_inches = 'tight')
        return
    matplotlib.pyplot.show()
    return 
