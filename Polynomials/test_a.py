from A_functions import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def Z_calc(X, Y, A_fct):
    # Calculates the value of the function A on a mesh given by X and Y coordinates lists
    mesh = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            mesh[i][j] = A_fct.evaluate([X[i][j], Y[i][j]])
    return mesh

if __name__=='__main__':
    """
    Test on a circular boundary, uniform sampling, following a sinusoid boundary condition
    """
    list_theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
    sample = np.array([np.array([np.cos(list_theta[i]), np.sin(list_theta[i])]) for i in range(len(list_theta))])
    sample_values = np.sin(list_theta)
    chosen_A = psi_exp #psi_exp or psi_tanh

    #Computing l:
    l = min([dist_min(sample[i], sample) for i in range(len(sample))])
    l_square = l**2
    width = 1

    #Defining A
    A_fct = A(sample, sample_values, chosen_A, width)

    """
    Trying to break the function
    """
    def boundary(sample):
        results = np.zeros(len(sample))

    """
    Plotting the resulting A
    """
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = Z_calc(X, Y, A_fct)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.scatter3D(sample[:,0], sample[:,1], sample_values, cmap='Greens')
    plt.show()