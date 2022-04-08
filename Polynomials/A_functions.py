import numpy as np

"""
Useful functions to use for the definition of A
"""


def distance(x, x_i):
    # Returns the distance between two vectors x and x_i
    return np.linalg.norm(x - x_i)


def dist_min(x, x_list):
    # Returns the minimal distance from x to a list of points
    # x_list has to be an iterable
    min_dist = np.inf
    for y in x_list:
        if y.all() != x.all():
            dist = distance(x, y)
            min_dist = min(min_dist, dist)
    return min_dist


"""
Different psi functions used to compute A (exp or tanh variants)
"""


def psi_exp(x, x_i, l_square, width):
    dist_square = distance(x, x_i)**2
    if dist_square < l_square:
        return np.exp(-1/(width*(l_square - dist_square)))*np.exp(1/(width*l_square))
    return 0


def psi_tanh(x, x_i, l_square, width):
    return 0.5*(1-np.tanh(distance(x, x_i)/l_square**0.5)**2)


"""
Definition of the function A
"""


class A2D():
    """
    Generates the function A on a boundary sample and computes values
    of the function in the considered space
    """

    def __init__(self, sample, sample_values, psi_used, width=4):
        """
        Defines the variables used to set A
        - sample: list of coordinates of boundary points
        - sample_values: list of values of g evaluated at the boundary points sample
        - psi_used: psi function used for the boundary (psi_exp or psi_tanh)
        - width: coefficient used to smooth the function between samples
        """
        self.sample = sample
        self.sample_values = sample_values
        l = min([dist_min(sample[i], sample) for i in range(len(sample))])
        self.l_square = l**2
        self.width = width
        self.psi = psi_used

    def evaluate(self, X):
        """
        Returns the value of A at coordinates X
        - X: vector of coordinates of a point in 2D. X.shape == (2,) or len(X) == 2
        """
        res = 0
        for i in range(len(self.sample)):
            res += self.sample_values[i]*self.psi(
                X,
                self.sample[i],
                self.l_square,
                self.width
            )
        return res
