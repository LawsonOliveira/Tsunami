import numpy


def get_normalized_norm_L1(A):
    normalization_coef = int(sum([s for s in A.shape]))
    norm_L1 = numpy.sum(numpy.abs(A))
    return norm_L1 / normalization_coef


def get_normalized_norm_L2(A):
    normalization_coef = int(sum([s for s in A.shape]))
    adj_A = numpy.conjugate(A.T)
    # abs to ensure Python return a real number
    norm_L2 = numpy.abs(numpy.trace(numpy.dot(adj_A, A)))
    return norm_L2 / normalization_coef
