import numpy as np


def strip_zeros(a):
    """Strip un-necessary leading (rightmost) zeroes
    from a polynomial"""

    return np.trim_zeros(a, trim='b')


def padding(a, dim):

    return np.pad(a, (dim-len(a), 0), 'constant', constant_values=(0))


def to_same_dim(a, b):

    if len(a) > len(b):
       return a, padding(b, len(a))

    elif len(a) < len(b):
        return padding(a, len(b)), b

    else:
        return a, b


def zeros(dim):
    """Returns dim coefficients for -1 degree polynomial"""

    return np.zeros(dim, dtype='uint8')


def zerodegree_pol(dim):
    """Returns dim coefficients for a zero degree polynomial"""

    out = zeros(dim)
    out[0] = 1

    return out
