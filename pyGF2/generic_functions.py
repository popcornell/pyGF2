import numpy as np


def strip_zeros(a):
    """Strip un-necessary leading (rightmost) zeroes
    from a polynomial"""

    return np.trim_zeros(a, trim='b')


def check_type(a, b):
    """Type check and force cast to uint8 ndarray

    Notes
    -----
    Ideally for best performance one should always use uint8 or bool when using this library.

    """

    if isinstance(a, np.ndarray):
        a = np.array(a, dtype="uint8")
    if isinstance(b, np.ndarray):
        b = np.array(b, dtype="uint8")

    if a.dtype is not "uint8":
        a = a.astype("uint8")

    if b.dtype is not "uint8":
        b = b.astype("uint8")

    return a, b


def padding(a, dim):
    """Zero-pad input array a a to length dim, zeroes are appended at the right"""

    return np.pad(a, (0, dim-len(a)), 'constant', constant_values=(0))


def to_same_dim(a, b):
    """Given two arrays a and b returns the two arrays with the shorter zero-padded to have
    the same dimension of the longer. The arrays are padded with zeroes appended to the right.
    """

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
