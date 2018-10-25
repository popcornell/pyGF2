import numpy as np
import numba

@numba.jit(forceobj=True)
def strip_zeros(a):
    return np.trim_zeros(a, trim='f')

@numba.jit(forceobj=True)
def padding(a, dim):


    return np.pad(a, (dim-len(a), 0), 'constant', constant_values=(0))

@numba.jit(forceobj=True)
def to_same_dim(a, b):

    if len(a) > len(b):
       return a, padding(b, len(a))

    elif len(a) < len(b):
        return padding(a, len(b)), b

    else:
        return a, b

@numba.jit(forceobj=True)
def zeros(dim):

    return np.zeros(dim, dtype='uint8')

@numba.jit(forceobj=True)
def unitary_pol(dim):

    out = zeros(dim-1)
    out = np.insert(out, dim-1, values=1)

    return out
