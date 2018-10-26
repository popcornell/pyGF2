import numpy as np
from pyGF2.generic_functions import strip_zeros, check_type



def xor(a, b):
    """Computes the element-wise XOR of two ndarrays"""

    return np.logical_xor(a, b, dtype='uint8').astype("uint8")




def gf2_add(a, b):

    """Add two polynomials in GF(p)[x]

    Parameters
    ----------
    a : ndarray (uint8 or uint8) or list
        Addend polynomial's coefficients.
    b : ndarray (uint8 or uint8) or list
        Addend polynomial's coefficients.
    Returns
    -------
    q : ndarray of uint8
        Resulting polynomial's coefficients.


    Notes
    -----
    Rightmost element in the arrays is the leading coefficient of the polynomial.
    In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
    in Sympy, for example, the leftmost element is the leading coefficient.

    Examples
    ========

    >>> a = np.array([1,0,1], dtype="uint8")
    >>> b = np.array([1,1], dtype="uint8")
    >>> gf2_add(a,b)
    array([0, 1, 1], dtype=uint8)
"""
    a, b = check_type(a, b)

    a, b = strip_zeros(a), strip_zeros(b)

    N = len(a)

    D = len(b)

    if N == D:
        res = xor(a, b)

    elif N > D:

        res = np.concatenate((xor(a[:D], b), a[D:]))

    else:

        res = np.concatenate((xor(a, b[:N]), b[N:]))

    return strip_zeros(res)


