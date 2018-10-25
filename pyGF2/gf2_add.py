import numpy as np
from pyGF2.generic_functions import strip_zeros



def xor(a, b):

    return np.logical_xor(a, b, dtype='uint8').astype("uint8")


def check_type(a, b):

    if isinstance(a, np.ndarray):
        a = np.array(a, dtype="uint8")
    if isinstance(b, np.ndarray):
        b = np.array(b, dtype="uint8")

    if a.dtype is not "uint8":
        a = a.astype("uint8")

    if b.dtype is not "uint8":
        b = b.astype("uint8")

    return a, b


def gf2_add(a, b):

    """Add polynomials in GF(2)

    NOTE: rightmost array element is
          the leading coefficient

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


