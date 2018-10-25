import numpy as np
from generic_functions import strip_zeros

def gf2_mul(a, b):
    """Multiply polynomials in GF(2)

    NOTE: rightmost array element is
          the leading coefficient

    Parameters
    ----------
    a : ndarray (uint8 or bool) or list
        Multiplicand polynomial's coefficients.
    b : ndarray (uint8 or bool) or list
        Multiplier polynomial's coefficients.
    Returns
    -------
    q : ndarray of uint8
        Resulting polynomial's coefficients.

    Examples
    ========

    >>> a = np.array([1,0,1], dtype="uint8")
    >>> b = np.array([1,1,1], dtype="uint8")
    >>> gf2_mul(a,b)
    array([1, 1, 0, 1, 1], dtype=uint8)
"""

    #cast here ?#TODO also use bools ?

    fsize = len(a) + len(b) - 1

    fsize = 2*fsize #** np.ceil(np.log2(fsize)).astype(int) #TODO better to use powers of 2 ?

    fslice = slice(0, fsize//2)

    ta = np.fft.fft(a, fsize)
    tb = np.fft.fft(b, fsize)

    res = np.fft.ifft(ta*tb)[fslice].copy()

    k = np.mod(np.rint(np.real(res)), 2).astype('uint8')

    return strip_zeros(k) #TODO really strip ?