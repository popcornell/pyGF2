import numpy as np
from pyGF2.generic_functions import strip_zeros

def gf2_mul(a, b):
    """Multiply polynomials in GF(2), FFT instead of convolution in time domain is used
       to speed up computation significantly.

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

    fsize = len(a) + len(b) - 1

    fsize = 2**np.ceil(np.log2(fsize)).astype(int) #use nearest power of two much faster

    fslice = slice(0, fsize)

    ta = np.fft.fft(a, fsize)
    tb = np.fft.fft(b, fsize)

    res = np.fft.ifft(ta*tb)[fslice].copy()

    k = np.mod(np.rint(np.real(res)), 2).astype('uint8')

    return strip_zeros(k)