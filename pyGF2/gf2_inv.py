import numpy as np
from pyGF2.generic_functions import strip_zeros
from pyGF2.gf2_long_div import gf2_div
from pyGF2.gf2_add import gf2_add





def mul(a, b):
    """Perform polynomial multiplication over GF2"""

    out = np.mod(np.convolve(a,b), 2).astype("uint8")

    return strip_zeros(out)



def gf2_inv(f, g):

    out = gf2_xgcd(f, g)[0]

    return out


def gf2_xgcd(b, a):
    """Perform Extended Euclidean Algorithm over GF2

    Given polynomials ``b`` and ``a`` in ``GF(p)[x]``, computes polynomials
    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*b + t*a = h``.
    The typical application of EEA is solving polynomial diophantine equations.

    NOTE: rightmost array element is
          the leading coefficient

    Parameters
    ----------
    b : ndarray (uint8 or bool) or list
        Multiplicand polynomial's coefficients.
    a : ndarray (uint8 or bool) or list
        Multiplier polynomial's coefficients.
    Returns
    -------
    y2 : ndarray of uint8
        Resulting polynomial's coefficients.
    x2 : ndarray of uint8
        Resulting polynomial's coefficients.
    b : ndarray of uint8
        Resulting polynomial's coefficients.

    Examples
    ========

    >>> a = np.array([1,0,1], dtype="uint8")
    >>> b = np.array([1,1,1], dtype="uint8")
    >>> gf2_mul(a,b)
    array([1, 1, 0, 1, 1], dtype=uint8)


    """



    x1 = np.array([1], dtype="uint8")
    y0 = np.array([1], dtype="uint8")

    x0 = np.array([], dtype="uint8")
    y1 = np.array([], dtype="uint8")

    while True:

        q, r = gf2_div(b, a)

        b = a

        if not r.any():
            break

        a = r

        if not (q.any() and x1.any()):  # if q is zero or x1 is zero
            x2 = x0
        elif not x0.any():  # if x0 is zero
            x2 = mul(x1, q)
        else:
            mulres = mul(x1, q)

            x2 = gf2_add(x0, mulres)

        if not (q.any() and y1.any()):
            y2 = y0
        elif not y0.any():
            y2 = mul(y1, q)
        else:
            mulres = mul(y1, q)

            y2 = gf2_add(y0, mulres)

        # update
        y0 = y1
        x0 = x1
        y1 = y2
        x1 = x2

    return y2, x2, b



