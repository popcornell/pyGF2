import numpy as np
from pyGF2.generic_functions import strip_zeros
from pyGF2.gf2_div import gf2_div
from pyGF2.gf2_add import gf2_add


def gf2_inv(f, g):
    """ Given a polynomial ``f`` and an irriducible polynomial ``g`` both in GF(p)[x], computes the
        multiplicative inverse ``out``, such that f*out == 1 mod(g) (All operations are intended in GF(p)[x]).

        Parameters
        ----------
        f : ndarray (uint8 or bool) or list
            Input polynomial's coefficients.
        g : ndarray (uint8 or bool) or list
            Irriducible polynomial's coefficients.

        Returns
        -------
        out : ndarray of uint8
            Multiplicative inverse polynomial's coefficients.

        Notes
        -----
        Rightmost element in the arrays is the leading coefficient of the polynomial.
        In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
        in Sympy, for example, the leftmost element is the leading coefficient.


        Examples
        ========

        >>> x = np.array([1, 1, 0, 1], dtype="uint8")
        >>> y = np.array([1, 0, 0, 0, 0, 1], dtype="uint8")
        >>> gf2_inv(x,y)
        array([0, 1, 1, 1], dtype=uint8)

        """

    out = gf2_xgcd(f, g)[0]

    return out


def gf2_xgcd(b, a):
    """Perform Extended Euclidean Algorithm over GF2

    Given polynomials ``b`` and ``a`` in ``GF(p)[x]``, computes polynomials
    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*b + t*a = h``.
    The typical application of EEA is solving polynomial diophantine equations and findining multiplicative inverse.


    Parameters
    ----------
    b : ndarray (uint8 or bool) or list
        b polynomial's coefficients.
    a : ndarray (uint8 or bool) or list
        a polynomial's coefficients.
    Returns
    -------
    y2 : ndarray of uint8
         s polynomial's coefficients.
    x2 : ndarray of uint8
         t polynomial's coefficients.
    b : ndarray of uint8
        h polynomial's coefficients.

    Notes
    -----
    Rightmost element in the arrays is the leading coefficient of the polynomial.
    In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
    in Sympy, for example, the leftmost element is the leading coefficient.

    Examples
    ========

    >>> x = np.array([1, 1, 1, 1, 1, 0, 1, 0, 1], dtype="uint8")
    >>> y = np.array([1, 0, 1], dtype="uint8")
    >>> gf2_xgcd(x,y)
    (array([0, 1, 1, 1], dtype=uint8),
     array([1, 1], dtype=uint8),
     array([1], dtype=uint8))

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


def mul(a, b):
    """Performs polynomial multiplication over GF2.

       Parameters
       ----------
       b : ndarray (uint8 or bool) or list
           Multiplicand polynomial's coefficients.
       a : ndarray (uint8 or bool) or list
           Multiplier polynomial's coefficients.
       Returns
       -------
       out : ndarray of uint8


       Notes
       -----
       This function performs exactly the same operation as gf2_mul but here instead of the fft, convolution
       in time domain is used. This is because this function must be used multiple times in gf2_xgcd and performing the
       fft in that instance introduced significant overhead.
    """

    out = np.mod(np.convolve(a, b), 2).astype("uint8")

    return strip_zeros(out)
