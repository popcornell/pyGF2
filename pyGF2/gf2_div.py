import numpy as np
from pyGF2.generic_functions import strip_zeros


def gf2_div(dividend, divisor):
    """This function implements polynomial division over GF2.

    Given univariate polynomials ``dividend`` and ``divisor`` with coefficients in GF2,
    returns polynomials ``q`` and ``r``
    (quotient and remainder) such that ``f = q*g + r`` (operations are intended for polynomials in GF2).

    The input arrays are the coefficients (including any coefficients
    equal to zero) of the dividend and "denominator
    divisor polynomials, respectively.
    This function was created by heavy modification of numpy.polydiv.

    Parameters
    ----------
    dividend : ndarray (uint8 or bool)
        Dividend polynomial's coefficients.
    divisor : ndarray (uint8 or bool)
        Divisor polynomial's coefficients.

    Returns
    -------
    q : ndarray of uint8
        Quotient polynomial's coefficients.

    r : ndarray of uint8
        Quotient polynomial's coefficients.

    Notes
    -----
    Rightmost element in the arrays is the leading coefficient of the polynomial.
    In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
    in Sympy, for example, the leftmost element is the leading coefficient.


    Examples
    ========

    >>> x = np.array([1, 0, 1, 1, 1, 0, 1], dtype="uint8")
    >>> y = np.array([1, 1, 1], dtype="uint8")
    >>> gf2_div(x, y)
    (array([1, 1, 1, 1, 1], dtype=uint8), array([], dtype=uint8))

    """

    N = len(dividend) - 1
    D = len(divisor) - 1

    if dividend[N] == 0 or divisor[D] == 0:
        dividend, divisor = strip_zeros(dividend), strip_zeros(divisor)

    if not divisor.any():  # if every element is zero
        raise ZeroDivisionError("polynomial division")
    elif D > N:
        q = np.array([])
        return q, dividend

    else:
        u = dividend.astype("uint8")
        v = divisor.astype("uint8")

        m = len(u) - 1
        n = len(v) - 1
        scale = v[n].astype("uint8")
        q = np.zeros((max(m - n + 1, 1),), u.dtype)
        r = u.astype(u.dtype)

        for k in range(0, m - n + 1):
            d = scale and r[m - k].astype("uint8")
            q[-1 - k] = d
            r[m - k - n:m - k + 1] = np.logical_xor(r[m - k - n:m - k + 1], np.logical_and(d, v))

        r = strip_zeros(r)

    return q, r
