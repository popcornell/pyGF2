import numba

import numpy as np
from generic_functions_2 import strip_zeros, to_same_dim, unitary_pol, zeros


'''
POLYNOMIAL MULTIPLICATION
'''

@numba.jit(forceobj=True)
def poly_mul(a, b):

    k = np.mod(np.convolve(a, b), 2)

    return k


'''
GF2 ADDITION/SUBTRACTION (XOR)
'''


@numba.jit(forceobj=True)
def gf2_add(a, b):
    a, b = to_same_dim(a, b)

    return np.logical_xor(a, b, dtype='uint8').astype('uint8')


'''MULTIPLICATION IN Z mod x^p+1'''


@numba.jit(forceobj=True)
def z_mul(a, b):

    return np.rint(np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))).astype('uint16')  # TODO ? type ?


'''MULTIPLICATION IN GF(2) mod x^p+1'''


@numba.jit(forceobj=True)
def gf2_mul(a, b):
    return np.mod(z_mul(a, b), 2).astype('uint8')



#@numba.jit(forceobj=True)
def gf2_div(f, g):
    df = len(f) - 1
    dg = len(g) - 1

    if not g.any():  # if every element is zero
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        z = np.zeros(dg, dtype='uint8')
        return z, f  # TODO not sure possible bug df or dh ?

    else:
        f, g = np.copy(f), np.copy(g)  # copying to avoid modification of orig arrays

        a, b = numba_div(strip_zeros(f), strip_zeros(g), df, dg)

        b = strip_zeros(b)  # nb stripped !!!!

        return a, b


@numba.jit(nopython=True, cache=True)
def numba_div(f, g, df, dg):
    # remove leading zeros zeros in advance
    h, dq, dr = f, df - dg, dg - 1

    for i in range(0, df + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff = coeff ^ (h[i + j - dg] and g[dg - j])

        if i <= dq:
            coeff = coeff

        h[i] = coeff

    return h[:dq + 1], h[dq + 1:]  # not stripped


'''EXTENDED EUCLIDEAN ALGORITHM IN GF(2)'''


# TODO numba accel
def xgcd(b, a):
    db = max(len(b), len(a))

    x1 = unitary_pol(db)
    y0 = unitary_pol(db)

    x0 = zeros(db)
    y1 = zeros(db)

    while True:

        q, r = gf2_div(b, a)

        b = a

        if not r.any():
            break

        a = r

        if not (q.any() and x1.any()):  # if q is zeros or x1 is zeros
            x2 = x0
        elif not x0.any():  # if x0 is all zeros
            x2 = poly_mul(x1, q)
        else:
            mulres = poly_mul(x1, q)

            x2 = gf2_add(x0, mulres)

        if not (q.any() and y1.any()):
            y2 = y0
        elif not y0.any():
            y2 = poly_mul(y1, q)
        else:
            mulres = poly_mul(y1, q)

            y2 = gf2_add(y0, mulres)

        # update
        y0 = y1
        x0 = x1
        y1 = y2
        x1 = x2

    return y2, x2, b


#@numba.jit(forceobj=True)
def gf2_inv(f, g):
    f = np.copy(f)
    g = np.copy(g)

    f = np.flip(f, axis=0)
    g = np.flip(g, axis=0)

    out = xgcd(strip_zeros(f), g)[0]

    out = np.flip(out, axis=0)

    return out


@numba.jit(forceobj=True)
def circtranspose(a):
    at = np.copy(a)

    at[1:] = np.flip(at[1:], axis=0)

    return at

