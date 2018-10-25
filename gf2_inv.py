import numpy as np
from generic_functions import  strip_zeros
from gf2_long_div import gf2_div
from gf2_add import gf2_add




def mul(a, b):

    out = np.mod(np.convolve(a,b), 2).astype("uint8")

    return strip_zeros(out)


def gf2_inv(f, g):

    out = xgcd(f, g)[0]

    return out


def xgcd(b, a):
    db = max(len(b), len(a))

    x1 = np.array([1], dtype="uint8")#unitary_pol(db)
    y0 = np.array([1], dtype="uint8")#unitary_pol(db)

    x0 = np.array([], dtype="uint8")#zeros(db)
    y1 = np.array([], dtype="uint8")#zeros(db)

    while True:

        q, r = gf2_div(b, a)

        b = a

        if not r.any():
            break

        a = r

        if not (q.any() and x1.any()):  # if q is zeros or x1 is zeros
            x2 = x0
        elif not x0.any():  # if x0 is all zeros
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



