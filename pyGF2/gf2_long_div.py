import numpy as np
from pyGF2.generic_functions import strip_zeros


def gf2_div(dividend, divisor):
    '''This function implements long division over GF2.'''

    N = len(dividend) -1
    D = len(divisor) - 1

    if dividend[N] == 0 or divisor[D] == 0:
        #ValueError("remember to strip zeros")
        dividend, divisor = strip_zeros(dividend), strip_zeros(divisor)

    if not divisor.any():  # if every element is zero
        raise ZeroDivisionError("polynomial division")
    elif D > N:
        q = np.array([]) #np.zeros(1, dtype='uint8') #grado -1
        return q, dividend  # TODO not sure possible bug df or dh ?

    else:
        u = dividend.astype("uint8")
        v = divisor.astype("uint8")

        # w has the common type
        m = len(u) - 1
        n = len(v) - 1
        scale =  v[n].astype("uint8")
        q = np.zeros((max(m - n + 1, 1),), u.dtype)
        r = u.astype(u.dtype)

        for k in range(0, m-n+1):
            d = scale and r[m-k].astype("uint8")
            q[-1-k] = d
            r[m - k - n:m - k + 1] = np.logical_xor(r[m-k-n:m-k+1], np.logical_and(d , v))
        #while np.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        #    r = r[1:]
        r = strip_zeros(r)

    return q, r

    '''

        # deconvolution
        w = dividend[0] ^ divisor[0]  # xor
        q = np.zeros((max(N - D + 1, 1),), w.dtype)
        r = dividend.astype(w.dtype)
        for k in range(0, N - D + 1):
            d = r[k]
            q[k] = d
            r[k:k + D + 1] = r[k:k + D + 1] ^  ( d and divisor )
        while np.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
            r = r[1:]

        return q, r

        

        input = np.zeros(N - D + 1, float)
        input[0] = 1
        q = lfilter(dividend, divisor, input).astype("uint8")
        r = gf2_add(dividend , gf2_mul(divisor, q))

        

        fsize = max(N, D)

        fslice = slice(0, fsize)

        ta = np.fft.fft(dividend, fsize)
        tb = np.fft.fft(divisor, fsize)

        ta_r, tb_r = np.real(ta), np.real(tb)
        ta_c, tb_c = np.imag(ta[1:]), np.imag(tb[1:])

        tq_r, tr_r = np.divmod(ta_r, tb_r)
        tq_c, tr_c = np.divmod(ta_c, tb_c)

        tq = tq_r + 1j * (np.insert(tq_c, 0, values=0))
        tr = tr_r + 1j * (np.insert(tr_c, 0, values=0))

        q = np.mod(np.rint(np.real(np.fft.ifft(tq, len(dividend))[fslice].copy())), 2).astype('uint8')
        r = np.mod(np.rint(np.real(np.fft.ifft(tr, len(dividend))[fslice].copy())), 2).astype('uint8')

'''




