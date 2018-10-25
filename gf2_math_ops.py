import numba

import numpy as np
from generic_functions import strip_zeros, to_same_dim, unitary_pol, zeros, padding
from scipy import signal


'''
POLYNOMIAL MULTIPLICATION
'''
@numba.jit()
def poly_mul(a, b):

    #fsize = len(a) + len(b) -1

    #fsize = 2*fsize #** np.ceil(np.log2(fsize)).astype(int)

    #fslice = slice(0, fsize//2)

    #ta = np.fft.fft(a, fsize)
    #tb = np.fft.fft(b, fsize)

    #res = np.fft.ifft(ta*tb)[fslice].copy()

    #k = np.mod(np.rint(np.real(res)), 2).astype('uint8')

    k = np.mod(np.convolve(a, b), 2)

    return k

'''
GF2 ADDITION/SUBTRACTION (XOR)
'''
@numba.jit()
def gf2_add(a, b):

    a, b = to_same_dim(a, b)

    return np.logical_xor(a, b, dtype='uint8').astype('uint8')


'''MULTIPLICATION IN Z mod x^p+1'''

@numba.jit()
def z_mul(a, b):

    #fsize = #2**np.ceil(np.log2(len(a))).astype(np.int64)

    #fslice = slice(0, fsize)

    #ta = np.fft.fft(a, fsize)
    #tb = np.fft.fft(b, fsize)

    #res = np.fft.ifft(ta * tb)[fslice].copy()

    #res= np.rint(np.real(res)).astype('uint16')

    #a, b = to_same_dim(a, b)

    return np.rint(np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))).astype('uint16') #TODO ? type ?

'''MULTIPLICATION IN GF(2) mod x^p+1'''

@numba.jit()
def gf2_mul(a, b):

    return np.mod(z_mul(a, b), 2).astype('uint8')


@numba.jit()
def gf22_mul(x, y):
    irr_poly = np.array(([1] + [0] * (11083 - 1) + [1]), dtype='uint8')

    mulres = poly_mul(x, y) #np.mod(np.convolve(x, y), 2)

    out = gf2_div(mulres, irr_poly)[1]

    #padlen = (len(irr_poly)-1)-len(out)

    out = padding(out, (len(irr_poly)-1))

    return out


'''DIVISION IN GF(2) mod x^p+1'''

#TODO deconvolution ?

'''
def gf22_div(a, b):


    fsize = max(len(a), len(b))


    fslice = slice(0, fsize)



    ta = np.fft.fft(a, fsize)
    tb = np.fft.fft(b, fsize)

    ta_r, tb_r = np.real(ta), np.real(tb)
    ta_c, tb_c = np.imag(ta[1:]), np.imag(tb[1:])



    tq_r, tr_r = np.divmod(ta_r, tb_r)
    tq_c, tr_c = np.divmod(ta_c, tb_c)

    tq = tq_r + 1j*(np.insert(tq_c, 0, values=0))
    tr = tr_r  +1j *(np.insert(tr_c, 0, values=0))

    q = np.mod(np.rint(np.real(np.fft.ifft(tq, len(a))[fslice].copy())), 2 ).astype('uint8')
    r = np.mod(np.rint(np.real(np.fft.ifft(tr, len(a))[fslice].copy())), 2 ).astype('uint8')
    
    

    return q, r
'''

@numba.jit()
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


@numba.jit(nopython=True)
def numba_div(f, g, df, dg):

    # remove leading zeros zeros in advance
    h, dq, dr = f, df - dg, dg - 1

    for i in range(0, df + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff = coeff ^  (h[i + j - dg] and g[dg - j])

        if i <= dq:
            coeff = coeff

        h[i] = coeff

    return h[:dq + 1], h[dq + 1:] # not stripped



'''EXTENDED EUCLIDEAN ALGORITHM IN GF(2)'''

#TODO numba accel
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

        if not (q.any() and x1.any()): # if q is zeros or x1 is zeros
            x2 = x0
        elif not x0.any(): # if x0 is all zeros
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

    return  y2, x2, b

@numba.jit()
def gf2_inv(f, g):
    f = np.copy(f)
    g = np.copy(g)

    f = np.flip(f, axis=0)
    g = np.flip(g, axis=0)

    out = xgcd(strip_zeros(f), g)[0]

    out = np.flip(out, axis= 0)


    return out

@numba.jit()
def circtranspose(a):

    at = np.copy(a)

    at[1:] = np.flip(at[1:], axis=0)

    return at



def main():


    #a = np.array((1,0,0,1,1,0), dtype='uint8')
    #b = np.array((1,0,1), dtype='uint8')

    #c = poly_mul(a, b)

    #aa = gf2_div(c, b)

    #aaa= gf22_div(c, b)

    #print(c)

    irr_poly = np.array([1] + [0] * (11083 - 1) + [1], dtype='uint8')

    from sympy_math_ops import gf2_inv as sympy_inv

    from sympy_math_ops import gf2_mul as sympy_mul

    from sympy_math_ops import gf2_div as sympy_div

    from sympy_math_ops import gf2_gcd

    for i in range(100):

        #length1 #= np.random.randint(0,11084, 1)
        #length2  #np.random.randint(0,11084, 1)

        a = np.random.randint(0, 2, 11083, dtype=np.uint8)
        b = np.random.randint(0, 2, 11083, dtype=np.uint8)

        mul2 = gf2_mul(a, b)

        #a = np.flip(a, axis=0)
        #b = np.flip(b, axis=0)
        mul1 = gf22_mul(a, b)
        #mul1 = np.flip(mul1, axis=0)






        assert (mul1 == mul2).all()



    for i in range(1):

        length1 = np.random.randint(0,11084, 1)
        length2 = np.random.randint(0,11084, 1)

        a = np.random.randint(0, 2, length1[0], dtype=np.int64)
        b = np.random.randint(0, 2, length2[0], dtype=np.int64)


        mul3 = poly_mul(a, b)
        mul4 = np.mod(np.convolve(a,b), 2)


        assert (mul3 == mul4).all()







    for i in range(100):

        a = np.random.randint(0, 2, 11083, dtype=np.uint8)

        if np.sum(a) % 2 ==0:
            continue
        else:
            inv1 = gf2_inv(a, irr_poly)

            inv1 = inv1[-11083:]

            inv2 = sympy_inv(a, irr_poly)

            assert (inv2 == inv1).all()

            check2 = gf22_mul(inv2, a)

            check22 = sympy_mul(inv2, a)

            assert np.sum(check2) == 1

            assert (check2[-1] == 1)

            assert np.sum(check22) == 1

            assert (check22[-1] == 1)

            #inv2 =np.roll(inv2,1)

            check3 = gf2_mul(inv2, a)

            assert np.sum(check3) == 1

            assert (check3[-1] == 1)











if __name__ == '__main__':
    main()