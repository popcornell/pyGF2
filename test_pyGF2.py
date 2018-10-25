import unittest
import numpy as np
from pyGF2 import gf2_add, gf2_mul, gf2_div, gf2_xgcd, strip_zeros


class test_gf2(unittest.TestCase):
    """Test pyGF2 functions with random polynomials"""



    def test_gf2_add(self):
        """Check if addition and subtraction are the same"""


        for i in range(100):
            degree1 = np.random.randint(0, 11084, 1)
            degree2 = np.random.randint(0, 11084, 1)

            a = np.random.randint(0, 2, degree1, dtype='uint8')
            b = np.random.randint(0, 2, degree2, dtype='uint8')

            sum = gf2_add(a, b)

            a_2 = gf2_add(sum, b)
            b_2 = gf2_add(sum, a)

            assert np.array_equal(a_2, strip_zeros(a)) == True
            assert np.array_equal(b_2, strip_zeros(b)) == True


    def test_gf2_mul(self):
        """Check multiplication using distributive property"""

        for i in range(100):

            degree1 = np.random.randint(0, 11084, 1)
            degree2 = np.random.randint(0, 11084, 1)
            degree3 = np.random.randint(0, 11084, 1)

            # distributive property

            a = np.random.randint(0, 2, degree1, dtype='uint8')
            b = np.random.randint(0, 2, degree2, dtype='uint8')
            c = np.random.randint(0, 2, degree3, dtype='uint8')

            res1 = gf2_mul(c, gf2_add(a,b))

            res2 = gf2_add(gf2_mul(c, a), gf2_mul(c, b))

            assert np.array_equal(res1, res2) == True

            # check if same with np.convolve

            res1_2  = np.mod(np.convolve(c, gf2_add(a, b)),2).astype("uint8")

            assert np.array_equal(res1, strip_zeros(res1_2)) == True


    def test_gf2_div(self):
        """Test polynomial divisiopn in GF2"""

        for i in range(100):
            
            degree1 = np.random.randint(0, 11084, 1)
            degree2 = np.random.randint(0, 11084, 1)

            dividend = np.random.randint(0, 2, degree1, dtype="uint8")
            divisor = np.random.randint(0, 2, degree2, dtype="uint8")

            quotient, remainder = gf2_div(dividend, divisor)

            dividend2 = gf2_add(gf2_mul(quotient, divisor), remainder)

            assert np.array_equal(strip_zeros(dividend), dividend2) == True


    def test_gf2_inv(self):
        """Test Extended Euclidean Algorithm in GF2"""

        p = 11083

        for i in range(100):

            a = np.random.randint(0, 2, p).astype("uint8")

            irr_poly = np.array([1] + [0] * (p - 1) + [1], dtype='uint8')

            s,t, h = gf2_xgcd(a, irr_poly)

            check = gf2_add( gf2_mul(a, s), gf2_mul(irr_poly, t))

            assert np.array_equal(check, h) == True


























