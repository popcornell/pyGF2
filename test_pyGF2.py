import unittest
import numpy as np
from gf2_long_div import gf2_div
from gf2_multiplication import gf2_mul
from gf2_add import gf2_add
from generic_functions import strip_zeros
from gf2_inv import xgcd

class test_gf2(unittest.TestCase):



    def test_gf2_add(self):


        for i in range(100):
            grade1 = np.random.randint(0, 11084, 1)
            grade2 = np.random.randint(0, 11084, 1)

            a = np.random.randint(0, 2, grade1, dtype='uint8')
            b = np.random.randint(0, 2, grade2, dtype='uint8')

            sum = gf2_add(a, b)

            a_2 = gf2_add(sum, b)
            b_2 = gf2_add(sum, a)

            assert np.array_equal(a_2, strip_zeros(a)) == True
            assert np.array_equal(b_2, strip_zeros(b)) == True


    def test_gf2_mul(self):

        for i in range(100):

            grade1 = np.random.randint(0, 11084, 1)
            grade2 = np.random.randint(0, 11084, 1)
            grade3 = np.random.randint(0, 11084, 1)

            # distributive property

            a = np.random.randint(0, 2, grade1, dtype='uint8')
            b = np.random.randint(0, 2, grade2, dtype='uint8')
            c = np.random.randint(0, 2, grade3, dtype='uint8')

            res1 = gf2_mul(c, gf2_add(a,b))

            res2 = gf2_add(gf2_mul(c, a), gf2_mul(c, b))

            assert np.array_equal(res1, res2) == True

            # check if same with np.convolve

            res1_2  = np.mod(np.convolve(c, gf2_add(a, b)),2).astype("uint8")

            assert np.array_equal(res1, strip_zeros(res1_2)) == True


    def test_gf2_div(self):

        for i in range(100):
            
            grade1 = np.random.randint(0, 11084, 1)
            grade2 = np.random.randint(0, 11084, 1)

            dividend = np.random.randint(0, 2, grade1)
            divisor = np.random.randint(0, 2, grade2)

            quotient, remainder = gf2_div(dividend, divisor)

            dividend2 = gf2_add(gf2_mul(quotient, divisor), remainder)

            assert np.array_equal(dividend, dividend2) == True


    def test_gf2_inv(self):

        p = 11083

        for i in range(100):

            a = np.random.randint(0, 2, p).astype("uint8")

            irr_poly = np.array([1] + [0] * (p - 1) + [1], dtype='uint8')

            s1,t1, h1 = xgcd(a, irr_poly)

            check = gf2_add( gf2_mul(a, s1), gf2_mul(irr_poly, t1))

            assert np.array_equal(check, h1) == True


























