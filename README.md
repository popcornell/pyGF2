# pyGF2
**Optimized Polynomial arithmetic over GF2\[x]**.

This module supports polynomial addition **gf2_add**, multiplication **gf2_mul**
division **gf2_div** and Extended Euclidean Algorithm **gf2_xgcd**.

Polynomials must be represented as ndarrays of **uint8** including zero-valued coefficients.
The rightmost array element is assumed to represent the leading coefficient.

----
For example:

    >>>a =np.array([1, 0, 1, 1], dtype="uint8")

----
It is equivalent to the polynomial *x^3+x^2+1*.
In other words the ordering is the same as the one used in MATLAB.
Sympy and numpy.polys instead assume the leftmost array element to be the leading coefficient.

The use of such ordering allows to use numpy.fft to perform multiplication between polynomials.


----

In general a great effort has been put to speed-up all the operations making this module
suitable for handling very large degree (tens of thousands) GF2\[x] polynomials such
as the ones required in coding theory.

Speed-wise it is orders of magnitude faster for polynomials of large degree
when it is compared to both numpy.polys functions and Sympy.galoistools.

For Example:

 - **numpy.polymul** vs **gf2_mul** for two random GF2\[x] polynomial with 100000 elements:
   **5.63 s vs 105 ms**

 - **numpy.polydiv** encounters overflow problems when dealing with large polynomials.
   **gf2_div** solves that by using only modulo 2 operations.

 - **sympy.galoistools.gf_gcdex** vs **gf2_xgcd** for two random GF2\[x] polynomial with 11083 elements:
   **2 m 5 s vs 1.29 s**