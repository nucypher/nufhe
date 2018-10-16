.. implementation-details:

----------------------
Implementation details
----------------------


Polynomial multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~

The main bottleneck of ``NuFHE`` gates is bootstrapping, and in it the most time is taken by multiplication of polynomials. In the FHE scheme used, the polynomials are negacyclic (modulo :math:`x^N + 1`, where :math:`N = 1024` by default), defined for integers modulo :math:`2^{32}` (with the coefficients stored as signed 32-bit integers). One of the factors always has coefficients in range :math:`[-1024, 1024)`. Two methods are can be used for multiplication, convolution via FFT and convolution via NTT (number theory transform) [1]_.


FFT
---

Since the polynomials are negacyclic, it is not enough to transform the coefficients to Fourier space, multiply them and then transform the result back --- that would correspond to the regular cyclic convolution, that is multiplication of polynomials modulo :math:`x^N - 1`. Our polynomials are negacyclic, which makes things slightly more complicated.

A straightforward approach is to extend the array of each polynomial's coefficients, turning :math:`(a_0, \dots, a_{N-1})` into :math:`(a_0, \dots, a_{N-1}, -a_0, \dots, -a_{N-1})`. This way the regular convolution of these extended arrays will result in the negacyclic convolution of original arrays.

The Fourier transform of such a signal (of total size :math:`2N`) results in an array containing only :math:`N` non-zero elements (in the positions with odd indices), of which the last :math:`N/2` are complex conjugates of the first :math:`N/2`. This indicates that a transform of size :math:`N/2` should be sufficient to obtain it. And indeed, if one uses standard approaches [2]_ and takes advantage of the two properties of the extended array: real elements and antiperiodicity, the problem can be reduced to a transform of size :math:`N/2` with some pre- and post-processing.

The algorithm built this way maps poorly on the execution model of a GPU, because the pre- and post-processing required is not perfectly parallelizable.  In ``NuFHE`` a technique based on D. J. Bernstein's tangent FFT [3]_ [4]_ is used, which in its core still has an :math:`N/2`-size Fourier transform, but with a much simpler processing.

The algorithm is as follows. Given a vector :math:`\boldsymbol{a}` of length :math:`N`, we define the forward transform as:

.. math::

    \boldsymbol{c}
        = \mathrm{TFFT} \left[ \boldsymbol{a} \right]
        = \mathrm{FFT} \left[ \boldsymbol{b} \right],

where :math:`\boldsymbol{b}` is a :math:`N/2`-vector with the elements

.. math::

    b_j = \left( a_j - ia_{j+N/2} \right) w^j ,\quad j \in [0, N/2),

and :math:`w = \exp\left( -\pi i / N \right)` is a :math:`2N`-th root of unity. Note that the complex vector :math:`\boldsymbol{c}` consists of the first :math:`N/2` non-zero elements Fourier-transformed extended coefficient array described above, except in a different order. Since we will only use the Fourier-space array for convolution, the order does not matter.

The inverse transform :math:`\boldsymbol{a} = \mathrm{ITFFT} \left[ \boldsymbol{c} \right]` is calculated as:

.. math::

    \boldsymbol{b} = \left( \mathrm{IFFT}\left[ \boldsymbol{c} \right] \right)^*

.. math::

    a_j = \mathrm{Re} \left( b_j w^j \right), \quad
    a_{j + N/2} = \mathrm{Im} \left( b_j w^j \right), \quad
    j \in [0, N/2).

Using this pair of transforms, the negacyclic multiplication of two polynomials with coefficients :math:`\boldsymbol{u}` and :math:`\boldsymbol{v}` is performed simply as

.. math::

    \mathrm{ITFFT} \left[
        \mathrm{TFFT} \left[ \boldsymbol{u} \right] \circ
        \mathrm{TFFT} \left[ \boldsymbol{v} \right]
    \right],

where :math:`\circ` stands for elementwise multiplication of two vectors.

Such pre- and post-processing is simple, perfectly parallel and requires only sequential memory access, which makes it ideal for use on a GPU.

Note that this method will only work as long as the maximum possible result does not exceed the capacity of the floating point number used for the underlying FFT (since the modulo :math:`2^{32}` can only be taken after the FFT). In our case we have coefficients limited by 32 bits and 11 bits respectively, plus 10 bits due to the polynomial size (1024), which fits into 53 bits of the double-precision floating-point significand.


NTT
---

Alternatively, polynomial multiplication can be performed using an NTT, which is essentially an FFT operating on the elements of a finite field of size :math:`M`, where :math:`M` is a prime number. Same as in the case of FFT, using an unmodified pair NTT-INTT results in the regular cyclic convolution, and additional steps are necessary to turn it into the negacyclic one. The scheme is very similar to the one used for FFT, and is described in [5]_.

Given a vector :math:`\boldsymbol{a}` of length :math:`N`, we define the forward transform as:

.. math::

    \boldsymbol{c}
        = \mathrm{TNTT} \left[ \boldsymbol{a} \right]
        = \mathrm{NTT} \left[ \boldsymbol{b} \right],

where :math:`\boldsymbol{b}` is a :math:`N`-vector with the elements

.. math::

    b_j = a_j w^j ,\quad j \in [0, N),

:math:`w = g^{(M - 1) / (2 N)}`, and :math:`g` is a primitive element of the field. This means that :math:`w` is a :math:`2N`-th root of unity in the field, just like the one in the FFT section. Note that :math:`M - 1` must be a multiple of :math:`2N`.

Correspondingly, the inverse transform :math:`\boldsymbol{a} = \mathrm{ITNTT} \left[ \boldsymbol{c} \right]` is

.. math::

    \boldsymbol{a} = \mathrm{INTT}\left[ \boldsymbol{c} \right]

.. math::

    a_j = b_j w^{-j}, \quad j \in [0, N).

Same as in the case of FFT, the negacyclic multiplication of two polynomials with coefficients :math:`\boldsymbol{u}` and :math:`\boldsymbol{v}` is performed as

.. math::

    \mathrm{ITNTT} \left[
        \mathrm{TNTT} \left[ \boldsymbol{u} \right] \circ
        \mathrm{TNTT} \left[ \boldsymbol{v} \right]
    \right].

Since the polynomial coefficients are signed integers, they have to be converted to the field elements first, by taking them modulo :math:`M`. The field must be large enough to accommodate the full range of possible outcome values (53 bits by deafult), before modulo :math:`2^{32}` can be taken.


The choice of modulus in NTT
----------------------------

``NuFHE``, following `cuFHE <https://github.com/vernamlab/cuFHE>`_, uses a specifically chosen modulus and root of unity, which allow for some performance optimizations.

The modulus (the size of the finite field) is chosen to be :math:`M = 2^{64} - 2^{32} + 1`. It has several important properties. First, since the field elements are stored in 64-bit unsigned integers, arithmetic operations using this modulus can take advantage of its form. For example, :math:`a\,\mathrm{mod}\,M` is simply :math:`a` if :math:`a < M` and :math:`a + \mathrm{UInt32}(-1)` if :math:`a \ge M`. Similar optimizations can be employed for subtraction, multiplication or bitshift.

Second, :math:`M - 1` is a multiple of :math:`2^{32}`, which means that it supports NTTs up to that size (when the size is a power of 2), and multiplication of polynomials of up to the size :math:`2^{31}`.

The :math:`N`-th root of unity :math:`w_N` used in NTT can theoretically be based on any primitive element :math:`g` by setting :math:`w = g^{(M - 1) / N}`. ``NuFHE`` (again, following ``cuFHE``) uses a "magic" constant :math:`c = 12037493425763644479`, which is a :math:`((M-1)/2^{32})`-th power of some primitive element. Therefore, for a given :math:`N` (which must be a power of 2), one takes :math:`w_N = c^{2^{32}/N}`. The advantage of using this constant is that :math:`c^{2^{32}/64} = 8`, which means that in NTT one can replace most of multiplications by various powers of :math:`w_N` by modulo bitshifts, which are much faster.


References
~~~~~~~~~~

.. [1] `J. M. Pollard,` `"The Fast Fourier Transform in a Finite Field" <http://dx.doi.org/10.2307/2004932>`_, Mathematics of Computation 25(114), 365--365 (1971).

.. [2] `L. R. Rabiner,` `"On the Use of Symmetry in FFT Computation" <http://dx.doi.org/10.1109/TASSP.1979.1163235>`_, IEEE Transactions on Acoustics Speech and Signal Processing 27(3), 233--239 (1979).

.. [3] `D. J. Bernstein,` `"The Tangent FFT" <https://dx.doi.org/10.1007/978-3-540-77224-8_34>`_, Applied Algebra, Algebraic Algorithms and Error-Correcting Codes 291--300 (2007).

.. [4] `D. J. Bernstein,` `"Fast multiplication and its applications" <http://cr.yp.to/lineartime/multapps-20080515.pdf>`_, Algorithmic Number Theory 44 (2008).

.. [5] `P. Longa` and `M. Naehrig,` `"Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography" <https://www.microsoft.com/en-us/research/publication/speeding-up-the-number-theoretic-transform-for-faster-ideal-lattice-based-cryptography/>`_.

