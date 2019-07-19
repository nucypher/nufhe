## Copyright (C) 2018 NuCypher
##
## This file is part of nufhe.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Arithmetic functions use the CUDA assembler code from from https://github.com/vernamlab/cuFHE
##
## cuFHE is licensed under MIT license
##
## Copyright (c) 2018 Vernam Group
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

<%def name="ff_elem_def(prefix)">
typedef struct _${prefix}
{
    ${u64} val;
} ${prefix};


WITHIN_KERNEL INLINE ${prefix} ${prefix}pack(${u64} x)
{
    ${prefix} res = {x};
    return res;
}

WITHIN_KERNEL INLINE ${u64} ${prefix}unpack(${prefix} x)
{
    return x.val;
}

#define ${prefix}zero (${prefix}pack(0));


#define ${prefix}UNPACK(hi, lo, src) {lo = (src); hi = (src) >> 32;}

#ifdef CUDA
#define ${prefix}PACK(hi, lo) (((${u64})(hi) << 32) | (lo))
#else
#define ${prefix}PACK(hi, lo) upsample(hi, lo)
#endif


#define ${prefix}SUB_CC(hi, lo, x1, x2) { bool cc = (x1) < (x2); lo = (x1) - (x2); if (cc) { hi -= 1; } }
#define ${prefix}ADD_CC(hi, lo, x1, x2, hi_prev) { lo = (x1) + (x2); bool cc = lo < (x2); hi = hi_prev; if (cc) { hi += 1; } }

</%def>


<%def name="add_def(prefix)">
// Addition in FF(P): val_ = a + b mod P.
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} a, ${ff_elem} b)
{
    %if method == "cuda_asm":

    ${ff_elem} res = {0};
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p;\n\t"
        // this = a + b;
        "add.u64            %0, %1, %2;\n\t"
        // this += (uint32_t)(-(this < b || this >= FFP_MODULUS));
        "setp.lt.u64        p, %0, %2;\n\t"
        "set.ge.or.u32.u64  m, %0, %3, p;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "add.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(res.val)
        : "l"(a.val), "l"(b.val), "l"(${ff.modulus}));
    return res;

    %elif method == "c" or method == "c_from_asm":

    /*
    Algorithm:
    We calculate `s = x + y`
    Now there are three variants:
    - `s < P` and no integer overflow: all good, `result = s`.
    - `s > P` and no integer overflow: `result = s - P = s + (2^32 - 1)`
    - integer overflow, so essentially `s = x + y - N`.
      This means that we need to calculate `result = s + N - P = s + (2^32 - 1)`.
    Note that the last two variants result in the same modifier being applied.
    */
    ${ff_elem} res = {a.val + b.val};
    res.val += ((res.val < b.val) || res.val >= ${ff.modulus}) ? 0xffffffff : 0;
    return res;

    %endif
}
</%def>


<%def name="sub_def(prefix)">
/** Subtraction in FF(P): val_ = a + b mod P. */
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} a, ${ff_elem} b)
{
    %if method == "cuda_asm":

    ${ff_elem} res = {0};
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        // this = a - b;
        "sub.u64            %0, %1, %2;\n\t"
        // this -= (uint32_t)(-(this > a));
        "set.gt.u32.u64     m, %0, %1;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "sub.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(res.val)
        : "l"(a.val), "l"(b.val));
    return res;

    %elif method == "c" or method == "c_from_asm":

    /*
    Algorithm:
    We calculate `s = x - y`
    Now there are three variants:
    - no underflow (x >= y): all good, `result = s`.
    - underflow (detected if `s > x`), so essentially `s = x - y + N`.
      This means we need to calculate `s - N + P = s - (2^32 - 1)`
    */

    ${ff_elem} res = {a.val - b.val};
    ${ff.u32} x = -(res.val > a.val);
    res.val -= x;
    return res;

    %endif
}
</%def>


<%def name="mod_def(prefix)">
// TODO: technically, this operation is inplace, so `a` can be passed by pointer
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff.u64} a)
{
    %if method == "cuda_asm":

    ${ff_elem} res = {0};
    asm("{\n\t"
        ".reg .u32        m;\n\t"
        ".reg .u64        t;\n\t"
        "mov.u64          %0, %1;\n\t"
        "set.ge.u32.u64   m, %0, %2;\n\t"
        "mov.b64          t, {m, 0};\n\t"
        "add.u64         %0, %0, t;\n\t"
        "}"
        : "+l"(res.val)
        : "l"(a), "l"(${ff.modulus}));
    return res;

    %elif method == "c" or method == "c_from_asm":

    // uses the fact that 2 * P > max(UInt64)
    // and that a::UInt64 - P == a + 2^32 - 1

    ${ff_elem} res = {a};
    res.val += (res.val >= ${ff.modulus}) ? 0xffffffff : 0;
    return res;

    %endif
}
</%def>


<%def name="mul_def(prefix)">
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} a, ${ff_elem} b)
{
    /*
    Algorithm:
    Let M = 2^32
    Then `(a * b) = m0 + m1 * M + m2 * M^2 + m3 * M^3`
    and `(a * b) mod P = (m0 - m2 - m3 + (m1 + m2) * M) mod P`.
    Now `m0 - m2 - m3` can range from `-2M` to `M`, so if it's negative,
    we need to carry 1 or 2 into the sum `m1 + m2`, and process the overflow correctly.
    */

    %if method == "cuda_asm":

    ${ff.u64} res = 0;
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          m0, m1, m2, m3;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p, q;\n\t"
        // 128-bit = 64-bit * 64-bit
        "mul.lo.u64         t, %1, %2;\n\t"
        "mov.b64            {m0, m1}, t;\n\t"
        "mul.hi.u64         t, %1, %2;\n\t"
        "mov.b64            {m2, m3}, t;\n\t"
        // 128-bit mod P with add / sub
        "add.u32            r1, m1, m2;\n\t"
        "sub.cc.u32         r0, m0, m2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // fix result
        "setp.eq.u32        p|q, m2, 0;\n\t"
        "mov.b64            t, {m0, m1};\n\t"
        // ret -= (uint32_t)(-(ret > mul[0] && m[2] == 0));
        "set.gt.and.u32.u64 m3, %0, t, p;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < mul[0] && m[2] != 0));
        "set.lt.and.u32.u64 m3, %0, t, q;\n\t"
        "add.cc.u32         r0, r0, m3;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(res)
        : "l"(a.val), "l"(b.val));

    return ${mod}(res);


    %elif method == "c":

        // 128-bit = 64-bit * 64-bit
        ${ff.u32} r0, r1, m0, m1, m2, m3;

        #ifdef CUDA
        ${ff.u64} m3_m2 = __umul64hi(a.val, b.val);
        #else
        ${ff.u64} m3_m2 = mul_hi(a.val, b.val);
        #endif
        ${ff.u64} m1_m0 = a.val * b.val;

        ${ff_elem}UNPACK(m3, m2, m3_m2);
        ${ff_elem}UNPACK(m1, m0, m1_m0);

        bool c1, c2;

        r1 = m1;
        r0 = -m3;
        c1 = m1 > 0;
        c2 = m3 > 0;
        if (c2) r1--;
        if (!c1 && c2) {r0++; if (m3 > 1) r1--; }
        ${ff_elem} x1 = { ${ff_elem}PACK(r1, r0) }; // m1 * M - m3

        r1 = m2;
        r0 = -m2;
        c1 = m2 > 0;
        if (c1) r1--;
        // at this point we have (r1, r0) = (m2 * M - m2) mod P

        r0 += m0;
        c1 = r0 < m0;
        if (c1) { r1++; }
        c2 = (r0 > 0 && r1 == 0xffffffff);
        if (r0 > 0 && r1 == 0xffffffff) { r1 = 0; r0--; }

        ${ff_elem} x2 = { ${ff_elem}PACK(r1, r0) }; // m2 * M - m2 + m0

        return ${add}(x1, x2);


    %elif method == "c_from_asm":

        // 128-bit = 64-bit * 64-bit
        ${ff.u64} res, t;
        ${ff.u32} r0, r1, m0, m1, m2, m3;

        #ifdef CUDA
        ${ff.u64} m3_m2 = __umul64hi(a.val, b.val);
        #else
        ${ff.u64} m3_m2 = mul_hi(a.val, b.val);
        #endif
        ${ff.u64} m1_m0 = a.val * b.val;

        ${ff_elem}UNPACK(m3, m2, m3_m2);
        ${ff_elem}UNPACK(m1, m0, m1_m0);

        // 128-bit mod P with add / sub
        r1 = m1 + m2;
        ${ff_elem}SUB_CC(r1, r0, m0, m2);
        ${ff_elem}SUB_CC(r1, r0, r0, m3);
        res = ${ff_elem}PACK(r1, r0);

        // fix result
        bool p = m2 == 0;
        bool q = !p;

        t = ${ff_elem}PACK(m1, m0);

        // ret -= (uint32_t)(-(ret > mul[0] && m[2] == 0));
        m3 = (res > t && p) ? 0xffffffff : 0;
        ${ff_elem}SUB_CC(r1, r0, r0, m3);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret < mul[0] && m[2] != 0));
        m3 = (res < t && q) ? 0xffffffff : 0;
        ${ff_elem}ADD_CC(r1, r0, r0, m3, r1);
        res = ${ff_elem}PACK(r1, r0);

        return ${mod}(res);

    %endif
}
</%def>


<%def name="prepare_for_mul_def(prefix)">
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} a)
{
    /*
    Convert the given number to Montgomery representation with the fixed modulus (M=2**64-2**32+1)
    and fixed word size (R=2**64), for later use with `mul_prepared()`.
    The result is `a * R mod M`.
    */

    ${ff.u64} lo = a.val & 0xffffffff;
    ${ff.u64} hi = a.val >> 32;
    ${ff.u64} y = (lo << 32) - lo;
    ${ff_elem} y_ff = { y };
    ${ff_elem} hi_ff = { hi };
    return ${sub}(y_ff, hi_ff);
}
</%def>


<%def name="mul_prepared_def(prefix)">
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} a, ${ff_elem} b)
{
    /*
    This function performs Montgomery multiplication with the fixed modulus (M=2**64-2**32+1)
    and fixed word size (R=2**64), which helps simplify the algorithm.
    The result is `a * b * R**(-1) mod M`.

    Note that if you multiply two numbers `a` and `b` in Montgomery representation
    (a' = a * R mod M, b' = b * R mod M), the result is the Montgomery representation of their
    product (a' * b' * R**(-1) mod M = a * b * R mod M).
    But if one of the numbers is in Montgomery representation and the other is not, you
    get the normal product back (a * b' * R**(-1) mod M = a * b mod M).

    This way if one of the factors is precalculated, you can convert it
    into Montgomery representation, and use this function instead of the general
    multiplication function (which is slower).
    */

    %if method == "cuda_asm":

    ${ff.u64} hi = 0;
    ${ff.u64} p2 = 0;
    asm("{\n\t"
        ".reg .u64  lo, hi, t1, t2, t3; \n\t"
        "mul.lo.u64 lo, %2, %3;         \n\t"
        "mul.hi.u64 hi, %2, %3;         \n\t"
        "shl.b64    t1, lo, 32;         \n\t" // t1 = lo << 32
        "add.u64    lo, lo, t1;         \n\t" // (u) lo = (lo << 32) + lo
        "shr.b64    t1, lo, 32;         \n\t" // t1 = lo >> 32
        "shl.b64    t2, lo, 32;         \n\t" // (uu) t2 = lo << 32
        "sub.cc.u64 t3, lo, t2;         \n\t"
        "sub.u64    lo, lo, t1;         \n\t" // (p2) lo = u - (u >> 32)
        "subc.u64   lo, lo, 0;          \n\t"
        "mov.b64    %0, hi;             \n\t"
        "mov.b64    %1, lo;             \n\t"
        "}"
        : "+l"(hi), "+l"(p2)
        : "l"(a.val), "l"(b.val));

    %elif method == "c" or method == "c_from_asm":

    #ifdef CUDA
    ${ff.u64} hi = __umul64hi(a.val, b.val);
    #else
    ${ff.u64} hi = mul_hi(a.val, b.val);
    #endif
    ${ff.u64} lo = a.val * b.val;

    ${ff.u64} u = (lo << 32) + lo;
    ${ff.u64} p2 = u - (u >> 32);
    ${ff.u64} uu = u << 32;
    if (uu > u)
    {
        p2 -= 1;
    }

    %endif

    ${ff_elem} hi_ff = { hi };
    ${ff_elem} p2_ff = { p2 };

    return ${sub}(hi_ff, p2_ff);
}
</%def>


<%def name="pow_def(prefix)">
// Exponentiation in FF(P): val_ = val_ ^ e mod P.
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} x, ${dtypes.ctype(exp_dtype)} e)
{
    if (0 == e)
    {
        x.val = 1;
        return x;
    }
    ${ff_elem} y = {1};
    ${ff.u32} n = e;
    while (n > 1)
    {
        if (0 != (n & 0x1))
            y = ${mul}(y, x);
        x = ${mul}(x, x);
        n >>= 1;
    }
    return ${mul}(x, y);
}
</%def>


<%def name="inv_pow2_def(prefix)">
<%
    assert numpy.dtype(exp_dtype) == numpy.dtype('uint32')
    exp_ctype = dtypes.ctype(exp_dtype)
%>
/**
* Return the inverse of 2^log_n in FF(P): 2^{-log_n} mod P.
* @param log_n An integer in (0, 32]
*/
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${exp_ctype} log_n)
{
    ${ff.u32} r0 = (1 << (32 - (${ff.u32})log_n)) + 1;
    ${ff.u32} r1 = -r0;
    ${ff.u64} r = ((${ff.u64})r1 << 32) | r0;
    ${ff_elem} res = {r};
    return res;
}
</%def>


<%def name="lsh_def(prefix)">
<%
    assert numpy.dtype(exp_dtype) == numpy.dtype('uint32')
    exp_ctype = dtypes.ctype(exp_dtype)
%>
/**
* Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
* @param[in] l An integer in [0, 32)
*/
WITHIN_KERNEL INLINE ${ff_elem} ${prefix}(${ff_elem} x, ${exp_ctype} l)
{
    /*
    Algorithm:

    We can decompose the shift as

        res = x * 2^l = x * M^k * 2^j,

    where M=2^32, k is integer, and 0 <= j < 32.
    k=0 corresponds to 0 <= l < 32, k=1 to 32 <= l < 64 and so on.

    After the multiplication by 2^j, the result contains 3 32-bit parts:

        x * 2^j = t2 * M^2 + t1 * M + t0, where 0 <= t0, t1, t2 < M

    Thus

        res = t2 * M^(2+k) + t1 * M^(1+k) + t0 * M^k

    Taking the modulus P = M^2 - M + 1, we get

    k = 0, l in [0, 32)   : (t1 + t2) * M + t0 - t2  = (t1 * M + t0) + (t2 * M) - (t2)
    k = 1, l in [32, 64)  : (t0 + t1) * M - t1 - t2  = ((t0 + t1) * M) - (t1 + t2)
    k = 2, l in [64, 96)  : (t0 - t2) * M - t0 - t1  = (t0 * M) - (t2 * M + t0) - (t1)
    k = 3, l in [96, 128) : (-t1 - t2) * M - t0 + t2 = (t2) - (t1 * M + t0) - (t2 * M)
    k = 4, l in [128, 160): (-t0 - t1) * M + t1 + t2 = (t1 + t2) - ((t0 + t1) * M)
    k = 5, l in [160, 192): (-t0 + t2) * M + t0 + t1 = (t2 * M + t0) - (t0 * M) + (t1)

    Here the arithmetic operations outside the parentheses have to be performed modulo P.

    The processing for the things inside the parentheses is simpler:

    - (x * M + y) = PACK(x, y) -- packing two 32-bit integers into a 64-bit one;
    - (x + y) = PACK(s < y, s), where s = x + y
      (that is, check for overflow and add 1 in the high half)
    - ((x + y) * M) = PACK(s, s < y ? 0xffffffff : 0)
      (that is, check for overflow and add (M-1) in the low half)
    */


    %if method == "c":

    ${ff.u32} j = l - ${exp_range - 32};

    ${ff.u32} t0, t1, t2;
    t0 = x.val;
    t0 = t0 << j;
    ${ff.u64} s = x.val >> (32 - j);
    ${ff_elem}UNPACK(t2, t1, s);


    %if exp_range == 32:

        ${ff_elem} x1 = { ${ff_elem}PACK(t1, t0) };
        ${ff_elem} x2 = { ${ff_elem}PACK(t2, 0) };
        ${ff_elem} x3 = { ${ff_elem}PACK(0, t2) };
        return ${sub}(${add}(x1, x2), x3);


    %elif exp_range == 64:

        ${ff.u32} lo, hi;

        hi = t0 + t1;
        lo = -(hi < t1);
        ${ff_elem} x1 = { ${ff_elem}PACK(hi, lo) }; // (t0 + t1) * M

        lo = t1 + t2;
        hi = lo < t2;
        ${ff_elem} x2 = { ${ff_elem}PACK(hi, lo) }; // t1 + t2

        return ${sub}(x1, x2);


    %elif exp_range == 96:

        ${ff.u32} lo, hi;
        hi = t0;
        lo = -t0;
        if (t0 > 0) hi--;
        ${ff_elem} x1 = { ${ff_elem}PACK(hi, lo) }; // t0 * M - t0
        ${ff_elem} x2 = { ${ff_elem}PACK(t2, t1) }; // t2 * M + t1
        return ${sub}(x1, x2);


    %elif exp_range == 128:

        ${ff_elem} x1 = { ${ff_elem}PACK(0, t2) };
        ${ff_elem} x2 = { ${ff_elem}PACK(t1, t0) };
        ${ff_elem} x3 = { ${ff_elem}PACK(t2, 0) };
        return ${sub}(x1, ${add}(x2, x3));


    %elif exp_range == 160:

        ${ff.u32} lo, hi;

        lo = t1 + t2;
        hi = lo < t2;
        ${ff_elem} x1 = { ${ff_elem}PACK(hi, lo) }; // t1 + t2

        hi = t0 + t1;
        lo = -(hi < t1);
        ${ff_elem} x2 = { ${ff_elem}PACK(hi, lo) }; // (t0 + t1) * M

        return ${sub}(x1, x2);


    %elif exp_range == 192:

        ${ff_elem} x1 = { ${ff_elem}PACK(t2, t0) };
        ${ff_elem} x2 = { ${ff_elem}PACK(t0, 0) };
        ${ff_elem} x3 = { ${ff_elem}PACK(0, t1) };
        return ${add}(x1, ${sub}(x3, x2));

    %endif


    %elif method == "cuda_asm":

    %if exp_range == 32:

        asm("{\n\t"
            ".reg .u32      r0, r1;\n\t"
            ".reg .u32      t0, t1, t2;\n\t"
            ".reg .u32      n;\n\t"
            ".reg .u64      s;\n\t"
            // t[2] = (uint32_t)(x >> (64-l));
            // t[1] = (uint32_t)(x >> (32-l));
            // t[0] = (uint32_t)(x << l);
            "mov.b64        {r0, r1}, %0;\n\t"
            "shl.b32        t0, r0, %1;\n\t"
            "sub.u32        n, 32, %1;\n\t"
            "shr.b64        s, %0, n;\n\t"
            "mov.b64        {t1, t2}, s;\n\t"
            // mod P
            "add.u32        r1, t1, t2;\n\t"
            "sub.cc.u32     r0, t0, t2;\n\t"
            "subc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
            "mov.b64        s, {t0, t1};\n\t"
            "set.lt.u32.u64 t2, %0, s;\n\t"
            "add.cc.u32     r0, r0, t2;\n\t"
            "addc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        return ${mod}(x.val);

    %elif exp_range == 64:

        asm("{\n\t"
            ".reg .u32          r0, r1;\n\t"
            ".reg .u32          t0, t1, t2;\n\t"
            ".reg .u32          n;\n\t"
            ".reg .u64          s;\n\t"
            ".reg .pred         p, q;\n\t"
            // t[2] = (uint32_t)(x >> (96-l));
            // t[1] = (uint32_t)(x >> (64-l));
            // t[0] = (uint32_t)(x << (l-32));
            "mov.b64            {r0, r1}, %0;\n\t"
            "sub.u32            n, %1, 32;\n\t"
            "shl.b32            t0, r0, n;\n\t"
            "sub.u32            n, 32, n;\n\t"
            "shr.b64            s, %0, n;\n\t"
            "mov.b64            {t1, t2}, s;\n\t"
            // mod P
            "add.u32            r1, t0, t1;\n\t"
            "sub.cc.u32         r0, 0, t1;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "sub.cc.u32         r0, r0, t2;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
            "setp.eq.u32        p|q, t1, 0;\n\t"
            "mov.b64            s, {0, t0};\n\t"
            "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
            "sub.cc.u32         r0, r0, t2;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
            "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
            "add.cc.u32         r0, r0, t2;\n\t"
            "addc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        return ${mod}(x.val);

    %elif exp_range == 96:

        asm("{\n\t"
            ".reg .u32      r0, r1;\n\t"
            ".reg .u32      t0, t1, t2;\n\t"
            ".reg .u32      n;\n\t"
            ".reg .u64      s;\n\t"
            // t[2] = (uint32_t)(x >> (128-l));
            // t[1] = (uint32_t)(x >> (96-l));
            // t[0] = (uint32_t)(x << (l-64));
            "mov.b64        {r0, r1}, %0;\n\t"
            "sub.u32        n, %1, 64;\n\t"
            "shl.b32        t0, r0, n;\n\t"
            "sub.u32        n, 32, n;\n\t"
            "shr.b64        s, %0, n;\n\t"
            "mov.b64        {t1, t2}, s;\n\t"
            // mod P
            "add.cc.u32     r0, t1, t0;\n\t"
            "addc.u32       r1, t2, 0;\n\t"
            "sub.u32        r1, r1, t0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            // ret -= (uint32_t)(-(ret > ((uint64_t *)t)[1]));
            "mov.b64        s, {t1, t2};\n\t"
            "set.gt.u32.u64 t2, %0, s;\n\t"
            "sub.cc.u32     r0, r0, t2;\n\t"
            "subc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        x = ${mod}(x.val);
        x.val = ${ff.modulus} - x.val;
        return x;

    %elif exp_range == 128:

        asm("{\n\t"
            ".reg .u32      r0, r1;\n\t"
            ".reg .u32      t0, t1, t2;\n\t"
            ".reg .u32      n;\n\t"
            ".reg .u64      s;\n\t"
            // t[2] = (uint32_t)(x >> (160-l));
            // t[1] = (uint32_t)(x >> (128-l));
            // t[0] = (uint32_t)(x << (l-96));
            "mov.b64        {r0, r1}, %0;\n\t"
            "sub.u32        n, %1, 96;\n\t"
            "shl.b32        t0, r0, n;\n\t"
            "sub.u32        n, 32, n;\n\t"
            "shr.b64        s, %0, n;\n\t"
            "mov.b64        {t1, t2}, s;\n\t"
            // mod P
            "add.u32        r1, t1, t2;\n\t"
            "sub.cc.u32     r0, t0, t2;\n\t"
            "subc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
            "mov.b64        s, {t0, t1};\n\t"
            "set.lt.u32.u64 t2, %0, s;\n\t"
            "add.cc.u32     r0, r0, t2;\n\t"
            "addc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        x = ${mod}(x.val);
        x.val = ${ff.modulus} - x.val;
        return x;

    %elif exp_range == 160:

        asm("{\n\t"
            ".reg .u32          r0, r1;\n\t"
            ".reg .u32          t0, t1, t2;\n\t"
            ".reg .u32          n;\n\t"
            ".reg .u64          s;\n\t"
            ".reg .pred         p, q;\n\t"
            // t[2] = (uint32_t)(x >> (192-l));
            // t[1] = (uint32_t)(x >> (160-l));
            // t[0] = (uint32_t)(x << (l-128));
            "mov.b64            {r0, r1}, %0;\n\t"
            "sub.u32            n, %1, 128;\n\t"
            "shl.b32            t0, r0, n;\n\t"
            "sub.u32            n, 32, n;\n\t"
            "shr.b64            s, %0, n;\n\t"
            "mov.b64            {t1, t2}, s;\n\t"
            // mod P
            "add.u32            r1, t0, t1;\n\t"
            "sub.cc.u32         r0, 0, t1;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "sub.cc.u32         r0, r0, t2;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
            "setp.eq.u32        p|q, t1, 0;\n\t"
            "mov.b64            s, {0, t0};\n\t"
            "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
            "sub.cc.u32         r0, r0, t2;\n\t"
            "subc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
            "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
            "add.cc.u32         r0, r0, t2;\n\t"
            "addc.u32           r1, r1, 0;\n\t"
            "mov.b64            %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        x = ${mod}(x.val);
        x.val = ${ff.modulus} - x.val;
        return x;

    %elif exp_range == 192:

        asm("{\n\t"
            ".reg .u32      r0, r1;\n\t"
            ".reg .u32      t0, t1, t2;\n\t"
            ".reg .u32      n;\n\t"
            ".reg .u64      s;\n\t"
            // t[2] = (uint32_t)(x << (l-160));
            // t[1] = (uint32_t)(x >> (224-l));
            // t[0] = (uint32_t)(x >> (192-l));
            "mov.b64        {r0, r1}, %0;\n\t"
            "sub.u32        n, %1, 160;\n\t"
            "shl.b32        t2, r0, n;\n\t"
            "sub.u32        n, 32, n;\n\t"
            "shr.b64        s, %0, n;\n\t"
            "mov.b64        {t0, t1}, s;\n\t"
            // mod P
            "add.cc.u32     r0, t0, t2;\n\t"
            "addc.u32       r1, t1, 0;\n\t"
            "sub.u32        r1, r1, t2;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            // ret += (uint32_t)(-(ret > ((uint64_t *)t)[0]));
            "mov.b64        s, {t0, t1};\n\t"
            "set.gt.u32.u64 t2, %0, s;\n\t"
            "sub.cc.u32     r0, r0, t2;\n\t"
            "subc.u32       r1, r1, 0;\n\t"
            "mov.b64        %0, {r0, r1};\n\t"
            "}"
            : "+l"(x.val)
            : "r"(l));
        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        return ${mod}(x.val);

    %endif


    %elif method == "c_from_asm":

    %if exp_range == 32:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;

        // t[2] = (uint32_t)(x >> (64-l));
        // t[1] = (uint32_t)(x >> (32-l));
        // t[0] = (uint32_t)(x << l);

        ${ff_elem}UNPACK(r1, r0, res);

        t0 = r0 << l;
        n = 32 - l;
        s = res >> n;

        ${ff_elem}UNPACK(t2, t1, s);

        // mod P
        r1 = t1 + t2;
        ${ff_elem}SUB_CC(r1, r0, t0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        s = ${ff_elem}PACK(t1, t0);
        t2 = -(res < s);
        ${ff_elem}ADD_CC(r1, r0, r0, t2, r1);
        res = ${ff_elem}PACK(r1, r0);

        return ${mod}(res);


    %elif exp_range == 64:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;
        bool p, q;

        // t[2] = (uint32_t)(x >> (96-l));
        // t[1] = (uint32_t)(x >> (64-l));
        // t[0] = (uint32_t)(x << (l-32));

        ${ff_elem}UNPACK(r1, r0, res);

        n = l - 32;
        t0 = r0 << n;
        n = 32 - n;
        s = res >> n;

        ${ff_elem}UNPACK(t2, t1, s);

        // mod P
        r1 = t0 + t1;
        ${ff_elem}SUB_CC(r1, r0, 0, t1);
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        p = t1 == 0;
        q = !p;
        s = ${ff_elem}PACK(t0, 0);
        t2 = (res > s && p) ? 0xffffffff : 0;
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        t2 = (res < s && q) ? 0xffffffff : 0;
        ${ff_elem}ADD_CC(r1, r0, r0, t2, r1);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        return ${mod}(res);


    %elif exp_range == 96:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;

        // t[2] = (uint32_t)(x >> (128-l));
        // t[1] = (uint32_t)(x >> (96-l));
        // t[0] = (uint32_t)(x << (l-64));

        ${ff_elem}UNPACK(r1, r0, res);
        n = l - 64;
        t0 = r0 << n;
        n = 32 - n;
        s = res >> n;
        ${ff_elem}UNPACK(t2, t1, s);

        // mod P
        ${ff_elem}ADD_CC(r1, r0, t1, t0, t2);

        r1 -= t0;
        res = ${ff_elem}PACK(r1, r0);

        // ret -= (uint32_t)(-(ret > ((uint64_t *)t)[1]));
        s = ${ff_elem}PACK(t2, t1);
        t2 = (res > s) ? 0xffffffff : 0;
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        ${ff_elem} ret = ${mod}(res);
        ret.val = ${ff.modulus} - ret.val;
        return ret;


    %elif exp_range == 128:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;

        // t[2] = (uint32_t)(x >> (160-l));
        // t[1] = (uint32_t)(x >> (128-l));
        // t[0] = (uint32_t)(x << (l-96));

        ${ff_elem}UNPACK(r1, r0, res);
        n = l - 96;
        t0 = r0 << n;
        n = 32 - n;
        s = res >> n;
        ${ff_elem}UNPACK(t2, t1, s);

        // mod P
        r1 = t1 + t2;
        ${ff_elem}SUB_CC(r1, r0, t0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        s = ${ff_elem}PACK(t1, t0);
        t2 = (res < s) ? 0xffffffff : 0;
        ${ff_elem}ADD_CC(r1, r0, r0, t2, r1);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        ${ff_elem} ret = ${mod}(res);
        ret.val = ${ff.modulus} - ret.val;
        return ret;


    %elif exp_range == 160:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;
        bool p, q;

        // t[2] = (uint32_t)(x >> (192-l));
        // t[1] = (uint32_t)(x >> (160-l));
        // t[0] = (uint32_t)(x << (l-128));
        ${ff_elem}UNPACK(r1, r0, res);
        n = l - 128;
        t0 = r0 << n;
        n = 32 - n;
        s = res >> n;
        ${ff_elem}UNPACK(t2, t1, s);

        // mod P
        r1 = t0 + t1;
        ${ff_elem}SUB_CC(r1, r0, 0, t1);
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        p = t1 == 0;
        q = !p;
        s = ${ff_elem}PACK(t0, 0);
        t2 = (res > s && p) ? 0xffffffff : 0;
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        t2 = (res < s && q) ? 0xffffffff : 0;
        ${ff_elem}ADD_CC(r1, r0, r0, t2, r1);
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret >= FFP_MODULUS));
        ${ff_elem} ret = ${mod}(res);
        ret.val = ${ff.modulus} - ret.val;
        return ret;

    %elif exp_range == 192:

        ${ff.u64} res = x.val;

        ${ff.u32} r0, r1, t0, t1, t2, n;
        ${ff.u64} s;

        // t[2] = (uint32_t)(x << (l-160));
        // t[1] = (uint32_t)(x >> (224-l));
        // t[0] = (uint32_t)(x >> (192-l));
        ${ff_elem}UNPACK(r1, r0, res);
        n = l - 160;
        t2 = r0 << n;
        n = 32 - n;
        s = res >> n;
        ${ff_elem}UNPACK(t1, t0, s);

        // mod P
        ${ff_elem}ADD_CC(r1, r0, t0, t2, t1);
        r1 -= t2;
        res = ${ff_elem}PACK(r1, r0);

        // ret += (uint32_t)(-(ret > ((uint64_t *)t)[0]));
        s = ${ff_elem}PACK(t1, t0);
        t2 = (res > s) ? 0xffffffff : 0;
        ${ff_elem}SUB_CC(r1, r0, r0, t2);
        res = ${ff_elem}PACK(r1, r0);

        return ${mod}(res);

    %endif

    %endif
}
</%def>
