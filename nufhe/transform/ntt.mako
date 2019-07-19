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

## NTT kernels use code from https://github.com/vernamlab/cuFHE
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

<%def name="ntt1024(prefix)">
<%
    i32 = dtypes.ctype(numpy.int32)

    def lsh(label, shift):
        build_call = lambda module: (
            "{module}({label}, {shift})".format(module=module, label=label, shift=shift))
        if shift < 32:
            return build_call(lsh32)
        elif shift < 64:
            return build_call(lsh64)
        elif shift < 96:
            return build_call(lsh96)
        elif shift < 128:
            return build_call(lsh128)
        elif shift < 160:
            return build_call(lsh160)
        elif shift < 192:
            return build_call(lsh192)
        else:
            raise NotImplementedError(shift)
%>

%if use_constant_memory:
#define ${prefix}CDATA_QUALIFIER CONSTANT_MEM_ARG
%else:
#define ${prefix}CDATA_QUALIFIER GLOBAL_MEM_ARG
%endif


WITHIN_KERNEL INLINE void ${prefix}swap(${ff_elem} *a, ${ff_elem} *b)
{
    ${ff_elem} t = *b;
    *b = *a;
    *a = t;
}


WITHIN_KERNEL INLINE void ${prefix}NTT2(${ff_elem}* r0, ${ff_elem}* r1)
{
    ${ff_elem} t = ${sub}(*r0, *r1);
    *r0 = ${add}(*r0, *r1);
    *r1 = t;
}


WITHIN_KERNEL INLINE void ${prefix}NTT2_pair(${ff_elem}* r)
{
    ${prefix}NTT2(&r[0], &r[1]);
}


WITHIN_KERNEL INLINE void ${prefix}NTTInv2(${ff_elem}* r0, ${ff_elem}* r1)
{
    ${prefix}NTT2(r0, r1);
}


WITHIN_KERNEL INLINE void ${prefix}NTTInv2_pair(${ff_elem}* r)
{
    ${prefix}NTT2(&r[0], &r[1]);
}


WITHIN_KERNEL INLINE void ${prefix}NTT8(${ff_elem}* r)
{
    ${prefix}NTT2(&r[0], &r[4]);
    ${prefix}NTT2(&r[1], &r[5]);
    ${prefix}NTT2(&r[2], &r[6]);
    ${prefix}NTT2(&r[3], &r[7]);
    r[5] = ${lsh("r[5]", 24)};
    r[6] = ${lsh("r[6]", 48)};
    r[7] = ${lsh("r[7]", 72)};
    // instead of calling NTT4 ...
    ${prefix}NTT2(&r[0], &r[2]);
    ${prefix}NTT2(&r[1], &r[3]);
    r[3] = ${lsh("r[3]", 48)};
    ${prefix}NTT2(&r[4], &r[6]);
    ${prefix}NTT2(&r[5], &r[7]);
    r[7] = ${lsh("r[7]", 48)};
    ${prefix}NTT2_pair(&r[0]);
    ${prefix}NTT2_pair(&r[2]);
    ${prefix}NTT2_pair(&r[4]);
    ${prefix}NTT2_pair(&r[6]);
    // ... we save 2 swaps (otherwise 4) here
    ${prefix}swap(&r[1], &r[4]);
    ${prefix}swap(&r[3], &r[6]);
}


WITHIN_KERNEL INLINE void ${prefix}NTTInv8(${ff_elem}* r)
{
    ${prefix}NTTInv2(&r[0], &r[4]);
    ${prefix}NTTInv2(&r[1], &r[5]);
    ${prefix}NTTInv2(&r[2], &r[6]);
    ${prefix}NTTInv2(&r[3], &r[7]);
    r[5] = ${lsh("r[5]", 168)};
    r[6] = ${lsh("r[6]", 144)};
    r[7] = ${lsh("r[7]", 120)};
    // instead of calling NTT4 ...
    ${prefix}NTTInv2(&r[0], &r[2]);
    ${prefix}NTTInv2(&r[1], &r[3]);
    r[3] = ${lsh("r[3]", 144)};
    ${prefix}NTTInv2(&r[4], &r[6]);
    ${prefix}NTTInv2(&r[5], &r[7]);
    r[7] = ${lsh("r[7]", 144)};
    ${prefix}NTTInv2_pair(&r[0]);
    ${prefix}NTTInv2_pair(&r[2]);
    ${prefix}NTTInv2_pair(&r[4]);
    ${prefix}NTTInv2_pair(&r[6]);
    // ... we save 2 swaps (otherwise 4) here
    ${prefix}swap(&r[1], &r[4]);
    ${prefix}swap(&r[3], &r[6]);
}


WITHIN_KERNEL INLINE void ${prefix}NTT8x2Lsh_1(${ff_elem}* s)
{
    %for j in range(1, 8):
    s[${j}] = ${lsh("s[{j}]".format(j=j), 12 * j)};
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}NTT8x2Lsh(${ff_elem}* s, ${ff.u32} col)
{
    if (1 == col)
        ${prefix}NTT8x2Lsh_1(s);
}


WITHIN_KERNEL INLINE void ${prefix}NTTInv8x2Lsh_1(${ff_elem}* s)
{
    %for j in range(1, 8):
    s[${j}] = ${lsh("s[{j}]".format(j=j), 192 - 12 * j)};
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}NTTInv8x2Lsh(${ff_elem}* s, ${ff.u32} col)
{
    if (1 == col)
        ${prefix}NTTInv8x2Lsh_1(s);
}


%for i in range(1, 8):
WITHIN_KERNEL INLINE void ${prefix}NTT8x8Lsh_${i}(${ff_elem}* s)
{
    %for j in range(1, 8):
    s[${j}] = ${lsh("s[{j}]".format(j=j), 3 * i * j)};
    %endfor
}
%endfor


WITHIN_KERNEL INLINE void ${prefix}NTT8x8Lsh(${ff_elem}* s, ${ff.u32} col)
{
    if (1 == col)
        ${prefix}NTT8x8Lsh_1(s);
    else if (2 == col)
        ${prefix}NTT8x8Lsh_2(s);
    else if (3 == col)
        ${prefix}NTT8x8Lsh_3(s);
    else if (4 == col)
        ${prefix}NTT8x8Lsh_4(s);
    else if (5 == col)
        ${prefix}NTT8x8Lsh_5(s);
    else if (6 == col)
        ${prefix}NTT8x8Lsh_6(s);
    else if (7 == col)
        ${prefix}NTT8x8Lsh_7(s);
}


%for i in range(1, 8):
WITHIN_KERNEL INLINE void ${prefix}NTTInv8x8Lsh_${i}(${ff_elem}* s)
{
    %for j in range(1, 8):
    s[${j}] = ${lsh("s[{j}]".format(j=j), 192 - 3 * i * j)};
    %endfor
}
%endfor


WITHIN_KERNEL INLINE void ${prefix}NTTInv8x8Lsh(${ff_elem}* s, ${ff.u32} col)
{
    if (1 == col)
        ${prefix}NTTInv8x8Lsh_1(s);
    else if (2 == col)
        ${prefix}NTTInv8x8Lsh_2(s);
    else if (3 == col)
        ${prefix}NTTInv8x8Lsh_3(s);
    else if (4 == col)
        ${prefix}NTTInv8x8Lsh_4(s);
    else if (5 == col)
        ${prefix}NTTInv8x8Lsh_5(s);
    else if (6 == col)
        ${prefix}NTTInv8x8Lsh_6(s);
    else if (7 == col)
        ${prefix}NTTInv8x8Lsh_7(s);
}


WITHIN_KERNEL INLINE void ${prefix}Index3DFrom1D(uint3 *t3d, unsigned int t1d, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z)
{
    t3d->x = t1d % dim_x;
    t1d /= dim_x;
    t3d->y = t1d % dim_y;
    t3d->z = t1d / dim_y;
}


WITHIN_KERNEL INLINE void ${prefix}_forward(
        ${ff_elem}* r,
        LOCAL_MEM_ARG ${ff_elem}* s,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* twd,
        const ${ff.u32} t1d)
{
    uint3 t3d;
    ${prefix}Index3DFrom1D(&t3d, t1d, 8, 8, 2);

    LOCAL_MEM_ARG ${ff_elem}* ptr;

    ${prefix}NTT8(r);
    ${prefix}NTT8x2Lsh(r, t3d.z);
    ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
    LOCAL_BARRIER;

    ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        r[i] = ptr[i << 6];
    ${prefix}NTT2_pair(r);
    ${prefix}NTT2_pair(r + 2);
    ${prefix}NTT2_pair(r + 4);
    ${prefix}NTT2_pair(r + 6);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[i << 6] = r[i];
    LOCAL_BARRIER;

    ptr = &s[t1d];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        r[i] = ${mul_prepared}(ptr[i << 7], twd[i << 7 | t1d]); // mult twiddle
    ${prefix}NTT8(r);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[i << 7] = r[i];
    LOCAL_BARRIER;

    ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        r[i] = ptr[i << 2];
    ${prefix}NTT8x8Lsh(r, t1d >> 4); // less divergence if put here!
    ${prefix}NTT8(r);
}


WITHIN_KERNEL INLINE void ${prefix}_inverse(
        ${ff_elem}* r,
        LOCAL_MEM_ARG ${ff_elem}* s,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* twd,
        const unsigned int t1d)
{
    uint3 t3d;
    ${prefix}Index3DFrom1D(&t3d, t1d, 8, 8, 2);

    LOCAL_MEM_ARG ${ff_elem}* ptr;

    ${prefix}NTTInv8(r);
    ${prefix}NTTInv8x2Lsh(r, t3d.z);
    ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
    LOCAL_BARRIER;

    ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        r[i] = ptr[i << 6];
    ${prefix}NTT2_pair(r);
    ${prefix}NTT2_pair(r + 2);
    ${prefix}NTT2_pair(r + 4);
    ${prefix}NTT2_pair(r + 6);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[i << 6] = r[i];
    LOCAL_BARRIER;

    ptr = &s[t1d];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        r[i] = ${mul_prepared}(ptr[i << 7], twd[i << 7 | t1d]); // mult twiddle
    ${prefix}NTTInv8(r);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i ++)
        ptr[i << 7] = r[i];
    LOCAL_BARRIER;

    ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
    #pragma unroll
        for (unsigned int i = 0; i < 8; i ++)
    r[i] = ptr[i << 2];
    ${prefix}NTTInv8x8Lsh(r, t1d >> 4); // less divergence if put here!
    ${prefix}NTTInv8(r);
}


WITHIN_KERNEL INLINE void ${prefix}forward(
        ${ff_elem}* r_out,
        ${ff_elem}* r_in,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    // Preprocess
    %for i in range(8):
    r_out[${i}] = ${mul_prepared}(
        r_in[${i}],
        cdata[1024 + ${i * 128} + thread_in_xform]
        );
    %endfor

    ${prefix}_forward(r_out, temp, cdata, thread_in_xform);
}


WITHIN_KERNEL INLINE void ${prefix}inverse(
        ${ff_elem}* r_out,
        ${ff_elem}* r_in,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    ${prefix}_inverse(r_in, temp, cdata, thread_in_xform);

    // Postprocess
    %for i in range(8):
    r_out[${i}] = ${mul_prepared}(
        r_in[${i}],
        cdata[1024 + ${i * 128} + thread_in_xform]
        );
    %endfor
}


WITHIN_KERNEL INLINE ${ff_elem} ${prefix}i32_to_elem(${i32} x)
{
    ${ff_elem} res = { (${ff.u64})x - (${ff.u32})(-(x < 0)) };
    return res;
}


WITHIN_KERNEL INLINE ${i32} ${prefix}ff_to_i32(${ff_elem} x)
{
    // Interpreting anything > P/2 as a negative integer,
    // then taking modulo 2^31
    const ${ff.u64} med = ${ff.modulus} / 2;
    return (${i32})(x.val) - (x.val > med);
}



WITHIN_KERNEL INLINE void ${prefix}forward_i32(
        ${ff_elem}* r_out,
        ${i32}* r_in,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    %for i in range(8):
    r_out[${i}] = ${prefix}i32_to_elem(r_in[${i}]);
    %endfor
    ${prefix}forward(r_out, r_out, temp, cdata, thread_in_xform);
}


WITHIN_KERNEL INLINE void ${prefix}inverse_i32(
        ${i32}* r_out,
        ${ff_elem}* r_in,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    ${prefix}inverse(r_in, r_in, temp, cdata, thread_in_xform);
    %for i in range(8):
    r_out[${i}] = ${prefix}ff_to_i32(r_in[${i}]);
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}noop()
{
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
}


WITHIN_KERNEL INLINE void ${prefix}forward_i32_shared(
        LOCAL_MEM_ARG ${ff_elem}* in_out,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    ${ff_elem} r[8];
    %for i in range(8):
    r[${i}] = in_out[${i * 128} + thread_in_xform];
    %endfor
    LOCAL_BARRIER;
    ${prefix}forward(r, r, temp, cdata, thread_in_xform);
    LOCAL_BARRIER;
    %for i in range(8):
    in_out[${i * 128} + thread_in_xform] = r[${i}];
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}inverse_i32_shared_add(
        LOCAL_MEM_ARG ${i32}* out,
        LOCAL_MEM_ARG ${ff_elem}* in,
        LOCAL_MEM_ARG ${ff_elem}* temp,
        ${prefix}CDATA_QUALIFIER ${ff_elem}* cdata,
        unsigned int thread_in_xform)
{
    ${ff_elem} r[8];
    %for i in range(8):
    r[${i}] = in[${i * 128} + thread_in_xform];
    %endfor
    LOCAL_BARRIER;
    ${prefix}inverse(r, r, temp, cdata, thread_in_xform);
    LOCAL_BARRIER;
    %for i in range(8):
    out[${i * 128} + thread_in_xform] += ${prefix}ff_to_i32(r[${i}]);
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}noop_shared()
{
    LOCAL_BARRIER;
    ${prefix}noop();
    LOCAL_BARRIER;
}

</%def>
