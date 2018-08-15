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

<%def name="fft512(prefix)">

<%
    i32 = dtypes.ctype(numpy.int32)
%>

%if use_constant_memory:
#define ${prefix}CDATA_QUALIFIER CONSTANT_MEM_ARG
%else:
#define ${prefix}CDATA_QUALIFIER GLOBAL_MEM_ARG
%endif


#define complex_ctr COMPLEX_CTR(double2)
#define conj(a) complex_ctr((a).x, -(a).y)
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))

typedef ${dtypes.ctype(numpy.complex128)} complex_t;
typedef ${dtypes.ctype(numpy.float64)} real_t;


WITHIN_KERNEL void swap(complex_t *a, complex_t *b)
{
    complex_t c = *a;
    *a = *b;
    *b = c;
}

// shifts the sequence (a1, a2, a3, a4, a5) transforming it to
// (a5, a1, a2, a3, a4)
WITHIN_KERNEL void shift32(
    complex_t *a1, complex_t *a2, complex_t *a3, complex_t *a4, complex_t *a5)
{
    complex_t c1, c2;
    c1 = *a2;
    *a2 = *a1;
    c2 = *a3;
    *a3 = c1;
    c1 = *a4;
    *a4 = c2;
    c2 = *a5;
    *a5 = c1;
    *a1 = c2;
}

WITHIN_KERNEL void _fftKernel2(complex_t *a)
{
    complex_t c = a[0];
    a[0] = c + a[1];
    a[1] = c - a[1];
}
#define fftKernel2(a, direction) _fftKernel2(a)

WITHIN_KERNEL void _fftKernel2S(complex_t *d1, complex_t *d2)
{
    complex_t c = *d1;
    *d1 = c + *d2;
    *d2 = c - *d2;
}
#define fftKernel2S(d1, d2, direction) _fftKernel2S(d1, d2)

WITHIN_KERNEL void fftKernel4(complex_t *a, const int direction)
{
    fftKernel2S(a + 0, a + 2, direction);
    fftKernel2S(a + 1, a + 3, direction);
    fftKernel2S(a + 0, a + 1, direction);
    a[3] = conj_transp_and_mul(a[3], direction);
    fftKernel2S(a + 2, a + 3, direction);
    swap(a + 1, a + 2);
}

WITHIN_KERNEL void fftKernel4s(complex_t *a0, complex_t *a1,
    complex_t *a2, complex_t *a3, const int direction)
{
    fftKernel2S(a0, a2, direction);
    fftKernel2S(a1, a3, direction);
    fftKernel2S(a0, a1, direction);
    *a3 = conj_transp_and_mul(*a3, direction);
    fftKernel2S(a2, a3, direction);
    swap(a1, a2);
}

WITHIN_KERNEL void bitreverse8(complex_t *a)
{
    swap(a + 1, a + 4);
    swap(a + 3, a + 6);
}

WITHIN_KERNEL void fftKernel8(complex_t *a, const int direction)
{
    const complex_t w1  = complex_ctr(
        0.7071067811865475,
        0.7071067811865475 * direction);
    const complex_t w3  = complex_ctr(
        -0.7071067811865475,
        0.7071067811865475 * direction);
    fftKernel2S(a + 0, a + 4, direction);
    fftKernel2S(a + 1, a + 5, direction);
    fftKernel2S(a + 2, a + 6, direction);
    fftKernel2S(a + 3, a + 7, direction);
    a[5] = ${mul}(w1, a[5]);
    a[6] = conj_transp_and_mul(a[6], direction);
    a[7] = ${mul}(w3, a[7]);
    fftKernel2S(a + 0, a + 2, direction);
    fftKernel2S(a + 1, a + 3, direction);
    fftKernel2S(a + 4, a + 6, direction);
    fftKernel2S(a + 5, a + 7, direction);
    a[3] = conj_transp_and_mul(a[3], direction);
    a[7] = conj_transp_and_mul(a[7], direction);
    fftKernel2S(a + 0, a + 1, direction);
    fftKernel2S(a + 2, a + 3, direction);
    fftKernel2S(a + 4, a + 5, direction);
    fftKernel2S(a + 6, a + 7, direction);
    bitreverse8(a);
}


WITHIN_KERNEL INLINE void ${prefix}_generic(
        ${elem_ctype}* a,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        int thread_in_xform,
        int direction)
{
    fftKernel8(a, direction);

    // Twiddle kernel
    {
        const int angf = thread_in_xform ;
        %for i in range(1, 8):
            a[${i}] = ${mul}(
                a[${i}],
                cdata[${i} * 64 + angf]
                );
        %endfor
    }

    // Transpose kernel
    {
        const int i = (thread_in_xform / 8);
        const int j = (thread_in_xform % 8);

        const int lmem_store_idx = thread_in_xform;
        const int lmem_load_idx = j * 68 + i;

        %for component in ('x', 'y'):
            %for k in range(8):
               temp[lmem_store_idx + ${k} * 68] = a[${k}].${component};
            %endfor
            LOCAL_BARRIER;
            %for k in range(8):
               a[${k}].${component} = temp[lmem_load_idx + ${k} * 8];
            %endfor
            LOCAL_BARRIER;
        %endfor
    }

    fftKernel8(a, direction);

    // Twiddle kernel
    {
        const int angf = thread_in_xform / 8 * 8;
        %for i in range(1, 8):
            a[${i}] = ${mul}(
                a[${i}],
                cdata[${i} * 64 + angf]
                );
        %endfor
    }

    // Transpose kernel
    {

        const int i = (thread_in_xform % 8);
        const int j = (thread_in_xform / 8);

        const int lmem_store_idx = thread_in_xform;
        const VSIZE_T lmem_load_idx = j * 72 + i;

        %for component in ('x', 'y'):
            %for k in range(8):
               temp[lmem_store_idx + ${k} * 72] = a[${k}].${component};
            %endfor
            LOCAL_BARRIER;
            %for k in range(8):
               a[${k}].${component} = temp[lmem_load_idx + ${k} * 8];
            %endfor
            LOCAL_BARRIER;
        %endfor
    }

    fftKernel8(a, direction);
}


WITHIN_KERNEL INLINE void ${prefix}forward(
        ${elem_ctype}* r_out,
        ${elem_ctype}* r_in,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        int thread_in_xform)
{
    // Preprocess
    %for i in range(8):
    r_out[${i}] = ${mul}(
        r_in[${i}],
        cdata[512 + ${i} * 64 + thread_in_xform]
        );
    %endfor

    ${prefix}_generic(r_out, temp, cdata, thread_in_xform, -1);
}


WITHIN_KERNEL INLINE void ${prefix}inverse(
        ${elem_ctype}* r_out,
        ${elem_ctype}* r_in,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        int thread_in_xform)
{
    ${prefix}_generic(r_in, temp, cdata, thread_in_xform, 1);

    // Postprocess
    %for i in range(8):
    r_out[${i}] = ${mul}(
        conj(r_in[${i}]),
        cdata[512 + ${i} * 64 + thread_in_xform]
        );
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}forward_i32(
        ${elem_ctype}* r_out,
        ${i32}* r_in,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        int thread_in_xform)
{
    %for i in range(8):
    r_out[${i}] = complex_ctr(r_in[${i}], -r_in[${i + 8}]);
    %endfor
    ${prefix}forward(r_out, r_out, temp, cdata, thread_in_xform);
}


WITHIN_KERNEL INLINE ${elem_ctype} ${prefix}i32_to_elem(${i32} x, ${i32} y)
{
    return complex_ctr(x, -y);
}


WITHIN_KERNEL INLINE ${i32} ${prefix}f64_to_i32(double x)
{
    // The result is within the range of int64, so it must be first
    // converted to signed integer and then taken modulo 2^31
    return (${i32})((${dtypes.ctype(numpy.int64)})(round(x)));
}


WITHIN_KERNEL INLINE void ${prefix}inverse_i32(
        ${i32}* r_out,
        ${elem_ctype}* r_in,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        int thread_in_xform)
{
    ${prefix}inverse(r_in, r_in, temp, cdata, thread_in_xform);
    %for i in range(8):
    r_out[${i}] = ${prefix}f64_to_i32(r_in[${i}].x);
    r_out[${i + 8}] = ${prefix}f64_to_i32(r_in[${i}].y);
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}noop()
{
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
    LOCAL_BARRIER;
}


WITHIN_KERNEL INLINE void ${prefix}forward_i32_shared(
        LOCAL_MEM_ARG ${elem_ctype}* in_out,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        unsigned int thread_in_xform)
{
    ${elem_ctype} r[8];
    %for i in range(8):
    r[${i}] = in_out[${i * 64} + thread_in_xform];
    %endfor
    LOCAL_BARRIER;
    ${prefix}forward(r, r, temp, cdata, thread_in_xform);
    LOCAL_BARRIER;
    %for i in range(8):
    in_out[${i * 64} + thread_in_xform] = r[${i}];
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}inverse_i32_shared_add(
        LOCAL_MEM_ARG ${i32}* out,
        LOCAL_MEM_ARG ${elem_ctype}* in,
        LOCAL_MEM_ARG ${temp_ctype}* temp,
        ${prefix}CDATA_QUALIFIER ${cdata_ctype}* cdata,
        unsigned int thread_in_xform)
{
    ${elem_ctype} r[8];
    %for i in range(8):
    r[${i}] = in[${i * 64} + thread_in_xform];
    %endfor
    LOCAL_BARRIER;
    ${prefix}inverse(r, r, temp, cdata, thread_in_xform);
    LOCAL_BARRIER;
    %for i in range(8):
    out[${i * 64} + thread_in_xform] += ${prefix}f64_to_i32(r[${i}].x);
    out[${(i + 8) * 64} + thread_in_xform] += ${prefix}f64_to_i32(r[${i}].y);
    %endfor
}


WITHIN_KERNEL INLINE void ${prefix}noop_shared()
{
    LOCAL_BARRIER;
    ${prefix}noop();
    LOCAL_BARRIER;
}

</%def>
