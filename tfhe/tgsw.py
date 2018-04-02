import numpy

from .tlwe import *


class TGswParams:

    def __init__(self, l: int, Bgbit: int, tlwe_params: TLweParams):

        Bg = 1 << Bgbit
        halfBg = Bg // 2

        h = Torus32(1) << (32 - numpy.arange(1, l+1) * Bgbit) # 1/(Bg^(i+1)) as a Torus32

        # offset = Bg/2 * (2^(32-Bgbit) + 2^(32-2*Bgbit) + ... + 2^(32-l*Bgbit))
        offset = int64_to_int32(sum(1 << (32 - numpy.arange(1, l+1) * Bgbit)) * halfBg)

        self.l = l # decomp length
        self.Bgbit = Bgbit # log_2(Bg)
        self.Bg = Bg # decomposition base (must be a power of 2)
        self.halfBg = halfBg # Bg/2
        self.maskMod = Bg - 1 # Bg-1
        self.tlwe_params = tlwe_params # Params of each row
        self.kpl = (tlwe_params.k + 1) * l # number of rows = (k+1)*l
        self.h = h # powers of Bgbit
        self.offset = offset # offset = Bg/2 * (2^(32-Bgbit) + 2^(32-2*Bgbit) + ... + 2^(32-l*Bgbit))


class TGswKey:

    def __init__(self, rng, params: TGswParams):
        tlwe_key = TLweKey(rng, params.tlwe_params)
        self.params = params # the parameters
        self.tlwe_params = params.tlwe_params # the tlwe params of each rows
        self.tlwe_key = tlwe_key


class TGswSampleArray:

    def __init__(self, params: TGswParams, shape):
        self.k = params.tlwe_params.k
        self.l = params.l
        self.samples = TLweSampleArray(params.tlwe_params, shape + (self.k + 1, self.l))


class TGswSampleFFTArray:

    def __init__(self, params: TGswParams, shape):
        self.k = params.tlwe_params.k
        self.l = params.l
        self.samples = TLweSampleFFTArray(params.tlwe_params, shape + (self.k + 1, self.l))


# Result += mu*H, mu integer
def tGswAddMuIntH(result: TGswSampleArray, messages, params: TGswParams):
    # TYPING: messages::Array{Int32, 1}

    k = params.tlwe_params.k
    l = params.l
    h = params.h

    # compute result += H

    # returns an underlying coefsT of TorusPolynomialArray, with the total size
    # (N, k + 1 [from TLweSample], l, k + 1 [from TGswSample], n)
    # messages: (n,)
    # h: (l,)
    # TODO: use an appropriate method
    # TODO: not sure if it's possible to fully vectorize it
    for bloc in range(k+1):
        result.samples.a.coefsT[:, bloc, :, bloc, 0] += (
            messages.reshape(messages.size, 1) * h.reshape(1, l))


# Result = tGsw(0)
def tGswEncryptZero(rng, result: TGswSampleArray, alpha: float, key: TGswKey):
    rlkey = key.tlwe_key
    tLweSymEncryptZero(rng, result.samples, alpha, rlkey)

# encrypts a constant message
def tGswSymEncryptInt(rng, result: TGswSampleArray, messages, alpha: float, key: TGswKey):
    # TYPING: messages::Array{Int32, 1}
    tGswEncryptZero(rng, result, alpha, key)
    tGswAddMuIntH(result, messages, key.params)


def tGswTorus32PolynomialDecompH(
        result: IntPolynomialArray, sample: TorusPolynomialArray, params: TGswParams):

    # GPU: array operations or (more probably) a custom kernel

    N = params.tlwe_params.N
    l = params.l
    k = params.tlwe_params.k
    Bgbit = params.Bgbit

    maskMod = params.maskMod
    halfBg = params.halfBg
    offset = params.offset

    decal = lambda p: 32 - p * Bgbit

    ps = numpy.arange(1, l+1).reshape(1, 1, l, 1)
    sample_coefs = sample.coefsT.reshape(sample.shape + (1, N))

    # do the decomposition
    result.coefs[:,:,:,:] = (((sample_coefs + offset) >> decal(ps)) & maskMod) - halfBg


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tGswToFFTConvert(result: TGswSampleFFTArray, source: TGswSampleArray, params: TGswParams):
    tLweToFFTConvert(result.samples, source.samples, params.tlwe_params)


def tLweFFTAddMulRTo(res, a, b, bk_idx):
    # GPU: array operations or (more probably) a custom kernel

    ml, kplus1, Ndiv2 = res.shape
    l = a.shape[-2]

    d = a.reshape(ml, kplus1, l, 1, Ndiv2)
    for i in range(kplus1):
        for j in range(l):
            res += d[:,i,j,:,:] * b[bk_idx,i,j,:,:]



# External product (*): accum = gsw (*) accum
def tGswFFTExternMulToTLwe(
        accum: TLweSampleArray, gsw: TGswSampleFFTArray, bk_idx, params: TGswParams,
        tmpa: TLweSampleFFTArray, deca: IntPolynomialArray, decaFFT: LagrangeHalfCPolynomialArray):

    tlwe_params = params.tlwe_params
    k = tlwe_params.k
    l = params.l
    kpl = params.kpl
    N = tlwe_params.N

    tGswTorus32PolynomialDecompH(deca, accum.a, params)

    ip_ifft_(decaFFT, deca)

    tLweFFTClear(tmpa, tlwe_params)

    res = tmpa.a.coefsC
    a = decaFFT.coefsC
    b = gsw.samples.a.coefsC

    tLweFFTAddMulRTo(res, a, b, bk_idx)

    tLweFromFFTConvert(accum, tmpa, tlwe_params)
