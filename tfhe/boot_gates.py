from .keys import *


#*#*****************************************
# zones on the torus -> to see
#*#*****************************************

"""
 * Homomorphic bootstrapped NAND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_NAND_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,1/8) - ca - cb
    NandConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, NandConst, in_out_params)
    lweSubTo(temp_result, ca, in_out_params)
    lweSubTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped OR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_OR_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,1/8) + ca + cb
    OrConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrConst, in_out_params)
    lweAddTo(temp_result, ca, in_out_params)
    lweAddTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped AND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_AND_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,-1/8) + ca + cb
    AndConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndConst, in_out_params)
    lweAddTo(temp_result, ca, in_out_params)
    lweAddTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped XOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_XOR_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,1/4) + 2*(ca + cb)
    XorConst = modSwitchToTorus32(1, 4)
    lweNoiselessTrivial(temp_result, XorConst, in_out_params)
    lweAddMulTo(temp_result, 2, ca, in_out_params)
    lweAddMulTo(temp_result, 2, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped XNOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_XNOR_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,-1/4) + 2*(-ca-cb)
    XnorConst = modSwitchToTorus32(-1, 4)
    lweNoiselessTrivial(temp_result, XnorConst, in_out_params)
    lweSubMulTo(temp_result, 2, ca, in_out_params)
    lweSubMulTo(temp_result, 2, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped NOT gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_NOT_(bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray):
    in_out_params = bk.params.in_out_params
    lweNegate(result, ca, in_out_params)


"""
 * Homomorphic bootstrapped COPY gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_COPY_(bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray):
    in_out_params = bk.params.in_out_params
    lweCopy(result, ca, in_out_params)


"""
 * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
 * Takes a boolean value)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_CONSTANT_(bk: TFHECloudKey, result: LweSampleArray, vals):
    in_out_params = bk.params.in_out_params
    MU = modSwitchToTorus32(1, 8)
    if isinstance(vals, numpy.ndarray):
        mus = numpy.array([MU if x else -MU for x in vals])
    else:
        mus = numpy.ones(result.shape, Torus32) * (MU if vals else -MU)
    lweNoiselessTrivial(result, mus, in_out_params)


"""
 * Homomorphic bootstrapped NOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_NOR_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,-1/8) - ca - cb
    NorConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, NorConst, in_out_params)
    lweSubTo(temp_result, ca, in_out_params)
    lweSubTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped AndNY Gate: not(a) and b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_ANDNY_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,-1/8) - ca + cb
    AndNYConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndNYConst, in_out_params)
    lweSubTo(temp_result, ca, in_out_params)
    lweAddTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped AndYN Gate: a and not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_ANDYN_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,-1/8) + ca - cb
    AndYNConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndYNConst, in_out_params)
    lweAddTo(temp_result, ca, in_out_params)
    lweSubTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped OrNY Gate: not(a) or b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_ORNY_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,1/8) - ca + cb
    OrNYConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrNYConst, in_out_params)
    lweSubTo(temp_result, ca, in_out_params)
    lweAddTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped OrYN Gate: a or not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_ORYN_(
        bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    #compute: (0,1/8) + ca - cb
    OrYNConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrYNConst, in_out_params)
    lweAddTo(temp_result, ca, in_out_params)
    lweSubTo(temp_result, cb, in_out_params)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


"""
 * Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
 * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def tfhe_gate_MUX_(
        bk: TFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, c: LweSampleArray):

    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params
    extracted_params = bk.params.tgsw_params.tlwe_params.extracted_lweparams

    temp_result = LweSampleArray(in_out_params, result.shape)
    temp_result1 = LweSampleArray(extracted_params, result.shape)
    u1 = LweSampleArray(extracted_params, result.shape)
    u2 = LweSampleArray(extracted_params, result.shape)

    #compute "AND(a,b)": (0,-1/8) + a + b
    AndConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndConst, in_out_params)
    lweAddTo(temp_result, a, in_out_params)
    lweAddTo(temp_result, b, in_out_params)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u1, bk.bkFFT, MU, temp_result)

    #compute "AND(not(a),c)": (0,-1/8) - a + c
    lweNoiselessTrivial(temp_result, AndConst, in_out_params)
    lweSubTo(temp_result, a, in_out_params)
    lweAddTo(temp_result, c, in_out_params)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u2, bk.bkFFT, MU, temp_result)

    # Add u1=u1+u2
    MuxConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result1, MuxConst, extracted_params)
    lweAddTo(temp_result1, u1, extracted_params)
    lweAddTo(temp_result1, u2, extracted_params)

    # Key switching
    lweKeySwitch(result, bk.bkFFT.ks, temp_result1)
