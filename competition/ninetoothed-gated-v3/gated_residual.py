"""Original NineToothed expressions for H3 gate*branch + residual.

Gate/branch multiplication rounds to the input dtype before residual addition.
The fixed baseline uses FP32 __fadd_rd; emulate that directed addition using
NineToothed expressions. The launcher must disable FP fusion/reassociation.
Tensor storage, references and checks live in verify.py, not these expressions.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(residual, gate, branch, output):
    return (residual.tile((1, 128)), gate.tile((1, 128)),
            branch.tile((1, 128)), output.tile((1, 128)))


def fp32(residual, gate, branch, output):
    a = ntl.cast(residual, ntl.float32)
    product = ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32)
    rounded = a + product
    # TwoSum recovers the exact addition's residual when the rounded sum is finite.
    part = rounded - a
    error = (a - (rounded - part)) + (product - part)
    bits = ntl.cast(rounded, ntl.int32, bitcast=True)
    step = ntl.cast(ntl.where(rounded < 0, 1, -1), ntl.int32)
    previous = ntl.cast(bits + step, ntl.float32, bitcast=True)
    directed = ntl.where(error < 0, previous, rounded)
    # Positive finite overflow rounds down to max finite, negative overflow to -inf.
    finite_inputs = (ntl.abs(a) <= 3.4028234663852886e38) & (ntl.abs(product) <= 3.4028234663852886e38)
    finite_corrected = ntl.where((rounded > 3.4028234663852886e38) & finite_inputs, 3.4028234663852886e38, directed)
    # Directed cancellation gives -0; +0 + +0 is the sole positive-zero case.
    signs = ntl.cast(a, ntl.int32, bitcast=True) | ntl.cast(product, ntl.int32, bitcast=True)
    negative_zero = (rounded == 0) & ((a != 0) | (product != 0) | (signs < 0))
    # Triton 3.8 scalar_constant treats the Python literal -0.0 as null (+0).
    # Keep the selection in integer space so that bit 31 survives lowering.
    directed_bits = ntl.cast(finite_corrected, ntl.int32, bitcast=True)
    output_bits = ntl.where(negative_zero, ntl.cast(-2147483648, ntl.int32), directed_bits)
    output = ntl.cast(output_bits, ntl.float32, bitcast=True)


def fp16(residual, gate, branch, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.float16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.float16)


def bf16(residual, gate, branch, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)


def make(dtype_name):
    dtype, application = {'float32': (nt.float32, fp32), 'float16': (nt.float16, fp16),
                          'bfloat16': (nt.bfloat16, bf16)}[dtype_name]
    return nt.make(arrangement, application, tuple(Tensor(2, dtype=dtype) for _ in range(4)),
                   kernel_name=f'h3_gated_residual_{dtype_name}', num_warps=4, num_stages=1,
                   max_num_configs=1)


def add_arrangement(left, right, output):
    return left.tile((128,)), right.tile((128,)), output.tile((128,))


def add_application(left, right, output):
    output = left + right


def make_add(dtype_name):
    return nt.make(add_arrangement, add_application,
                   tuple(Tensor(1, dtype=getattr(nt, dtype_name)) for _ in range(3)),
                   kernel_name=f'h3_probe_add_{dtype_name}', num_warps=4, num_stages=1,
                   max_num_configs=1)
