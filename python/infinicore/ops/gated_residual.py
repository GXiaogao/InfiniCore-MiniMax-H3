from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gated_residual(residual, gate, branch, *, out=None):
    """HYGON: residual + cast_dtype(gate * branch), via the native dispatcher."""
    inputs = (residual._underlying, gate._underlying, branch._underlying)
    if out is None:
        return Tensor(_infinicore.gated_residual(*inputs))
    _infinicore.gated_residual_(out._underlying, *inputs)
    return out
