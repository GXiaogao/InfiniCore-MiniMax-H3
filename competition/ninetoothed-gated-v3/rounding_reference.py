"""CPU-only oracle; float64 compensated sum is independent of the FP32 DSL."""
import numpy as np


def add_down(left, right):
    a,b=np.broadcast_arrays(np.asarray(left,dtype=np.float32),np.asarray(right,dtype=np.float32))
    x,y=a.astype(np.float64),b.astype(np.float64)
    with np.errstate(all='ignore'):
        high=x+y
        # FastTwoSum ordered by magnitude retains tails even across huge exponent gaps.
        low=np.where(np.abs(x)>=np.abs(y),(x-high)+y,(y-high)+x)
        rounded=high.astype(np.float32)
        back=rounded.astype(np.float64)
        lower=(back>high)|((back==high)&(low<0))
        result=np.where(lower,np.nextafter(rounded,np.float32(-np.inf)),rounded)
        minus_zero=(result==0)&((a!=0)|(b!=0)|np.signbit(a)|np.signbit(b))
        return np.where(minus_zero,np.float32(-0.0),result).astype(np.float32)


def torch_add(left, right, dtype):
    """Used only for CPU test references. Multiplication rounding happens upstream."""
    import torch
    if left.device.type!='cpu' or right.device.type!='cpu':
        raise ValueError('Reference must run on CPU')
    if dtype==torch.float32:
        return torch.from_numpy(add_down(left.float().numpy(),right.float().numpy()))
    return (left.float()+right.float()).to(dtype)
