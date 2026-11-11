"""Original tiny NineToothed-only prerequisites; Torch stays in the harness."""
import ninetoothed as nt
import ninetoothed.language as ntl


def dot_arrangement(lhs,rhs,output):
    out=output.tile((16,16))
    left=lhs.tile((16,16)).tile((1,-1)).expand((-1,out.shape[1]))
    left.dtype=left.dtype.squeeze(0)
    right=rhs.tile((16,16)).tile((-1,1)).expand((out.shape[0],-1))
    right.dtype=right.dtype.squeeze(1)
    return left,right,out


def dot_application(lhs,rhs,output):
    acc=ntl.zeros(output.shape,dtype=ntl.float32)
    for index in range(lhs.shape[0]):
        acc+=ntl.dot(lhs[index],rhs[index])
    output=acc.to(ntl.float16)


def reduction_arrangement(value,output):
    return value.tile((1,128)),output.tile((1,128))


def reduction_application(value,output):
    shifted=value-ntl.max(value,1)[:,None]
    numerator=ntl.exp(shifted)
    output=numerator/ntl.sum(numerator,1)[:,None]


def make_primitives():
    options=dict(num_warps=4,num_stages=1,max_num_configs=1)
    dot=nt.make(dot_arrangement,dot_application,
                tuple(nt.Tensor(2,dtype=nt.float16) for _ in range(3)),
                kernel_name='h3_gfx936_probe_dot',**options)
    reduction=nt.make(reduction_arrangement,reduction_application,
                (nt.Tensor(2,dtype=nt.float32,other=float('-inf')),nt.Tensor(2,dtype=nt.float32)),
                kernel_name='h3_gfx936_probe_reduction',**options)
    return dot,reduction
