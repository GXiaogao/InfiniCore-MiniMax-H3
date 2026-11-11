"""NineToothed-only diagnostic kernels. Not a replacement attention implementation."""
import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(value,output):
    return value.tile((128,)),output.tile((128,))


def direct(value,output):
    loaded=value
    output=loaded.to(ntl.bfloat16)


def via_float32(value,output):
    loaded=value
    output=loaded.to(ntl.float32).to(ntl.bfloat16)


def make(name,input_dtype,intermediate=False):
    return nt.make(arrangement,via_float32 if intermediate else direct,
        (nt.Tensor(1,dtype=input_dtype),nt.Tensor(1,dtype=nt.bfloat16)),
        caller='torch',kernel_name=name,num_warps=4,num_stages=1,max_num_configs=1)
