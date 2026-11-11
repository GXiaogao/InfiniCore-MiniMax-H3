"""Public InfiniCore operator validation; no Triton execution in this process."""
import os
from common import write_json


def run(report,folder,result_path):
    import torch
    import numpy as np
    import infinicore as ic
    from infinicore.lib import _infinicore as native
    import jit_checks
    device=ic.device('cuda',0)
    if not torch.version.hip or device._underlying.type!=type(device._underlying).Type.HYGON or ic.use_ntops:
        raise RuntimeError('Expected native HYGON dispatcher')
    ic.set_device(device)
    types={torch.float32:ic.float32,torch.float16:ic.float16,torch.bfloat16:ic.bfloat16}
    def wrap(t):
        return ic.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),dtype=types[t.dtype],device=device if t.is_cuda else ic.device('cpu'))
    def count():return native.gated_residual_hygon_launch_count()
    initial=count()
    def launch(r,g,b,out):
        torch.cuda.synchronize()
        # Compare the actual original framework operators on the SAME device inputs,
        # including before an in-place update can overwrite residual storage.
        ir,ig,ib,io=wrap(r),wrap(g),wrap(b),wrap(out)
        old=ic.add(ir,ic.mul(ig,ib))
        old_owner=torch.empty_like(out)
        wrap(old_owner).copy_(old)
        ic.sync_device();torch.cuda.synchronize()
        before=count()
        ic.gated_residual(ir,ig,ib,out=io)
        ic.sync_device();torch.cuda.synchronize()
        if count()-before!=1:raise AssertionError('Public operator did not launch exactly one NineToothed kernel')
        if not torch.equal(out,old_owner):raise AssertionError('NineToothed result differs from native InfiniCore mul+add')
    def factory(handle,folder):
        return launch,{'route':'InfiniCore GatedResidual -> HYGON -> NineToothed HSACO'}
    import sys
    # Public-op tests need dtype labels only; they never construct a JIT handle.
    jit_checks.run_gated(report,folder,result_path,launch_factory=factory,make_handle=lambda dtype:dtype)
    for dtype_name in jit_checks.DTYPES:
        dtype=getattr(torch,dtype_name)
        for layout in ('inplace','batch1'):
            shape=(3,129) if layout=='inplace' else (1,3,65)
            host=[torch.randn(shape).to(dtype) for _ in range(3)]
            r,g,b=[x.cuda() for x in host];out=r if layout=='inplace' else torch.empty_like(r)
            from rounding_reference import torch_add
            expected=torch_add(host[0],(host[1].float()*host[2].float()).to(dtype),dtype)
            launch(r,g,b,out)
            error=jit_checks.check_values(out,expected)
            if not torch.equal(g.cpu(),host[1]) or not torch.equal(b.cpu(),host[2]):raise AssertionError('Input modified')
            if layout!='inplace' and not torch.equal(r.cpu(),host[0]):raise AssertionError('Residual modified')
            name=dtype_name+'_'+layout
            np.savez(folder/(name+'.npz'),actual=out.float().cpu().numpy(),expected=expected.float().numpy())
            report['cases'].append({'name':name,'passed':True,'error':error})
            print('PASS',name,error,flush=True);write_json(result_path,report)
    r,g,b=[torch.randn(3,65,device='cuda') for _ in range(3)];out=torch.zeros_like(r)
    base=torch.randn(3,66,device='cuda');one=torch.zeros(1,65,device='cuda')
    four=[torch.randn(1,1,3,65,device='cuda') for _ in range(4)]
    batch=[torch.randn(2,3,65,device='cuda') for _ in range(4)]
    rejects=[('shape',(r,g[:,:64],b,out)),('dtype',(r,g.half(),b,out)),('device',(r,g.cpu(),b,out)),
             ('rank4',four),('batch2',batch),('overlap_output',(r,g,b,one.expand(3,65))),
             ('partial_alias',(base[:,:65],g,b,base[:,1:])),('branch_alias',(r,g,b,b))]
    for name,tensors in rejects:
        args=[wrap(x) for x in tensors];before=count();original=[x.clone() for x in tensors]
        try:ic.gated_residual(*args[:3],out=args[3])
        except (ValueError,RuntimeError) as exc:message=str(exc)
        else:raise AssertionError('Invalid public input accepted: '+name)
        if count()!=before:raise AssertionError('Rejected input launched a kernel')
        for x,y in zip(tensors,original):
            if not torch.equal(x,y):raise AssertionError('Rejected input modified storage')
        report['cases'].append({'name':'reject_'+name,'passed':True,'error_message':message})
        print('PASS reject_'+name,flush=True)
    report['native_launch_count']=count()-initial
    if report['native_launch_count']!=36:raise AssertionError('Expected 36 successful public operator launches')
    report['backend']=native.gated_residual_hygon_backend_info()
    report['numeric_cases']=36;report['rejection_cases']=8
    report['native_mul_add_comparisons']=36
    if any(name in sys.modules for name in ('triton','ninetoothed')):
        raise AssertionError('Public runtime unexpectedly loaded Python GPU compiler')
    write_json(result_path,report)
