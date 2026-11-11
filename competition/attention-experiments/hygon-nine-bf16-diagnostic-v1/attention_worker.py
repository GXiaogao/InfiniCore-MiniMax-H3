#!/usr/bin/env python3
"""Actual NineToothed Attention JIT. All Torch math below is independent testing."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from attention_contract import (SCOPE,case_specs,causal_key_limit,sha256,
                                validate_report,verify_candidate_source,verify_upstream,
                                verify_candidate_attention,write_json)

HERE=Path(__file__).resolve().parent


def save_generated(handle,name,run_dir):
    source=Path(handle._source)
    destination=run_dir/'generated'/f'{name}.py'
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    return dict(name=name,path=str(destination.relative_to(run_dir)),sha256=sha256(destination))


def check_environment(args):
    integrity=verify_candidate_source(args.candidate)
    import torch
    import numpy as np
    import triton
    import ninetoothed as nt
    if any(os.environ.get(k) for k in ('HSA_OVERRIDE_GFX_VERSION','TRITON_OVERRIDE_ARCH')):
        raise RuntimeError('unexpected architecture override')
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    actual=dict(torch_version=torch.__version__,torch_file=torch.__file__,hip=torch.version.hip)
    if any(actual.get(k)!=baseline.get(k) for k in actual):
        raise RuntimeError('isolated worker changed base Torch identity')
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('usable HIP Torch required')
    if torch.__version__.split('+')[0]!='2.9.0':
        raise RuntimeError('fixed vendor Torch2.9.0 required')
    origins={}
    for name,version,root in [('ninetoothed','0.26.0',args.candidate/'site-packages'),
                            ('numpy','1.26.4',args.candidate/'site-packages'),
                            ('triton','3.5.1+das.opt1.dtk2604.torch290',args.vendor)]:
        dist=metadata.distribution(name)
        imported={'ninetoothed':nt,'numpy':np,'triton':triton}[name]
        if (dist.version!=version or not Path(dist.locate_file('')).resolve().is_relative_to(root.resolve())
                or not Path(imported.__file__).resolve().is_relative_to(root.resolve())):
            raise RuntimeError('unexpected package version/origin: '+name)
        origins[name]=dict(version=dist.version,file=imported.__file__)
    torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    target=triton.runtime.driver.active.get_current_target()
    arch=str(getattr(props,'gcnArchName','')).split(':')[0]
    if arch!='gfx936' or str(target.arch).split(':')[0]!=arch or target.backend not in ('hip','hcu'):
        raise RuntimeError(f'expected true gfx936 HIP target, got {props}, {target}')
    torch.backends.cuda.matmul.allow_tf32=False
    return dict(base_torch=actual,packages=origins,device=str(props),candidate_integrity=integrity,
                target=dict(backend=target.backend,arch=arch,warp_size=target.warp_size))


def cpu_reference(q,k,v,case):
    """Uniform mathematical reference; never returned as candidate output."""
    import torch
    q,k,v=(t.detach().cpu().double() for t in (q,k,v))
    group=case['hq']//case['hkv']
    k=k.repeat_interleave(group,dim=1)
    v=v.repeat_interleave(group,dim=1)
    scores=torch.matmul(q,k.transpose(-1,-2))*case['scale']
    if case['causal']:
        limits=torch.tensor([causal_key_limit(i,case['sq'],case['sk']) for i in range(case['sq'])])
        mask=torch.arange(case['sk'])[None,:]>=limits[:,None]
        scores.masked_fill_(mask,float('-inf'))
    return torch.matmul(torch.softmax(scores,dim=-1),v)


def guarded_tensor(batch,seq,heads,dim,dtype,layout,fill):
    import torch
    if layout=='bsnd':
        elements=batch*seq*heads*dim
        backing=torch.full((elements+2,),fill,dtype=dtype,device='cuda')
        bsnd=backing[1:-1].view(batch,seq,heads,dim)
        mask=torch.ones(backing.shape,dtype=torch.bool)
        mask[1:-1]=False
    else:
        backing=torch.full((batch,seq,heads,dim+8),fill,dtype=dtype,device='cuda')
        bsnd=backing[...,1:1+dim]
        mask=torch.ones(backing.shape,dtype=torch.bool)
        mask[...,1:1+dim]=False
    bhds=bsnd.permute(0,2,1,3)
    return backing,bhds,mask


def verify_case(handle,case,run_dir):
    import torch
    dtype=getattr(torch,case['dtype'])
    args=[guarded_tensor(case['batch'],length,heads,case['dim'],dtype,case['layout'],-19)
          for length,heads in ((case['sq'],case['hq']),(case['sk'],case['hkv']),(case['sk'],case['hkv']))]
    output_backing,output,output_guard=guarded_tensor(case['batch'],case['sq'],case['hq'],
                                                    case['dim'],dtype,case['layout'],-23)
    kv_len=torch.tensor([case['sk']],dtype=torch.int32,device='cuda')
    unused_mask=torch.zeros((1,1,1,1),dtype=dtype,device='cuda')
    kv_snapshot=kv_len.cpu().clone()
    mask_snapshot=unused_mask.cpu().clone()
    stream=torch.cuda.Stream(device=0)
    metrics=[]
    atol,rtol=(0.01,0.01) if case['dtype']=='float16' else (0.025,0.025)
    for repeat in range(2):
        generator=torch.Generator(device='cpu').manual_seed(20261110+repeat*1009+sum(map(ord,case['name'])))
        with torch.cuda.stream(stream):
            for backing,tensor,guard in args:
                tensor.copy_(torch.randn(tuple(tensor.shape),generator=generator,dtype=torch.float32).to(dtype))
            output.fill_(float('nan'))
        stream.synchronize()
        snapshots=[backing.cpu().clone() for backing,_,_ in args]
        q,k,v=(item[1] for item in args)
        expected=cpu_reference(q,k,v,case)
        print(f'case={case["name"]} round={repeat+1} status=JIT_OR_RUN',flush=True)
        started=time.monotonic()
        with torch.cuda.stream(stream):
            # premake returns 14 arrangement inputs; its no-cache application has 10 runtime arguments.
            handle(q,k,v,kv_len,unused_mask,case['causal'],case['scale'],output,False,2)
        stream.synchronize()
        actual=output.cpu().double()
        finite=bool(torch.isfinite(actual).all())
        difference=(actual-expected).abs()
        good=finite and bool(torch.all(difference<=atol+rtol*expected.abs()))
        inputs_unchanged=all(torch.equal(backing.cpu(),before) for (backing,_,_),before in zip(args,snapshots))
        inputs_unchanged=inputs_unchanged and torch.equal(kv_len.cpu(),kv_snapshot) and torch.equal(unused_mask.cpu(),mask_snapshot)
        guards_unchanged=bool(torch.all(output_backing.cpu()[output_guard]==-23))
        guards_unchanged=guards_unchanged and all(bool(torch.all(backing.cpu()[guard]==-19)) for backing,_,guard in args)
        if not (good and inputs_unchanged and guards_unchanged):
            torch.save(dict(actual=actual,expected=expected,case=case),run_dir/(case['name']+'-failed.pt'))
            raise AssertionError(f'{case["name"]}: finite={finite}, numerical={good}, inputs={inputs_unchanged}, guards={guards_unchanged}')
        metrics.append(dict(round=repeat+1,max_abs=float(difference.max()),
                            relative_l2=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-30)),
                            elapsed_seconds_including_possible_jit=time.monotonic()-started))
        print(f'case={case["name"]} round={repeat+1} status=PASS max_abs={metrics[-1]["max_abs"]:.8g}',flush=True)
    return dict(**case,passed=True,rounds=2,finite=True,inputs_unchanged=True,guards_unchanged=True,
                atol=atol,rtol=rtol,tolerance_scope='development-only-not-official-threshold',
                metrics=metrics,strides=dict(q=list(q.stride()),k=list(k.stride()),v=list(v.stride()),out=list(output.stride())))


def run_primitives(report,args):
    import torch
    from probe_primitives import make_primitives
    dot,reduction=make_primitives()
    report['generated_sources'] += [save_generated(dot,'dot',args.run_dir),save_generated(reduction,'reduction',args.run_dir)]
    generator=torch.Generator().manual_seed(20261110)
    left=torch.randn(32,32,generator=generator).half()
    right=torch.randn(32,32,generator=generator).half()
    actual=torch.full((32,32),float('nan'),dtype=torch.float16,device='cuda')
    dot(left.cuda(),right.cuda(),actual)
    torch.cuda.synchronize()
    expected=left.double()@right.double()
    if not torch.allclose(actual.cpu().double(),expected,atol=0.02,rtol=0.01):
        raise AssertionError('NineToothed dot prerequisite mismatch')
    report['primitives']['dot']=True
    write_json(args.run_dir/'attention-probe.json',report)
    print('primitive=dot status=PASS',flush=True)
    value=torch.randn(3,65,generator=generator)
    output=torch.full_like(value.cuda(),float('nan'))
    reduction(value.cuda(),output)
    torch.cuda.synchronize()
    if not torch.allclose(output.cpu().double(),torch.softmax(value.double(),dim=-1),atol=1e-5,rtol=1e-4):
        raise AssertionError('NineToothed reduction prerequisite mismatch')
    report['primitives']['reduction']=True
    print('primitive=reduction status=PASS',flush=True)


def collect_compiler_assets(run_dir):
    root=Path(os.environ['TRITON_CACHE_DIR']).resolve()
    destination=run_dir/'compiled'
    total=0
    records=[]
    truncated=False
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in ('.hsaco','.json','.ttir','.ttgir','.llir','.amdgcn'):
            continue
        if total+path.stat().st_size>64*1024*1024 or len(records)>=500:
            truncated=True
            break
        target=destination/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        total+=target.stat().st_size
        records.append(dict(path=str(target.relative_to(run_dir)),bytes=target.stat().st_size,sha256=sha256(target)))
    return dict(files=records,total_bytes=total,truncated=truncated)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','vendor','baseline','run-dir'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    report=dict(result='FAILED',last_stage='environment',validation_scope=SCOPE,cases=[],
                primitives=dict(dot=False,reduction=False),aot_status='NOT_TESTED',
                integration_status='NOT_TESTED',performance_measurement_valid=False,generated_sources=[])
    def save():write_json(args.run_dir/'attention-probe.json',report)
    try:
        save()
        report.update(check_environment(args))
        report['upstream']=verify_upstream(HERE/'upstream_attention.py')
        report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        spec=importlib.util.spec_from_file_location('h3_candidate_nine_attention',HERE/'candidate_attention.py')
        upstream=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        report['last_stage']='primitives';save()
        run_primitives(report,args)
        import ninetoothed as nt
        kernels={}
        for case in case_specs():
            report['last_stage']=case['name'];save()
            key=(case['dtype'],case['dim'],case['causal'])
            if key not in kernels:
                arrangement,application,tensors=upstream.premake(with_kv_cache=False,
                    emb_dim=case['dim'],is_causal=case['causal'],with_attn_mask=False,
                    causal_variant=upstream.CausalVariant.LOWER_RIGHT,dtype=getattr(nt,case['dtype']),
                    block_size_m=32,block_size_n=32)
                label=f'attention_{case["dtype"]}_d{case["dim"]}_causal{int(case["causal"])}'
                kernels[key]=nt.make(arrangement,application,tensors,caller='torch',kernel_name=label,
                                     num_warps=4,num_stages=1,max_num_configs=1)
                report['generated_sources'].append(save_generated(kernels[key],label,args.run_dir))
            report['cases'].append(verify_case(kernels[key],case,args.run_dir));save()
        report['compiler_assets']=collect_compiler_assets(args.run_dir)
        if not any(record['path'].endswith('.hsaco') for record in report['compiler_assets']['files']):
            raise RuntimeError('no HIP hsaco compilation evidence collected')
        report.update(result='JIT_PROBE_PASSED',last_stage='complete')
        validate_report(report)
        return 0
    except Exception as error:
        report.update(result='FAILED',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1
    finally:
        if 'compiler_assets' not in report:
            try:report['compiler_assets']=collect_compiler_assets(args.run_dir)
            except Exception as error:report['compiler_asset_collection_error']=str(error)
        save()


if __name__=='__main__':raise SystemExit(main())
