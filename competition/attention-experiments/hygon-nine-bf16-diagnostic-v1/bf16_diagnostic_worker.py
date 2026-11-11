#!/usr/bin/env python3
"""Observe cast paths and JIT/AOT on identical fresh inputs; never substitutes reference output."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import traceback

from attention_contract import case_specs,sha256,verify_candidate_attention,write_json
from attention_worker import check_environment,guarded_tensor,cpu_reference,save_generated,collect_compiler_assets
from aot_contract import validate_abi,validate_dynamic_lds,verify_export_files,launch_schema
from aot_native import Adapter,hip_runtime
from bf16_diagnostic_contract import validate_diagnostic,json_preview

HERE=Path(__file__).resolve().parent


def metrics(actual,expected,atol,rtol):
    import torch
    a,e=actual.detach().cpu().double(),expected.detach().cpu().double()
    d=(a-e).abs();finite=bool(torch.isfinite(a).all())
    # Test data finite; report malformed outputs explicitly without non-JSON NaN metrics.
    return dict(passed=finite and bool(torch.all(d<=atol+rtol*e.abs())),finite=finite,
                max_abs=float(d.max()) if finite else None,
                relative_l2=float(torch.linalg.vector_norm(a-e)/torch.linalg.vector_norm(e).clamp_min(1e-30)) if finite else None)


def conversions(report,args):
    import torch
    import ninetoothed as nt
    import numpy as np
    from bf16_cast_dsl import make
    values=torch.tensor([0.,1.,-1.,.125,-.125,1.4426950408889634,-1.4426950408889634,.18,-.18,3.75,-3.75],dtype=torch.float64)
    values=torch.cat([values,torch.randn(245,generator=torch.Generator().manual_seed(8761),dtype=torch.float64)*.18])
    stream=torch.cuda.Stream()
    for name,dtype,via in [('fp64-direct',nt.float64,False),('fp64-via-fp32',nt.float64,True),('fp32-direct',nt.float32,False)]:
        report['last_stage']='conversion-'+name
        write_json(args.run_dir/'bf16-diagnostic.json',report)
        handle=make('nine_cast_'+name.replace('-','_'),dtype,via)
        record=save_generated(handle,name,args.run_dir)
        x=values.to(torch.float32 if name=='fp32-direct' else torch.float64).cuda()
        backing=torch.full((values.numel()+2,),-23,dtype=torch.bfloat16,device='cuda')
        y=backing[1:-1];y.fill_(float('nan'))
        torch.cuda.synchronize()
        before=x.cpu().clone()
        with torch.cuda.stream(stream):handle(x,y)
        stream.synchronize()
        expected=(x.cpu().float() if via else x.cpu()).to(torch.bfloat16).double()
        observation=dict(name=name,generated=record,**metrics(y,expected,0,0))
        observation['first_actual']=json_preview(y.cpu().double()[:11].tolist())
        observation['first_expected']=json_preview(expected[:11].tolist())
        observation['inputs_unchanged']=bool(torch.equal(x.cpu(),before))
        observation['guards_unchanged']=bool(torch.all(backing.cpu()[[0,-1]]==-23))
        np.savez(args.run_dir/(name+'.npz'),input=x.cpu().double().numpy(),actual=y.cpu().double().numpy(),expected=expected.numpy())
        report['conversion'].append(observation)
        write_json(args.run_dir/'bf16-diagnostic.json',report)
        print('conversion='+name+' '+json.dumps(observation),flush=True)
        if not observation['inputs_unchanged'] or not observation['guards_unchanged']:raise ValueError('conversion damaged memory')


def attention_pair(report,args):
    import torch
    import ninetoothed as nt
    import numpy as np
    old=Path(os.environ['H3_BF16_OLD_RUN'])
    if sha256(old/'attention-export.json')!='7425c0b7c5a21f856ccb8c90effec11b84be5757b42b989033e660fd8e4bfabf':raise ValueError('previous export report identity differs')
    source=json.loads((old/'attention-export.json').read_text())
    if source.get('result')!='AOT_EXPORT_COMPLETE':raise ValueError('previous six-kernel export incomplete')
    root=old/'exports';verify_export_files(root,source['files'])
    info=next(i for i in source['exports'] if (i['dtype'],i['head_dim'],i['causal'])==('bfloat16',64,False))
    folder=root/info['folder'];verify_export_files(folder,info['files'])
    schema=json.loads((folder/'launch-schema.json').read_text())
    regenerated=launch_schema((folder/'ninetoothed-generated.py').read_text(),info['label'],'bfloat16',False);regenerated['head_dim']=64
    if schema!=regenerated:raise ValueError('old source/schema differ')
    metadata=json.loads((folder/'metadata.json').read_text())
    abi=validate_abi((folder/'kernel.amdgcn').read_text(),metadata,schema)
    validate_dynamic_lds((folder/'kernel.llir').read_text(),abi['dynamic_shared_bytes'])
    hip,_=hip_runtime()
    if str(hip)!=source['host']['hip_path'] or sha256(hip)!=source['host']['hip_sha256']:raise ValueError('HIP changed')
    report['aot_source']=dict(run=str(old),export_sha256=sha256(old/'attention-export.json'),hsaco_sha256=sha256(folder/'kernel.hsaco'),abi=abi)
    spec=importlib.util.spec_from_file_location('bf16_diagnostic_attention',HERE/'candidate_attention.py')
    dsl=importlib.util.module_from_spec(spec);spec.loader.exec_module(dsl)
    arrangement,application,tensors=dsl.premake(with_kv_cache=False,emb_dim=64,is_causal=False,with_attn_mask=False,
        causal_variant=dsl.CausalVariant.LOWER_RIGHT,dtype=nt.bfloat16,block_size_m=32,block_size_n=32)
    jit=nt.make(arrangement,application,tensors,caller='torch',kernel_name='nine_bf16_tail_pair_jit',num_warps=4,num_stages=1,max_num_configs=1)
    report['jit_source']=save_generated(jit,'attention-pair-jit',args.run_dir)
    adapter=Adapter(root/'libnine_attention_host.so',hip)
    try:
        handle=adapter.open(folder,metadata,abi)
        case=next(c for c in case_specs() if c['name']=='bfloat16-tail')
        for repeat in range(2):
            inputs=[guarded_tensor(1,length,2,64,torch.bfloat16,'bsnd',-19) for length in (17,33,33)]
            ab,ao,ag=guarded_tensor(1,17,2,64,torch.bfloat16,'bsnd',-23)
            jb,jo,jg=guarded_tensor(1,17,2,64,torch.bfloat16,'bsnd',-23)
            kv=torch.tensor([33],dtype=torch.int32,device='cuda');mask=torch.zeros((1,1,1,1),dtype=torch.bfloat16,device='cuda')
            gen=torch.Generator().manual_seed(20261110+repeat*1009+sum(map(ord,case['name'])))
            for _,t,_ in inputs:t.copy_(torch.randn(tuple(t.shape),generator=gen).bfloat16())
            ao.fill_(float('nan'));jo.fill_(float('nan'));torch.cuda.synchronize()
            before=[b.cpu().clone() for b,_,_ in inputs];kv_before=kv.cpu().clone();mask_before=mask.cpu().clone()
            q,k,v=[i[1] for i in inputs];expected=cpu_reference(q,k,v,case)
            stream=torch.cuda.Stream();count=adapter.calls(handle)
            with torch.cuda.stream(stream):
                adapter.launch(handle,schema,(q,k,v,kv,mask,False,.125,ao,False,2))
                jit(q,k,v,kv,mask,False,.125,jo,False,2)
            stream.synchronize()
            aot,jit_metrics=metrics(ao,expected,.025,.025),metrics(jo,expected,.025,.025)
            unchanged=all(torch.equal(b.cpu(),s) for (b,_,_),s in zip(inputs,before)) and torch.equal(kv.cpu(),kv_before) and torch.equal(mask.cpu(),mask_before)
            guards=bool(torch.all(ab.cpu()[ag]==-23)) and bool(torch.all(jb.cpu()[jg]==-23)) and all(bool(torch.all(b.cpu()[g]==-19)) for b,_,g in inputs)
            observation=dict(round=repeat+1,aot=aot,jit=jit_metrics,aot_passed=aot['passed'],jit_passed=jit_metrics['passed'],
                inputs_unchanged=unchanged,guards_unchanged=guards,host_calls=adapter.calls(handle)-count,
                aot_rows_identical=bool(torch.equal(ao[:,:,0,:],ao[:,:,1,:])))
            np.savez(args.run_dir/f'bf16-pair-{repeat+1}.npz',q=q.cpu().double().numpy(),k=k.cpu().double().numpy(),v=v.cpu().double().numpy(),
                expected=expected.numpy(),aot=ao.cpu().double().numpy(),jit=jo.cpu().double().numpy())
            report['attention'].append(observation);write_json(args.run_dir/'bf16-diagnostic.json',report)
            print('attention_pair='+json.dumps(observation),flush=True)
            if not unchanged or not guards or observation['host_calls']!=1:raise ValueError('pair memory/launch checks failed')
    finally:adapter.close()


def main():
    p=argparse.ArgumentParser()
    for key in ('candidate','vendor','baseline','run-dir'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();report=dict(result='FAILED',correctness_passed=False,conversion=[],attention=[],performance_measurement_valid=False)
    try:
        write_json(args.run_dir/'bf16-diagnostic.json',report)
        report.update(check_environment(args));report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        conversions(report,args)
        report['last_stage']='attention-pair';write_json(args.run_dir/'bf16-diagnostic.json',report)
        attention_pair(report,args)
        report['compiler_assets']=collect_compiler_assets(args.run_dir)
        report['result']='DIAGNOSTIC_COMPLETE';validate_diagnostic(report)
        print('BF16 DIAGNOSTIC COLLECTED; no AOT correctness claim',flush=True);return 0
    except Exception as e:
        report.update(result='FAILED',error=f'{type(e).__name__}: {e}');traceback.print_exc();return 1
    finally:
        if 'compiler_assets' not in report:
            try:report['compiler_assets']=collect_compiler_assets(args.run_dir)
            except Exception as e:report['compiler_collection_error']=str(e)
        write_json(args.run_dir/'bf16-diagnostic.json',report)


if __name__=='__main__':raise SystemExit(main())
