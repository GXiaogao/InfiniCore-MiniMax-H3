#!/usr/bin/env python3
"""NineToothed -> explicit generic HIP kernels. This process never launches attention."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

from attention_contract import case_specs,sha256,verify_candidate_attention,verify_upstream,write_json
from attention_worker import check_environment
from aot_contract import SCOPE,launch_schema,validate_abi,validate_dynamic_lds
from aot_native import build_host

HERE=Path(__file__).resolve().parent


def export_one(nt,upstream,case,root):
    import triton
    from triton.compiler import ASTSource
    label=f'nine_attention_{case["dtype"]}_d{case["dim"]}_c{int(case["causal"])}'
    folder=root/label;folder.mkdir()
    arrangement,application,tensors=upstream.premake(with_kv_cache=False,emb_dim=case['dim'],
        is_causal=case['causal'],with_attn_mask=False,causal_variant=upstream.CausalVariant.LOWER_RIGHT,
        dtype=getattr(nt,case['dtype']),block_size_m=32,block_size_n=32)
    handle=nt.make(arrangement,application,tensors,caller='torch',kernel_name=label,
                   num_warps=4,num_stages=1,max_num_configs=1)
    generated=folder/'ninetoothed-generated.py'
    shutil.copyfile(handle._source,generated)
    schema=launch_schema(generated.read_text(),label,case['dtype'],case['causal'])
    schema['head_dim']=case['dim']
    if list(handle._kernel.arg_names)!=schema['arg_names']:
        raise RuntimeError('generated JITFunction parameters differ from parsed AST')
    target=triton.runtime.driver.active.get_current_target()
    options=dict(num_warps=4,num_stages=1,num_ctas=1,enable_fp_fusion=True)
    write_json(folder/'launch-schema.json',schema)
    write_json(folder/'compile-request.json',dict(signature=schema['signature'],constexprs=schema['constexprs'],
        attrs={},options=options,target=str(target),generated_sha256=sha256(generated),
        size_stride_policy='dynamic-i64-no-value-or-alignment-specialization'))
    print(f'export={label} status=COMPILE',flush=True)
    compiled=triton.compile(ASTSource(fn=handle._kernel,signature=schema['signature'],
        constexprs=schema['constexprs'],attrs={}),target=target,options=options)
    metadata=compiled.metadata._asdict()
    metadata['target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
    write_json(folder/'metadata.json',metadata)
    for extension in ('hsaco','ttir','ttgir','llir','amdgcn'):
        value=compiled.asm[extension]
        (folder/('kernel.'+extension)).write_bytes(value if isinstance(value,bytes) else value.encode())
    abi=validate_abi(compiled.asm['amdgcn'],metadata,schema)
    validate_dynamic_lds(compiled.asm['llir'],abi['dynamic_shared_bytes'])
    write_json(folder/'abi.json',abi)
    files={p.name:sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}
    info=dict(label=label,dtype=case['dtype'],head_dim=case['dim'],causal=case['causal'],
              folder=label,files=files,abi=abi,generated_sha256=sha256(generated))
    write_json(folder/'export.json',info)
    print(f'export={label} status=EXPORTED args={abi["argument_count"]} shared={abi["shared_bytes"]}',flush=True)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','vendor','baseline','run-dir'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    report=dict(result='FAILED',validation_scope=SCOPE,last_stage='environment',exports=[],pid=__import__('os').getpid())
    def save():write_json(args.run_dir/'attention-export.json',report)
    try:
        save();report.update(check_environment(args))
        report['upstream']=verify_upstream(HERE/'upstream_attention.py')
        report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        import ninetoothed as nt
        spec=importlib.util.spec_from_file_location('h3_nine_aot_dsl',HERE/'candidate_attention.py')
        upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        root=args.run_dir/'exports';root.mkdir()
        report['last_stage']='host-build';save()
        report['host']=build_host(root)
        seen=set()
        for case in case_specs():
            key=case['dtype'],case['dim'],case['causal']
            if key in seen:continue
            report['last_stage']='export-'+case['name'];save()
            info=export_one(nt,upstream,case,root)
            report['exports'].append(info);seen.add(key);save()
        manifest={p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
        report.update(result='AOT_EXPORT_COMPLETE',last_stage='complete',files=manifest,
                      host_binary_sha256=sha256(root/'libnine_attention_host.so'),gpu_attention_launches=0)
        print('AOT EXPORT COMPLETE; device launches are tested in a separate process',flush=True)
        return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:save()


if __name__=='__main__':raise SystemExit(main())
