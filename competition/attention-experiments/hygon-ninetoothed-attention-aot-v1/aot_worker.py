#!/usr/bin/env python3
"""Fresh-process HSACO loader; no nt.make, Triton compile or generated JIT execution."""
import argparse
import json
import os
from pathlib import Path
import traceback

from attention_contract import case_specs,sha256,verify_candidate_attention,write_json
from attention_worker import check_environment,verify_case
from aot_contract import SCOPE,launch_schema,validate_abi,validate_dynamic_lds,validate_aot_report,verify_export_files
from aot_native import Adapter,hip_runtime

HERE=Path(__file__).resolve().parent


class BoundKernel:
    def __init__(self,adapter,handle,schema):
        self.adapter,self.handle,self.schema=adapter,handle,schema
        self.streams=[]
    def __call__(self,*args):
        self.streams.append(self.adapter.launch(self.handle,self.schema,args))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','vendor','baseline','run-dir'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    report=dict(result='FAILED',validation_scope=SCOPE,last_stage='environment',cases=[],
                integration_status='NOT_TESTED',performance_measurement_valid=False,
                fresh_process_launch=False,host_launch_count=0,export_count=0,pid=os.getpid())
    adapter=None
    def save():write_json(args.run_dir/'attention-aot.json',report)
    try:
        save();report.update(check_environment(args))
        import torch
        # Forbid accidental JIT/compilation in this launch process, including future refactors.
        import triton
        import ninetoothed
        from triton.runtime.jit import JITFunction
        def forbidden(*a,**kw):raise RuntimeError('JIT/compilation forbidden in AOT launch process')
        triton.compile=forbidden;ninetoothed.make=forbidden;JITFunction.run=forbidden
        source=json.loads((args.run_dir/'attention-export.json').read_text())
        if (source.get('result')!='AOT_EXPORT_COMPLETE' or len(source.get('exports',[]))!=6
                or source.get('pid')==os.getpid() or source.get('gpu_attention_launches')!=0):
            raise ValueError('missing complete independent export evidence')
        root=args.run_dir/'exports'
        verify_export_files(root,source['files'])
        report['export_report_sha256']=sha256(args.run_dir/'attention-export.json')
        report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        hip,driver=hip_runtime()
        if str(hip)!=source['host']['hip_path'] or sha256(hip)!=source['host']['hip_sha256']:
            raise ValueError('HIP runtime differs between export and launch')
        adapter=Adapter(root/'libnine_attention_host.so',hip)
        kernels={}
        for info in source['exports']:
            folder=(root/info['folder']).resolve()
            if not folder.is_relative_to(root.resolve()):raise ValueError('unsafe export folder')
            verify_export_files(folder,info['files'])
            schema=json.loads((folder/'launch-schema.json').read_text())
            regenerated=launch_schema((folder/'ninetoothed-generated.py').read_text(),info['label'],info['dtype'],info['causal'])
            regenerated['head_dim']=info['head_dim']
            if schema!=regenerated:raise ValueError('stored launch bindings differ from generated source')
            metadata=json.loads((folder/'metadata.json').read_text())
            abi=validate_abi((folder/'kernel.amdgcn').read_text(),metadata,schema)
            validate_dynamic_lds((folder/'kernel.llir').read_text(),abi['dynamic_shared_bytes'])
            if abi!=info['abi']:raise ValueError('export ABI differs on reload')
            handle=adapter.open(folder,metadata,abi)
            key=info['dtype'],info['head_dim'],info['causal']
            if key in kernels:raise ValueError('duplicate specialization')
            kernels[key]=BoundKernel(adapter,handle,schema)
        report['export_count']=len(kernels)
        report['fresh_process_launch']=True
        for case in case_specs():
            report['last_stage']=case['name'];save()
            kernel=kernels[case['dtype'],case['dim'],case['causal']]
            before=adapter.calls(kernel.handle)
            result=verify_case(kernel,case,args.run_dir)
            calls=adapter.calls(kernel.handle)-before
            # Imported verifier is byte-identical to the validated JIT harness; its launch is injected above.
            for metric in result['metrics']:
                metric['elapsed_seconds_with_validation_not_benchmark']=metric.pop('elapsed_seconds_including_possible_jit')
            result.update(host_calls=calls,explicit_stream_verified=all(kernel.streams),launch_path='HIP Module API; exported HSACO')
            if calls!=2 or not kernel.streams:raise ValueError('missing actual host launches')
            report['host_launch_count']+=calls
            report['cases'].append(result);save()
        adapter.close();adapter=None
        report.update(result='AOT_PROBE_PASSED',last_stage='complete')
        validate_aot_report(report)
        print('NINE ATTENTION AOT HOST PROBE PASSED; framework/model/performance NOT_TESTED',flush=True)
        return 0
    except Exception as error:
        report.update(result='FAILED',error=f'{type(error).__name__}: {error}')
        traceback.print_exc();return 1
    finally:
        if adapter:
            try:adapter.close()
            except Exception as error:report['cleanup_error']=str(error)
        save()


if __name__=='__main__':raise SystemExit(main())
