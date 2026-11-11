#!/usr/bin/env python3
"""Offline isolated AOT export, fresh process launch, original runtime postcheck."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

from attention_contract import (case_specs,read_verified_state,verify_manifest,verify_upstream,
                                verify_candidate_attention,write_json)
from aot_contract import SCOPE,validate_aot_report
from run_attention_probe import JobInterrupted,handle_signal,load_helper,run_child

HERE=Path(__file__).resolve().parent


def validate_exports(report):
    want={(c['dtype'],c['dim'],c['causal']) for c in case_specs()}
    exports=report.get('exports',[])
    got=[(e.get('dtype'),e.get('head_dim'),e.get('causal')) for e in exports]
    if (report.get('result')!='AOT_EXPORT_COMPLETE' or report.get('validation_scope')!=SCOPE
            or len(exports)!=6 or set(got)!=want or report.get('gpu_attention_launches')!=0
            or report.get('target',{}).get('arch')!='gfx936'
            or not report.get('files') or not report.get('host_binary_sha256')
            or type(report.get('pid')) is not int or report['pid']<=0):
        raise ValueError('incomplete independent export evidence')


def collect_evidence(run_dir):
    output=run_dir/'attention-aot-evidence.zip'
    files=[p for p in sorted(run_dir.rglob('*')) if p.is_file()
           and p!=output and p.name not in ('run.log','attention-aot-controller.json')
           and '.tmp.' not in p.name]
    total=sum(p.stat().st_size for p in files)
    if total>120*1024*1024:return dict(created=False,reason='uncompressed evidence exceeds120MiB',bytes=total)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:bundle.write(path,path.relative_to(run_dir).as_posix())
    return dict(created=True,file=output.name,uncompressed_bytes=total,files=len(files))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('work-home','run-dir','local-root'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    previous=signal.signal(signal.SIGTERM,handle_signal)
    report=dict(version='1.1-nine-attention-aot-dynamic-lds',result='FAILED',last_stage='preflight',
                validation_scope=SCOPE,integration_status='NOT_TESTED',performance_measurement_valid=False)
    started=time.monotonic()
    def save():write_json(args.run_dir/'attention-aot-controller.json',report)
    def stage(name):report['last_stage']=name;save();print('stage='+name,flush=True)
    try:
        stage('package-validation')
        verify_manifest(HERE,json.loads((HERE/'package-manifest.json').read_text())['files'])
        report['upstream']=verify_upstream(HERE/'upstream_attention.py')
        report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        if sys.platform!='linux' or sys.version_info[:2]!=(3,11):raise ValueError('Linux Python3.11 required')
        if any(os.environ.get(k) for k in ('HSA_OVERRIDE_GFX_VERSION','TRITON_OVERRIDE_ARCH')):
            raise ValueError('architecture overrides forbidden')
        stage('existing-candidate-state')
        state=read_verified_state(args.work_home);candidate=Path(state['candidate'])
        report['candidate']=str(candidate);report['candidate_integrity']=state['candidate_integrity']
        helper=load_helper();clean=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):clean.pop(key,None)
        clean['PYTHONNOUSERSITE']='1'
        stage('base-runtime-identity')
        baseline=json.loads(subprocess.check_output([sys.executable,'-c',helper.BASE_PROBE],env=clean,text=True,timeout=60))
        write_json(args.run_dir/'base-runtime.json',baseline)
        stage('job-local-vendor')
        wheel=args.work_home/'packages/minimax-h3-hygon-runtime-inputs'/helper.TRITON_FILE
        if helper.sha(wheel)!=helper.TRITON_SHA:raise ValueError('fixed vendor wheel hash differs')
        vendor=args.local_root/'vendor-site-packages'
        helper.extractor().install_wheels_direct([wheel],vendor)
        report['vendor_cache_patch']=helper.patch_triton_cache(vendor/'triton/runtime/cache.py')
        from llvm_compat import patch_vendor_backend
        report['vendor_llvm_compat']=patch_vendor_backend(vendor,HERE/'llvm_compat.py',args.run_dir/'llvm-compat')
        env=helper.worker_environment(candidate,vendor,args.local_root/'cache',clean)
        env['H3_LLVM_COMPAT_LOG_DIR']=str((args.run_dir/'llvm-compat').resolve())
        deadline=time.monotonic()+1200
        for name in ('aot_export','aot_worker'):
            stage(name)
            remaining=deadline-time.monotonic()
            if remaining<=0:raise subprocess.TimeoutExpired(name,1200)
            run_child([sys.executable,'-u',str(HERE/(name+'.py')),'--candidate',str(candidate),
                '--vendor',str(vendor),'--baseline',str(args.run_dir/'base-runtime.json'),
                '--run-dir',str(args.run_dir)],env,remaining,name)
            if name=='aot_export':
                validate_exports(json.loads((args.run_dir/'attention-export.json').read_text()))
        validate_aot_report(json.loads((args.run_dir/'attention-aot.json').read_text()))
        stage('base-runtime-postcheck')
        after=json.loads(subprocess.check_output([sys.executable,'-c',helper.BASE_PROBE],env=clean,text=True,timeout=60))
        if after!=baseline:raise RuntimeError('base Torch/NumPy identity changed')
        report.update(result='AOT_PROBE_PASSED',last_stage='complete',base_runtime_unchanged=True)
        print('NINE ATTENTION AOT/HOST VALIDATION PASSED; framework/model/performance NOT_TESTED',flush=True)
        return 0
    except subprocess.TimeoutExpired as error:
        report.update(result='FAILED_TIMEOUT',error=str(error));traceback.print_exc();return 124
    except JobInterrupted as error:
        report.update(result='INTERRUPTED',error=str(error));traceback.print_exc();return 143
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:
        report['elapsed_seconds_including_setup']=time.monotonic()-started
        try:report['evidence']=collect_evidence(args.run_dir)
        except Exception as error:report['evidence_error']=str(error)
        save();signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':raise SystemExit(main())
