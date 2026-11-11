#!/usr/bin/env python3
"""Offline isolated BF16 observations with original runtime postcheck."""
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

from attention_contract import (read_verified_state,verify_manifest,verify_upstream,
                                verify_candidate_attention,write_json)
from run_attention_probe import JobInterrupted,handle_signal,load_helper,run_child

HERE=Path(__file__).resolve().parent
SCOPE='bf16-diagnostic-only-not-formal-correctness-or-performance'


def collect_evidence(run_dir):
    output=run_dir/'attention-aot-evidence.zip'
    files=[p for p in sorted(run_dir.rglob('*')) if p.is_file()
           and p!=output and p.name not in ('run.log','bf16-diagnostic-controller.json')
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
    report=dict(version='1.1-bf16-cast-loaded-value-diagnostic',result='FAILED',last_stage='preflight',
                validation_scope=SCOPE,integration_status='NOT_TESTED',performance_measurement_valid=False)
    started=time.monotonic()
    def save():write_json(args.run_dir/'bf16-diagnostic-controller.json',report)
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
        env['H3_BF16_OLD_RUN']=str(args.work_home/'logs/infinifabric-nine-attention-aot-20261111T015938Z-364509')
        stage('bf16-diagnostic-worker')
        run_child([sys.executable,'-u',str(HERE/'bf16_diagnostic_worker.py'),'--candidate',str(candidate),
            '--vendor',str(vendor),'--baseline',str(args.run_dir/'base-runtime.json'),
            '--run-dir',str(args.run_dir)],env,1200,'bf16-diagnostic-worker')
        from bf16_diagnostic_contract import validate_diagnostic
        validate_diagnostic(json.loads((args.run_dir/'bf16-diagnostic.json').read_text()))
        stage('base-runtime-postcheck')
        after=json.loads(subprocess.check_output([sys.executable,'-c',helper.BASE_PROBE],env=clean,text=True,timeout=60))
        if after!=baseline:raise RuntimeError('base Torch/NumPy identity changed')
        report.update(result='DIAGNOSTIC_COLLECTED',last_stage='complete',base_runtime_unchanged=True)
        print('BF16 DIAGNOSTIC COLLECTED; numerical correctness NOT_CLAIMED',flush=True)
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
