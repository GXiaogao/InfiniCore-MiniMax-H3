#!/usr/bin/env python3
"""Offline controller: private candidate, fresh local vendor, no formal install."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from attention_contract import (SCOPE,read_verified_state,verify_manifest,verify_upstream,
                                verify_candidate_attention,write_json)

HERE=Path(__file__).resolve().parent


class JobInterrupted(Exception):
    pass


def handle_signal(number,frame):
    raise JobInterrupted(f'outer deadline/interruption signal={number}')


def load_helper():
    spec=importlib.util.spec_from_file_location('nt_attention_installer',HERE/'install_ninetoothed_gfx936.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_child(command,env,timeout_seconds,stage):
    print(f'stage={stage} status=START',flush=True)
    start=time.monotonic()
    process=subprocess.Popen(command,env=env,start_new_session=(os.name=='posix'))
    try:
        while True:
            remaining=timeout_seconds-(time.monotonic()-start)
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,timeout_seconds)
            try:
                rc=process.wait(timeout=min(20,remaining))
                if rc:raise subprocess.CalledProcessError(rc,command)
                return
            except subprocess.TimeoutExpired:
                print(f'stage={stage} status=RUNNING elapsed_seconds={time.monotonic()-start:.0f}',flush=True)
    finally:
        if process.poll() is None:
            try:
                if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                except ProcessLookupError:pass
                process.wait()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('work-home','run-dir','local-root'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv)
    previous_handler=signal.signal(signal.SIGTERM,handle_signal)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.2-hygon-nine-attention-ceil-div',result='FAILED',last_stage='preflight',
                validation_scope=SCOPE,integration_status='NOT_TESTED',performance_measurement_valid=False)
    start=time.monotonic()
    def stage(name):
        report['last_stage']=name
        write_json(args.run_dir/'attention-controller.json',report)
        print('stage='+name,flush=True)
    try:
        stage('package-validation')
        verify_manifest(HERE,json.loads((HERE/'package-manifest.json').read_text(encoding='utf-8'))['files'])
        report['upstream']=verify_upstream(HERE/'upstream_attention.py')
        report['dsl_patch']=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py')
        if sys.platform!='linux' or sys.version_info[:2]!=(3,11):
            raise RuntimeError('expected Linux Python3.11 batch environment')
        if any(os.environ.get(k) for k in ('HSA_OVERRIDE_GFX_VERSION','TRITON_OVERRIDE_ARCH')):
            raise RuntimeError('architecture overrides are forbidden in this probe')
        stage('existing-candidate-state')
        state=read_verified_state(args.work_home)
        candidate=Path(state['candidate'])
        report['candidate']=str(candidate)
        report['candidate_integrity']=state['candidate_integrity']
        helper=load_helper()
        clean=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):clean.pop(key,None)
        clean['PYTHONNOUSERSITE']='1'
        stage('base-runtime-identity')
        baseline=json.loads(subprocess.check_output([sys.executable,'-c',helper.BASE_PROBE],env=clean,text=True,timeout=60))
        write_json(args.run_dir/'base-runtime.json',baseline)
        stage('job-local-vendor')
        vendor_wheel=args.work_home/'packages/minimax-h3-hygon-runtime-inputs'/helper.TRITON_FILE
        if helper.sha(vendor_wheel)!=helper.TRITON_SHA:
            raise ValueError('fixed vendor Triton wheel checksum mismatch')
        vendor=args.local_root/'vendor-site-packages'
        helper.extractor().install_wheels_direct([vendor_wheel],vendor)
        report['vendor_cache_patch']=helper.patch_triton_cache(vendor/'triton/runtime/cache.py')
        from llvm_compat import patch_vendor_backend
        report['vendor_llvm_compat']=patch_vendor_backend(vendor,HERE/'llvm_compat.py',args.run_dir/'llvm-compat')
        report['vendor_wheel_sha256']=helper.TRITON_SHA
        env=helper.worker_environment(candidate,vendor,args.local_root/'cache',clean)
        env['H3_LLVM_COMPAT_LOG_DIR']=str((args.run_dir/'llvm-compat').resolve())
        # Do not trust a former job's vendor path saved in the installation receipt.
        stage('attention-worker')
        run_child([sys.executable,'-u',str(HERE/'attention_worker.py'),
                    '--candidate',str(candidate),'--vendor',str(vendor),
                    '--baseline',str(args.run_dir/'base-runtime.json'),'--run-dir',str(args.run_dir)],
                    env,1200,'attention-worker')
        from attention_contract import validate_report
        validate_report(json.loads((args.run_dir/'attention-probe.json').read_text(encoding='utf-8')))
        stage('base-runtime-postcheck')
        after=json.loads(subprocess.check_output([sys.executable,'-c',helper.BASE_PROBE],env=clean,text=True,timeout=60))
        if after!=baseline:raise RuntimeError('base Torch/NumPy identity changed')
        report.update(result='JIT_PROBE_PASSED',last_stage='complete',base_runtime_unchanged=True)
        print('NINE ATTENTION JIT PROBE PASSED; AOT/framework/model integration NOT_TESTED',flush=True)
        return 0
    except subprocess.TimeoutExpired as error:
        report.update(result='FAILED_TIMEOUT',error=str(error))
        traceback.print_exc()
        return 124
    except JobInterrupted as error:
        report.update(result='INTERRUPTED',error=str(error))
        traceback.print_exc()
        return 143
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        traceback.print_exc()
        return 1
    finally:
        report['elapsed_seconds_including_setup']=time.monotonic()-start
        write_json(args.run_dir/'attention-controller.json',report)
        signal.signal(signal.SIGTERM,previous_handler)


if __name__=='__main__':raise SystemExit(main())
