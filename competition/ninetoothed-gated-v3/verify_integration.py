#!/usr/bin/env python3
"""Isolated stage verification of public op, H3 routing and unchanged model regressions."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid
import zipfile
from common import sha256,write_json,run_child,stage_passed

STAGES=(('ops',44),('h3_off',18),('h3_on',18),('bridge_off',12),('bridge_on',12),('mha',85))


def check_bundle():
    root=Path(os.environ['INFINICORE_H3_GATED_BUNDLE']).resolve()
    value=json.loads((root/'bundle.json').read_text())
    for name,expected in value['files'].items():
        path=root/name
        if not path.resolve().is_relative_to(root) or sha256(path)!=expected:
            raise RuntimeError('Bundle checksum mismatch: '+name)
    return {'path':str(root),'manifest_sha256':sha256(root/'bundle.json'),'files':value['files']}


def model_stage(stage,report,folder,path):
    import torch
    from infinicore.lib import _infinicore as native
    from infinilm.infer_engine import InferEngine
    enabled=stage.endswith('_on')
    os.environ['INFINILM_H3_NINETOOTHED_GATED']='1' if enabled else '0'
    original=InferEngine.forward_raw
    start=native.gated_residual_hygon_launch_count()
    def counted(self,*args,**kwargs):
        before=native.gated_residual_hygon_launch_count()
        value=original(self,*args,**kwargs)
        torch.cuda.synchronize()
        after=native.gated_residual_hygon_launch_count()
        h3=self.model_type=='minimax_h3'
        expected=2*int(self.hf_config['num_layers']) if h3 and enabled else 0
        if after-before!=expected:raise AssertionError(f'{self.model_type}: expected {expected} NineToothed launches, got {after-before}')
        if h3:
            report['cases'].append({'name':'h3_forward_'+str(len(report['cases'])),'passed':True,
                                     'native_launches':after-before,'expected':expected})
            write_json(path,report)
        return value
    InferEngine.forward_raw=counted
    try:
        if stage.startswith('h3'):
            import verify_h3_native_integration as check
            model_report={'status':'initializing','cases':[]}
            check.run_device(model_report,folder/'model_report.json',folder)
            if model_report['status']!='passed' or len(model_report['cases'])!=6:
                raise AssertionError('Incomplete H3 model report')
        else:
            import verify_h3_encoder_bridge as check
            model_report={'status':'initializing','models':[]}
            check.run_device(model_report,folder/'model_report.json')
            if model_report['status']!='passed' or len(model_report['models'])!=3:
                raise AssertionError('Incomplete encoder bridge report')
    finally:InferEngine.forward_raw=original
    report['native_launches']=native.gated_residual_hygon_launch_count()-start
    report['h3_enabled']=enabled
    report['backend']=native.gated_residual_hygon_backend_info()


def worker(args):
    folder=args.folder/args.stage;folder.mkdir()
    os.environ['LOG_DIR']=str(folder)
    path=folder/'result.json'
    report={'run_id':args.run_id,'stage':args.stage,'passed':False,'cases':[],'kernels':[]}
    try:
        report['bundle']=check_bundle()
        import infinicore as ic
        import torch
        from infinicore.lib import _infinicore
        if sys.prefix!=sys.base_prefix:raise RuntimeError('Run integration tests with the normal initialized Python, not candidate venv')
        if os.environ.get('HSA_OVERRIDE_GFX_VERSION') or os.environ.get('TRITON_OVERRIDE_ARCH'):
            raise RuntimeError('Architecture overrides are not allowed')
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('Expected usable vendor HIP runtime')
        arch=getattr(torch.cuda.get_device_properties(0),'gcnArchName','')
        if arch.split(':')[0]!='gfx906':raise RuntimeError('Expected actual gfx906, got '+arch)
        from verify_h3_rotary_mha import loaded_libraries
        report['environment']={'python':sys.executable,'torch':torch.__version__,'hip':torch.version.hip,
            'device':torch.cuda.get_device_name(0),'arch':arch,'infinicore':ic.__file__,
            'libraries':loaded_libraries(),
            'extension':{'path':_infinicore.__file__,'sha256':sha256(_infinicore.__file__)}}
        write_json(path,report)
        if args.stage=='ops':
            from verify_ops import run
            run(report,folder,path)
        elif args.stage=='mha':
            import verify_h3_mha_infiniop as check
            sys.argv=[check.__file__,'--graph']
            code=check.main()
            if code not in (None,0):raise RuntimeError('MHA regression failed')
            reports=list(folder.glob('h3_mha_extended_*.json'))
            if len(reports)!=1:raise RuntimeError('Missing MHA report')
            value=json.loads(reports[0].read_text())
            report['cases']=[{'name':c['name'],'passed':c['status']=='passed'} for c in value['cases']]
            if len(report['cases'])!=85:raise RuntimeError('Incomplete MHA cases')
        else:model_stage(args.stage,report,folder,path)
        report['passed']=True
        return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:
        write_json(path,report);print('RESULT:',path,'passed=',report['passed'],flush=True)


def compare_runs(folder):
    import numpy as np
    checked=[]
    for prefix in ('h3','bridge'):
        off=folder/(prefix+'_off');on=folder/(prefix+'_on')
        files=list(off.rglob('*.npz'))
        want=18 if prefix=='h3' else 12
        if len(files)!=want:raise AssertionError(f'{prefix}: expected {want} output files, got {len(files)}')
        if {p.relative_to(off) for p in files}!={p.relative_to(on) for p in on.rglob('*.npz')}:
            raise AssertionError('Off/on output files differ')
        for path in files:
            relative=path.relative_to(off)
            with np.load(path,allow_pickle=False) as a,np.load(on/relative,allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise AssertionError('Output keys differ')
                for key in a.files:
                    left,right=a[key],b[key]
                    finite=all(np.isfinite(x).all() for x in (left,right) if x.dtype.kind in 'buifc')
                    if left.dtype!=right.dtype or not finite or not np.array_equal(left,right):
                        raise AssertionError(f'Off/on mismatch: {prefix}/{relative}:{key}')
                checked.append({'path':prefix+'/'+relative.as_posix(),'keys':a.files,'exact':True})
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=[x[0] for x in STAGES]);parser.add_argument('--folder',type=Path)
    parser.add_argument('--run-id');parser.add_argument('--timeout',type=int,default=1200)
    args=parser.parse_args()
    if args.stage:
        if not args.folder or not args.run_id:parser.error('stage requires folder and run-id')
        return worker(args)
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=Path(os.environ['WORK_HOME'])/'logs'/('h3_gated_rounding_integration_'+run_id);folder.mkdir(parents=True)
    report={'run_id':run_id,'passed':False,'stages':[],'scope':'public InfiniCore op and tiny native model regressions',
            'official_model_tested':False,'full_video_tested':False,'performance_tested':False,
            'bundle':check_bundle(),'script_hashes':{p.name:sha256(p) for p in Path(__file__).parent.iterdir() if p.is_file()}}
    setup=Path(os.environ['LOG_DIR'])
    report['rounding_setup']=str(setup)
    # Carry both same-input probes into the final evidence ZIP, without copying
    # old native libraries or unrestricted directory trees.
    for name in ('rounding_probe_before','rounding_probe_after'):
        probe=setup/name
        proof=json.loads((probe/'report.json').read_text())
        if not proof.get('passed'):raise RuntimeError('Missing successful rounding probe: '+name)
        destination=folder/'rounding_probes'/name;destination.mkdir(parents=True)
        for source in probe.iterdir():
            if source.is_file() and source.suffix in ('.json','.npz'):
                shutil.copyfile(source,destination/source.name)
    bundle=Path(report['bundle']['path'])
    evidence=[bundle/'bundle.json',bundle/'host_build.json']
    for dtype in ('float32','float16','bfloat16'):
        evidence.extend((bundle/('export_'+dtype)).iterdir())
    for source in evidence:
        if source.is_file():
            destination=folder/'bundle_evidence'/source.relative_to(bundle)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
    path=folder/'summary.json';write_json(path,report);print('REPORT:',path,flush=True)
    ready=True
    for stage,count in STAGES:
        if not ready:
            report['stages'].append({'stage':stage,'status':'NOT_RUN'});continue
        command=[sys.executable,'-X','faulthandler','-u',str(Path(__file__).resolve()),'--stage',stage,'--folder',str(folder),'--run-id',run_id]
        print('RUN:',stage,flush=True)
        with (folder/(stage+'.log')).open('w',encoding='utf-8') as log:rc=run_child(command,log,args.timeout)
        print('\n'.join((folder/(stage+'.log')).read_text(errors='replace').splitlines()[-50:]),flush=True)
        value=None
        try:value=json.loads((folder/stage/'result.json').read_text())
        except (OSError,ValueError):pass
        ready=stage_passed(rc,value,stage,run_id,count)
        report['stages'].append({'stage':stage,'exit_code':rc,'expected_cases':count,'status':'PASS' if ready else 'FAIL'})
        write_json(path,report)
    try:
        if ready:report['off_on_comparisons']=compare_runs(folder)
        report['passed']=ready
    except Exception as error:
        report['comparison_error']=str(error);traceback.print_exc()
    write_json(path,report)
    files=[p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.so','.pyc')]
    write_json(folder/'files.json',[{'path':p.relative_to(folder).as_posix(),'sha256':sha256(p),'size':p.stat().st_size} for p in files])
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[folder/'files.json']:z.write(p,folder.name+'/'+p.relative_to(folder).as_posix())
    print('EVIDENCE ZIP:',archive,flush=True)
    print('INTEGRATION CHECKS PASSED' if report['passed'] else 'INTEGRATION CHECKS FAILED',flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
