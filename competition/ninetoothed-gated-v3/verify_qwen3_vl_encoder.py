#!/usr/bin/env python3
"""Exercise the real H3 Qwen3-VL facade/native encoder on tiny synthetic fixtures.

Default: text/image requests. --video is a separate frame-boundary diagnostic.
No project mutations, new packages, official weights or performance claims.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from qwen3_vl_reference import (LAYERS, DTYPES, TOLERANCES, make_config, make_weights,
                               make_case, encoder_reference)
from verify_h3_rotary_mha import (compare_arrays, file_hash, loaded_libraries,
                                 source_metadata, write_report)


SOURCE_FILES = (
    'python/infinilm/multimodal/minimax_h3.py', 'python/infinilm/infer_engine.py',
    'python/infinilm/modeling_utils.py',
    'csrc/models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp',
    'csrc/models/qwen3_vl/qwen3_vl_text_model.cpp',
    'csrc/models/qwen3_vl/qwen3_vl_attention.cpp',
    'csrc/models/qwen3_5/qwen3_5_vision.cpp',
    'csrc/layers/attention/backends/static_attn.cpp',
)


def run_device(report, path, video):
    import torch
    import infinicore as ic
    import transformers
    from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
    from infinilm.lib import _infinilm
    from safetensors.torch import save_file

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Use the normal Hygon competition Pod Python')
    device = ic.device('cuda',0)
    if device._underlying.type != type(device._underlying).Type.HYGON or ic.use_ntops:
        raise RuntimeError('Expected native InfiniCore HYGON, without ntops substitution')
    torch.cuda.set_device(0)
    ic.set_device(device)

    class TinyConditionEncoder(MiniMaxH3ConditionEncoder):
        # Test-only truncation; the repository's production 50-layer class is unchanged.
        num_hidden_layers = LAYERS

    def read(value):
        native = value if isinstance(value,ic.Tensor) else ic.Tensor(value)
        torch.cuda.synchronize()
        output = torch.empty(tuple(native.shape),device='cuda',dtype=ic.utils.to_torch_dtype(native.dtype))
        ic.from_torch(output).copy_(native)
        ic.sync_device(); torch.cuda.synchronize()
        output = output.cpu()
        return output.float().numpy().copy() if output.is_floating_point() else output.numpy().copy()

    work = Path(os.environ['WORK_HOME'])
    sources = source_metadata(work)
    for name in SOURCE_FILES:
        sources[name] = {'sha256':file_hash(work/'Projects/InfiniLM'/name)}
    report['environment'] = {'python':sys.executable, 'torch':torch.__version__,
                             'hip':torch.version.hip, 'transformers':transformers.__version__,
                             'device':torch.cuda.get_device_name(0), 'native_device':str(device._underlying.type),
                             'libraries':loaded_libraries(), 'sources':sources,
                             'infinilm_extension':{'path':_infinilm.__file__,'sha256':file_hash(Path(_infinilm.__file__))}}
    print(json.dumps(report['environment'],indent=2),flush=True)
    write_report(path,report)
    for dtype_name in DTYPES:
        dtype = getattr(torch,dtype_name)
        def q(array):
            return torch.from_numpy(np.asarray(array,np.float32).copy()).to(dtype).float().numpy().copy()
        c = make_config(dtype_name)
        w = make_weights(c,q)
        record = {'dtype':dtype_name,'status':'running','runs':[],'negative_checks':[]}
        report['models'].append(record)
        def stage(name):
            record['stage']=name
            write_report(path,report)
            print(f'{dtype_name}: {name}',flush=True)
        stage('fixtures')
        folder = path.parent/dtype_name
        folder.mkdir()
        (folder/'config.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
        tensors = {k:torch.from_numpy(v.copy()).to(dtype).contiguous() for k,v in w.items()}
        # Unused head/norm keys verify load_only_model_keys and selected intermediate output.
        tensors['model.language_model.norm.weight']=torch.full((64,),9.,dtype=dtype)
        tensors['lm_head.weight']=torch.zeros((64,64),dtype=dtype)
        save_file(tensors,str(folder/'model.safetensors'))
        record.update(config=c,fixture_dir=str(folder),checkpoint_sha256=file_hash(folder/'model.safetensors'))
        del tensors
        stage('create_facade_and_load_native_weights')
        encoder = TinyConditionEncoder(folder,device='hygon',device_id=0,tp_size=1,dtype=dtype,max_cache_len=128)
        encoder.eval()
        stage('read_back_native_parameters')
        states=encoder.engine.state_dict()
        if len(states)!=1 or set(states[0])!=set(w):
            got=set(states[0]) if states else set()
            raise AssertionError(f'Native keys differ: missing={sorted(set(w)-got)}, extra={sorted(got-set(w))}')
        for name,value in states[0].items():
            native=value if isinstance(value,ic.Tensor) else ic.Tensor(value)
            if native.dtype!=getattr(ic,dtype_name):
                raise AssertionError(f'Parameter dtype mismatch: {name}')
            if not np.array_equal(read(native),w[name]):
                raise AssertionError(f'Parameter shape/value mismatch: {name}')
        record['parameters_exact']=len(w)
        print(f'  parameters_exact={len(w)}',flush=True)

        def torch_inputs(case):
            values = {k:torch.from_numpy(case[k].copy()).to('cuda') for k in ('input_ids','mm_token_type_ids')}
            values['attention_mask']=torch.ones_like(values['input_ids'])
            for kind,pixel_name,grid_name in [(1,'pixel_values','image_grid_thw'),(2,'pixel_values_videos','video_grid_thw')]:
                items=[m for m in case['media'] if m['kind']==kind]
                if items:
                    values[pixel_name]=torch.from_numpy(np.concatenate([m['pixels'] for m in items])).to(device='cuda',dtype=dtype)
                    values[grid_name]=torch.tensor([m['grid'] for m in items],dtype=torch.int64)
            return values

        stage('rejected_inputs')
        # Only call preparation here; malformed fixtures never enter native forward.
        base=torch_inputs(make_case(c,1,q))
        messages={'batch_two':'batch size 1', 'padded':'padded conditioner batches',
                  'unknown_modality':'Unsupported Qwen3-VL modality type',
                  'missing_pixels':'image spans do not match pixel inputs',
                  'extra_pixel_row':'pixel rows do not match grid_thw',
                  'wrong_span':'visual token span does not match the merged grid'}
        for name in ('batch_two','padded','unknown_modality','missing_pixels','extra_pixel_row','wrong_span'):
            bad={k:v.clone() for k,v in base.items()}
            expected_exception=ValueError
            if name=='batch_two':
                bad['input_ids']=bad['input_ids'].repeat(2,1)
                expected_exception=NotImplementedError
            elif name=='padded':
                bad['attention_mask'][0,-1]=0
                expected_exception=NotImplementedError
            elif name=='unknown_modality':
                bad['mm_token_type_ids'][0,2]=9
            elif name=='missing_pixels':
                del bad['pixel_values']
            elif name=='extra_pixel_row':
                bad['pixel_values']=torch.cat((bad['pixel_values'],bad['pixel_values'][:1]))
            else:
                bad['mm_token_type_ids'][0,7]=0
            try:
                encoder._prepare_native_inputs(bad)
            except expected_exception as exc:
                if messages[name] not in str(exc):
                    raise AssertionError(f'Wrong rejection for {name}: {exc}') from exc
                record['negative_checks'].append({'name':name,'status':'passed','error':str(exc)})
            else:
                raise AssertionError(f'Malformed input was not rejected: {name}')
        print(f'  rejected_inputs={len(record["negative_checks"])}',flush=True)
        for index in range(2 if video else 4):
            case=make_case(c,index,q,video=video)
            run={'name':case['name'],'status':'running','sequence':int(case['input_ids'].size)}
            record['runs'].append(run)
            stage(case['name']+'/reference')
            expected=encoder_reference(c,w,case,q)
            values=torch_inputs(case)
            originals={k:v.clone() for k,v in values.items()}
            stage(case['name']+'/prepare_metadata')
            prepared=encoder._prepare_native_inputs(values)
            exact={'input_ids':case['input_ids'],'position_ids':case['positions'],
                   'input_offsets':np.array([0,case['input_ids'].size],np.int32),
                   'past_kv_lengths':np.array([0],np.int32),
                   'total_kv_lengths':np.array([case['input_ids'].size],np.int32)}
            for name,expected_value in exact.items():
                if not np.array_equal(read(prepared[name]),expected_value):
                    raise AssertionError(f'Prepared metadata mismatch: {name}')
            for key in ('pixel_values','image_grid_thw','image_bound'):
                if len(prepared[key])!=len(case['media']):
                    raise AssertionError(f'Prepared list length mismatch: {key}')
            if prepared['image_req_ids']!=[0]*len(case['media']):
                raise AssertionError('Prepared request IDs mismatch')
            for i,media in enumerate(case['media']):
                for key,ref in [('pixel_values',media['pixels']),('image_grid_thw',np.array(media['grid'])),('image_bound',np.array(media['bounds']))]:
                    if not np.array_equal(read(prepared[key][i]),ref):
                        raise AssertionError(f'Prepared media mismatch: {key}/{i}')
            run['metadata_exact']=True
            del prepared
            stage(case['name']+'/native_facade_forward')
            torch.cuda.synchronize()
            with torch.no_grad():
                output=encoder(**values)
            ic.sync_device(); torch.cuda.synchronize()
            if len(output.hidden_states)!=LAYERS+1 or any(v is not None for v in output.hidden_states[:LAYERS]):
                raise AssertionError('Facade intermediate hidden-state structure mismatch')
            if output.hidden_states[LAYERS] is not output.last_hidden_state:
                raise AssertionError('Facade must expose selected hidden state')
            actual=output.last_hidden_state.detach().cpu().float().numpy().copy()
            np.savez(folder/f'outputs_{index}.npz',actual=actual,reference=expected,positions=case['positions'])
            run['error']=compare_arrays(actual,expected,*TOLERANCES[dtype_name])
            for name,value in values.items():
                if not torch.equal(value,originals[name]):
                    raise AssertionError(f'Input modified: {name}')
            run.update(status='passed',inputs_unchanged=True)
            write_report(path,report)
            print(f'  PASS {case["name"]} S={run["sequence"]} max_abs={run["error"]["max_abs"]:.6g} metadata_exact=True inputs_unchanged=True',flush=True)
        record.update(status='passed',stage='complete')
        write_report(path,report)
        del states,encoder
        gc.collect()
    report['status']='passed'
    write_report(path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--video',action='store_true',help='separate two-frame vision boundary diagnostic')
    args=parser.parse_args()
    mode='video' if args.video else 'image_text'
    count=6 if args.video else 12
    if args.describe:
        print(json.dumps({'mode':mode,'config':make_config('float32'),'selected_layers':LAYERS,
                          'native_forwards':count,'official_model':False,
                          'video_frame_boundary_risk':True},indent=2))
        return 0
    folder=Path(os.environ['LOG_DIR'])/f'qwen3_vl_encoder_{mode}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    folder.mkdir(parents=True)
    path=folder/'report.json'
    report={'status':'initializing','mode':mode,'models':[],'synthetic_model':True,
            'selected_layers':LAYERS,'production_selected_layers':50,'tp_size':1,
            'performance_tested':False,'official_model_tested':False,'tolerances':TOLERANCES,
            'video_frame_attention_reference':'isolated_per_frame',
            'script_hashes':{name:file_hash(Path(__file__).with_name(name)) for name in
                             ('verify_qwen3_vl_encoder.py','qwen3_vl_reference.py','verify_h3_rotary_mha.py')}}
    write_report(path,report)
    print(f'report: {path}',flush=True)
    try:
        run_device(report,path,args.video)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if report['models']:
            record=report['models'][-1]
            record['status']='failed'
            if record['runs'] and record['runs'][-1]['status']=='running':
                record['runs'][-1]['status']='failed'
        write_report(path,report)
        traceback.print_exc()
        print(f'FAILED: {mode}; report={path}',flush=True)
        return 1
    print(f'PASSED: {mode}, 3 models, {count} forwards, 18 rejection checks; report={path}',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
