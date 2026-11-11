#!/usr/bin/env python3
"""Validate tiny native Qwen3-VL Encoder -> H3 forwards on a Hygon Pod.

This is a synthetic component bridge, not the complete generation pipeline.
CPU references and fixture caches never replace either native model forward.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import numpy as np

import qwen3_vl_reference as qwen
import verify_h3_native_integration as h3
from verify_qwen3_vl_encoder import SOURCE_FILES
from verify_h3_rotary_mha import (continuous_cache, compare_arrays, file_hash,
                                 loaded_libraries, source_metadata, write_report)

SCRIPT_FILES = ('verify_h3_encoder_bridge.py', 'qwen3_vl_reference.py',
                'verify_qwen3_vl_encoder.py', 'verify_h3_native_integration.py',
                'verify_h3_block_components.py', 'verify_h3_rotary_mha.py')


def make_configs(dtype):
    encoder = qwen.make_config(dtype)
    backbone = h3.make_config(dtype)
    backbone['text_dim'] = encoder['text_config']['hidden_size']
    return encoder, backbone


def make_cases(config, quantize):
    return [qwen.make_case(config, 0, quantize),
            qwen.make_case(config, 1, quantize),
            qwen.make_case(config, 0, quantize, video=True),
            qwen.make_case(config, 3, quantize)]


def make_backbone_weights(config, quantize):
    weights = h3.make_weights(config, quantize)
    # Synthetic fixture only: make conditioning failures visible even at BF16
    # tolerances. This does not alter official checkpoints or model equations.
    name = 'transformer_blocks.0.adaln_proj.linear.bias'
    bias = weights[name].copy().reshape(3, 6, config['hidden_size'])
    bias[:, 2, :] += np.float32(1.)  # attention residual gate for every modality
    weights[name] = quantize(bias.reshape(-1))
    return weights


def check_condition_sensitivity(config, weights, inputs, quantize, expected, tolerance):
    """Require deliberately wrong conditions to fail the same output tolerance."""
    features = inputs['encoder_hidden_states']
    result = {}
    for label, wrong in (('zero', np.zeros_like(features)), ('inverted', -features)):
        outputs = h3.model_reference(config, weights, dict(inputs, encoder_hidden_states=wrong), quantize)
        result[label] = {}
        for kind in ('video', 'audio'):
            diff = np.abs(outputs[kind] - expected[kind])
            outside = int(np.count_nonzero(diff > tolerance[0] + tolerance[1]*np.abs(expected[kind])))
            if not np.isfinite(diff).all() or outside == 0:
                raise AssertionError(f'Fixture cannot reject {label} condition for {kind} at {tolerance}')
            result[label][kind] = {'max_abs': float(diff.max()), 'outside_tolerance': outside,
                                   'elements': int(diff.size)}
    return result


def make_bridge_inputs(config, features, repeat, quantize):
    """CPU reference inputs only; the device runner binds its real GPU features."""
    a = np.asarray(features)
    if (a.ndim != 3 or a.shape[0] != 1 or a.shape[1] < 1
            or a.shape[2] != config['text_dim']
            or not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all()):
        raise ValueError('Expected finite floating encoder features [1, N>0, text_dim]')
    rng = np.random.default_rng(2026102501 + repeat)
    nv, na = (3, 2) if repeat % 2 == 0 else (5, 3)
    n = a.shape[1]
    s = nv + na + n
    p = rng.permutation(s).astype(np.int64)
    vi, ai, ti = p[:nv], p[nv:nv+na], p[nv+na:]
    tags = np.empty(s, np.int64)
    tags[vi], tags[ai], tags[ti] = 0, 1, 2
    positions = rng.normal(0, 2, (s, 3)).astype(np.float32)
    def values(shape):
        return rng.normal(0, .5, shape).astype(np.float32)
    return {'video_hidden_states': values((1, nv, config['latents_dim']*int(np.prod(config['patch_size'])))),
            'audio_hidden_states': values((1, na, config['audio_latents_dim'])),
            'encoder_hidden_states': a.astype(np.float32, copy=True),
            'timestep': np.array([.125+.1*repeat, .75+.05*repeat], np.float32),
            'timestep_indices': ((np.arange(s)+repeat) % 2).astype(np.int64),
            'token_tags': tags, 'position_ids': np.arange(s, dtype=np.int64),
            'rotary_cos_sin_cache': quantize(continuous_cache(positions, config['rope_freq_dim'], config['rope_theta'])),
            'video_indices': vi, 'audio_indices': ai, 'text_indices': ti}


def save_repository_evidence(work, directory):
    result = source_metadata(work)
    for repo_name in ('InfiniCore', 'InfiniLM'):
        repo = work/'Projects'/repo_name
        entries = {}
        for label, arguments in (('status', ['status', '--short', '--branch']),
                                 ('tracked_diff', ['diff', '--binary', 'HEAD']),
                                 ('submodules', ['submodule', 'status', '--recursive'])):
            command = subprocess.run(['git', '-C', str(repo), *arguments],
                                     capture_output=True, timeout=60)
            target = directory/f'{repo_name}_{label}.txt'
            target.write_bytes(command.stdout + command.stderr)
            entries[label] = {'path': str(target), 'sha256': file_hash(target),
                              'exit_code': command.returncode}
        result[repo_name]['worktree_evidence'] = entries
    for name in (*SOURCE_FILES, 'csrc/models/minimax_h3/minimax_h3_transformer.cpp',
                 'csrc/models/minimax_h3/minimax_h3_transformer.hpp'):
        result[name] = {'sha256': file_hash(work/'Projects/InfiniLM'/name)}
    return result


def run_device(report, path):
    import torch
    import infinicore as ic
    import transformers
    from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from infinilm.lib import _infinilm
    from safetensors.torch import save_file

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Requires normal runtime Python on the Hygon competition Pod')
    device = ic.device('cuda', 0)
    if device._underlying.type != type(device._underlying).Type.HYGON or ic.use_ntops:
        raise RuntimeError('Expected native InfiniCore HYGON, without ntops substitution')
    torch.cuda.set_device(0)
    ic.set_device(device)

    class TinyConditionEncoder(MiniMaxH3ConditionEncoder):
        num_hidden_layers = qwen.LAYERS  # test fixture; production class remains 50

    def read(value):
        native = value if isinstance(value, ic.Tensor) else ic.Tensor(value)
        torch.cuda.synchronize()
        output = torch.empty(tuple(native.shape), device='cuda', dtype=ic.utils.to_torch_dtype(native.dtype))
        ic.from_torch(output).copy_(native)
        ic.sync_device()
        torch.cuda.synchronize()
        output = output.cpu()
        return output.float().numpy().copy() if output.is_floating_point() else output.numpy().copy()

    def parameters_exact(engine, weights, dtype_name, backbone=False):
        states = engine.state_dict()
        if len(states) != 1 or set(states[0]) != set(weights):
            raise AssertionError('Expected a complete single-rank native parameter set')
        for name, value in states[0].items():
            native = value if isinstance(value, ic.Tensor) else ic.Tensor(value)
            expected_dtype = ic.float32 if backbone and h3.preserve_fp32(name) else getattr(ic, dtype_name)
            if native.dtype != expected_dtype or not np.array_equal(read(native), weights[name]):
                raise AssertionError(f'Native parameter dtype/shape/value mismatch: {name}')
        return len(weights)

    report['environment'] = {
        'python': sys.executable, 'torch': torch.__version__, 'hip': torch.version.hip,
        'transformers': transformers.__version__, 'device': torch.cuda.get_device_name(0),
        'native_device': str(device._underlying.type), 'libraries': loaded_libraries(),
        'sources': save_repository_evidence(Path(os.environ['WORK_HOME']), path.parent),
        'infinilm_extension': {'path': _infinilm.__file__, 'sha256': file_hash(Path(_infinilm.__file__))}}
    print(json.dumps(report['environment'], indent=2), flush=True)
    report['status'] = 'running'
    write_report(path, report)

    for dtype_name in qwen.DTYPES:
        dtype = getattr(torch, dtype_name)
        def q(array):
            # Reference dtype rounding on CPU; not a device inference operation.
            return torch.from_numpy(np.asarray(array, np.float32).copy()).to(dtype).float().numpy().copy()
        ec, hc = make_configs(dtype_name)
        ew, hw = qwen.make_weights(ec, q), make_backbone_weights(hc, q)
        record = {'dtype': dtype_name, 'status': 'running', 'runs': []}
        report['models'].append(record)
        def stage(name):
            record['stage'] = name
            write_report(path, report)
            print(f'{dtype_name}: {name}', flush=True)
        stage('fixtures')
        folder = path.parent/dtype_name
        folder.mkdir()
        record['checkpoints'] = {}
        for kind, config, weights in (('encoder', ec, ew), ('h3', hc, h3.checkpoint_layout(hw, hc, 'diffusers'))):
            destination = folder/kind
            destination.mkdir()
            (destination/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
            tensors = {k: torch.from_numpy(v.copy()).to(
                torch.float32 if kind == 'h3' and h3.preserve_fp32(k) else dtype).contiguous()
                for k, v in weights.items()}
            save_file(tensors, str(destination/'model.safetensors'))
            record['checkpoints'][kind] = {'path': str(destination), 'config': config,
                                           'sha256': file_hash(destination/'model.safetensors')}
            del tensors
        stage('load_native_models')
        encoder = TinyConditionEncoder(folder/'encoder', device='hygon', device_id=0,
                                       tp_size=1, dtype=dtype, max_cache_len=128)
        encoder.eval()
        engine = InferEngine(str(folder/'h3'), device=device, distributed_config=DistConfig(tp_size=1),
                             enable_graph_compiling=False, weight_load_mode='sync', pre_transpose=False)
        load_model_state_dict_by_file(engine, str(folder/'h3'), dtype=engine.dtype)
        stage('read_back_native_parameters')
        record['encoder_parameters_exact'] = parameters_exact(encoder.engine, ew, dtype_name)
        record['h3_parameters_exact'] = parameters_exact(engine, hw, dtype_name, backbone=True)
        print(f'  parameters_exact: encoder={len(ew)}, h3={len(hw)}', flush=True)

        for index, case in enumerate(make_cases(ec, q)):
            run = {'name': case['name'], 'status': 'running', 'encoder_sequence': int(case['input_ids'].size)}
            record['runs'].append(run)
            stage(case['name']+'/encoder_reference')
            expected_encoder = qwen.encoder_reference(ec, ew, case, q)
            values = {k: torch.from_numpy(case[k].copy()).to('cuda') for k in ('input_ids','mm_token_type_ids')}
            values['attention_mask'] = torch.ones_like(values['input_ids'])
            for kind, pixels, grid in ((1, 'pixel_values', 'image_grid_thw'),
                                       (2, 'pixel_values_videos', 'video_grid_thw')):
                media = [m for m in case['media'] if m['kind'] == kind]
                if media:
                    values[pixels] = torch.from_numpy(np.concatenate([m['pixels'] for m in media])).to(device='cuda', dtype=dtype)
                    values[grid] = torch.tensor([m['grid'] for m in media], dtype=torch.int64)
            originals = {k: v.clone() for k, v in values.items()}
            stage(case['name']+'/native_encoder')
            torch.cuda.synchronize()
            with torch.no_grad():
                output = encoder(**values)
            ic.sync_device()
            torch.cuda.synchronize()
            encoded = output.last_hidden_state
            if (len(output.hidden_states) != qwen.LAYERS+1
                    or output.hidden_states[qwen.LAYERS] is not encoded
                    or any(v is not None for v in output.hidden_states[:qwen.LAYERS])):
                raise AssertionError('Encoder facade selected hidden-state contract mismatch')
            if tuple(encoded.shape) != expected_encoder.shape or encoded.dtype != dtype or encoded.device.type != 'cuda':
                raise AssertionError('Encoder output shape/dtype/device contract mismatch')
            actual_encoder = encoded.detach().cpu().float().numpy().copy()
            # Reference inputs are diagnostic only. No CPU encoding is fed into the native H3.
            x = make_bridge_inputs(hc, actual_encoder, index, q)
            expected_x = dict(x, encoder_hidden_states=expected_encoder)
            run['h3_sequence'] = int(x['position_ids'].size)
            stage(case['name']+'/h3_references')
            conditional = h3.model_reference(hc, hw, x, q)
            end_to_end = h3.model_reference(hc, hw, expected_x, q)
            run['condition_sensitivity'] = check_condition_sensitivity(
                hc, hw, expected_x, q, end_to_end, h3.TOLERANCES[dtype_name])
            owners = {}
            for name, value in x.items():
                if name == 'encoder_hidden_states':
                    # Real native Encoder output stays on the device; keep its owner alive.
                    owners[name] = encoded.contiguous()
                else:
                    target_dtype = torch.int64 if value.dtype == np.int64 else (
                        dtype if name == 'rotary_cos_sin_cache' else torch.float32)
                    owners[name] = torch.from_numpy(value.copy()).to(device='cuda', dtype=target_dtype)
            native_inputs = {name: ic.from_torch(owner) for name, owner in owners.items()}
            if not np.array_equal(read(native_inputs['encoder_hidden_states']), actual_encoder):
                raise AssertionError('Encoder -> H3 handoff changed values')
            run['handoff'] = {'values_exact': True, 'shape': list(encoded.shape), 'dtype': str(dtype),
                              'device': str(encoded.device), 'source': 'native_encoder_output',
                              'cpu_roundtrip_on_inference_path': False,
                              'contiguous_copy_required': owners['encoder_hidden_states'].data_ptr() != encoded.data_ptr()}
            stage(case['name']+'/native_h3')
            torch.cuda.synchronize()
            native_output = engine.forward_raw(None, **native_inputs)
            ic.sync_device()
            torch.cuda.synchronize()
            actual = {'video': read(native_output['logits']), 'audio': read(native_output['hidden_states'])}
            artifact = folder/f'case_{index}.npz'
            arrays = dict(x)
            arrays.update(encoder_actual=actual_encoder, encoder_reference=expected_encoder,
                          input_ids=case['input_ids'], mm_token_type_ids=case['mm_token_type_ids'],
                          encoder_position_ids=case['positions'])
            for i, media in enumerate(case['media']):
                arrays.update({f'media_{i}_pixels': media['pixels'], f'media_{i}_grid': np.array(media['grid']),
                               f'media_{i}_bounds': np.array(media['bounds']), f'media_{i}_kind': np.array(media['kind'])})
            for kind in ('video','audio'):
                arrays.update({f'{kind}_actual': actual[kind], f'{kind}_conditional_reference': conditional[kind],
                               f'{kind}_chain_reference': end_to_end[kind]})
            np.savez(artifact, **arrays)
            run['arrays'] = {'path': str(artifact), 'sha256': file_hash(artifact)}
            stage(case['name']+'/compare')
            run['encoder_error'] = compare_arrays(actual_encoder, expected_encoder, *qwen.TOLERANCES[dtype_name])
            run['conditional_h3_error'], run['chain_error'] = {}, {}
            for kind in ('video','audio'):
                run['conditional_h3_error'][kind] = compare_arrays(actual[kind], conditional[kind], *h3.TOLERANCES[dtype_name])
                run['chain_error'][kind] = compare_arrays(actual[kind], end_to_end[kind], *h3.TOLERANCES[dtype_name])
            for name, value in native_inputs.items():
                if not np.array_equal(read(value), x[name]):
                    raise AssertionError(f'H3 input modified: {name}')
            for name, value in values.items():
                if not torch.equal(value, originals[name]):
                    raise AssertionError(f'Encoder input modified: {name}')
            run.update(status='passed', inputs_unchanged=True)
            write_report(path, report)
            print(f'  PASS {case["name"]} encoder_S={run["encoder_sequence"]} H3_S={run["h3_sequence"]}'
                  f' video_chain_max_abs={run["chain_error"]["video"]["max_abs"]:.6g}'
                  f' audio_chain_max_abs={run["chain_error"]["audio"]["max_abs"]:.6g} handoff_exact=True', flush=True)
            del native_output, native_inputs, owners, encoded, output, originals, values
        record.update(status='passed', stage='complete')
        write_report(path, report)
        del encoder, engine
        gc.collect()
    report['status'] = 'passed'
    write_report(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', help='show scope without importing GPU dependencies')
    args = parser.parse_args()
    scope = {'synthetic_model': True, 'encoder_selected_layers': qwen.LAYERS,
             'production_encoder_selected_layers': 50, 'h3_layers': 1, 'h3_token_refiner_layers': 1,
             'encoder_sequences': [7,11,9,5], 'dtypes': qwen.DTYPES, 'h3_checkpoint_layout': 'diffusers',
             'native_chain_forwards': 12, 'tp_size': 1, 'graph_tested': False,
             'fixture_attention_gate_bias_offset': 1.0,
             'zero_and_inverted_condition_rejection_required': True,
             'production_cache_generation_tested': False, 'full_pipeline_tested': False,
             'official_model_tested': False, 'performance_tested': False,
             'encoder_tolerances': qwen.TOLERANCES, 'conditional_h3_tolerances': h3.TOLERANCES,
             'chain_tolerances': h3.TOLERANCES}
    if args.describe:
        print(json.dumps(scope, indent=2))
        return 0
    if not os.environ.get('LOG_DIR') or not os.environ.get('WORK_HOME'):
        parser.error('Source $WORK_HOME/setup_env.sh first: LOG_DIR and WORK_HOME are required')
    folder = Path(os.environ['LOG_DIR'])/f'h3_encoder_bridge_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    folder.mkdir(parents=True)
    path = folder/'report.json'
    report = dict(scope, status='initializing', models=[], hostname=socket.gethostname(),
                  started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), command=[sys.executable, *sys.argv],
                  script_hashes={name: file_hash(Path(__file__).with_name(name)) for name in SCRIPT_FILES})
    write_report(path, report)
    print(f'report: {path}', flush=True)
    try:
        run_device(report, path)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if report['models']:
            record = report['models'][-1]
            record['status'] = 'failed'
            if record['runs'] and record['runs'][-1]['status'] == 'running':
                record['runs'][-1]['status'] = 'failed'
        write_report(path, report)
        traceback.print_exc()
        print(f'FAILED: encoder -> H3 bridge; report={path}', flush=True)
        return 1
    print(f'PASSED: 3 model pairs, 12 native Encoder -> H3 forwards; report={path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
