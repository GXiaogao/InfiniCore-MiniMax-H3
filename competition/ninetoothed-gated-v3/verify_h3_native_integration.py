#!/usr/bin/env python3
"""Run a synthetic, tiny native H3 Transformer through the existing InferEngine.

Not an official model/configuration or a video-generation benchmark. All fixtures
and reports stay in LOG_DIR. The sibling block-components and rotary scripts
provide CPU reference arithmetic and evidence utilities, not device inference.
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

from verify_h3_block_components import rms_reference, silu_reference, adaln_reference
from verify_h3_rotary_mha import (
    continuous_cache, rotary_reference, attention_reference, compare_arrays,
    file_hash, loaded_libraries, source_metadata, write_report,
)

DTYPES = ('float32', 'float16', 'bfloat16')
TOLERANCES = {'float32': (.0003, .0003), 'float16': (.02, .02), 'bfloat16': (.08, .08)}
FP32_PREFIXES = ('proj_in.', 'audio_proj_in.', 'time_embedder.', 'proj_out.', 'audio_proj_out.')


def make_config(dtype):
    return {'_class_name': 'MiniMaxH3Transformer3DModel', 'model_type': 'minimax_h3',
            'torch_dtype': dtype, 'hidden_size': 64, 'num_layers': 1,
            'token_refiner_num_layers': 1, 'num_attention_heads': 2, 'attention_head_dim': 64,
            'ffn_hidden_size': 128, 'latents_dim': 4, 'audio_latents_dim': 3, 'text_dim': 12,
            'timestep_input_dim': 8, 'time_embed_hidden_size': 32, 'time_embed_dim': 16,
            'rope_freq_dim': 8, 'rope_theta': 10000., 'max_position_embeddings': 256,
            'patch_size': [1, 2, 2], 'norm_eps': 1e-5, 'qk_norm_eps': 1e-5, 'final_norm_eps': 1e-5}


def preserve_fp32(name):
    return name.startswith(FP32_PREFIXES)


def make_weights(c, quantize):
    """Canonical native names, independent of the checkpoint remapper."""
    rng = np.random.default_rng(2026102401)
    h, d, heads, inter = [c[k] for k in ('hidden_size', 'attention_head_dim', 'num_attention_heads', 'ffn_hidden_size')]
    inner = d*heads
    vdim = c['latents_dim'] * int(np.prod(c['patch_size']))
    weights = {}
    def put(name, array):
        a = np.asarray(array, np.float32)
        weights[name] = a if preserve_fp32(name) else quantize(a)
    def linear(name, out_dim, in_dim, bias):
        put(name+'.weight', rng.normal(0, .45/np.sqrt(in_dim), (out_dim, in_dim)))
        if bias:
            put(name+'.bias', rng.normal(0, .08, out_dim))
    def norm(name, width):
        put(name+'.weight', 1 + rng.normal(0, .08, width))
    for name, out_dim, in_dim in (
        ('proj_in', h, vdim), ('audio_proj_in', h, c['audio_latents_dim']),
        ('context_embedder', h, c['text_dim']),
        ('time_embedder.linear_1', c['time_embed_hidden_size'], c['timestep_input_dim']),
        ('time_embedder.linear_2', c['time_embed_dim'], c['time_embed_hidden_size']),
        ('norm_out.linear', 2*h, c['time_embed_dim']),
        ('proj_out', vdim, h), ('audio_proj_out', c['audio_latents_dim'], h)):
        linear(name, out_dim, in_dim, True)
    norm('token_refiner.final_norm', h)
    norm('norm_out.norm', h)
    for prefix in ('token_refiner.refiner_blocks.0', 'transformer_blocks.0'):
        norm(prefix+'.norm1', h)
        norm(prefix+'.norm2', h)
        for proj in ('q', 'k', 'v'):
            linear(prefix+f'.attn.to_{proj}', inner, h, False)
        norm(prefix+'.attn.norm_q', d)
        norm(prefix+'.attn.norm_k', d)
        linear(prefix+'.attn.to_out.0', h, inner, False)
        linear(prefix+'.ff.net.0.proj.up_proj', inter, h, False)
        linear(prefix+'.ff.net.0.proj.gate_proj', inter, h, False)
        linear(prefix+'.ff.net.2', h, inter, False)
    linear('transformer_blocks.0.adaln_proj.linear', 18*h, c['time_embed_dim'], True)
    return weights


def checkpoint_layout(weights, c, layout):
    if layout not in ('diffusers', 'legacy'):
        raise ValueError(f'Unknown fixture layout: {layout}')
    packed = {k: v.copy() for k, v in weights.items()}
    for prefix in ('token_refiner.refiner_blocks.0', 'transformer_blocks.0'):
        base = prefix+'.ff.net.0.proj.'
        up, gate = [packed.pop(base+x+'.weight') for x in ('up_proj', 'gate_proj')]
        packed[base+'weight'] = np.concatenate((up, gate) if layout == 'diffusers' else (gate, up))
        if layout == 'legacy':
            parts = [packed.pop(prefix+f'.attn.to_{x}.weight').reshape(
                c['num_attention_heads'], c['attention_head_dim'], c['hidden_size']) for x in ('q', 'k', 'v')]
            packed[prefix+'.attn.qkv_proj.weight'] = np.stack(parts, axis=1).reshape(-1, c['hidden_size'])
    if layout == 'legacy':
        converted = {}
        replacements = [('proj_in.', 'video_patch_proj.'), ('audio_proj_in.', 'audio_patch_proj.'),
                        ('context_embedder.', 'condition_proj.'),
                        ('time_embedder.linear_1.', 'time_embedder.proj_in.'),
                        ('time_embedder.linear_2.', 'time_embedder.proj_out.'),
                        ('norm_out.norm.', 'final_layer.norm.'),
                        ('norm_out.linear.', 'final_layer.adaln_proj.linear.'),
                        ('proj_out.', 'final_layer.video_out.'), ('audio_proj_out.', 'final_layer.audio_out.')]
        for name, value in packed.items():
            name = name.replace('token_refiner.refiner_blocks.', 'token_refiner.blocks.')
            name = name.replace('transformer_blocks.', 'blocks.')
            for before, after in replacements:
                if name.startswith(before):
                    name = after + name[len(before):]
                    break
            name = name.replace('.attn.norm_q.', '.attn.q_norm.').replace('.attn.norm_k.', '.attn.k_norm.')
            name = name.replace('.attn.to_out.0.', '.attn.out_proj.')
            name = name.replace('.ff.net.0.proj.', '.mlp.fc1.').replace('.ff.net.2.', '.mlp.fc2.')
            converted[name] = value
        packed = converted
    # Existing loader deliberately discards this generated buffer key.
    packed['rope.inv_freq'] = np.array([1., .1], np.float32)
    return packed


def timestep_reference(timestep, width):
    half = width//2
    frequency = np.exp(-np.float32(np.log(10000.)) * np.arange(half, dtype=np.float32)/np.float32(half))
    angles = timestep[:, None] * frequency[None]
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1)


def make_model_inputs(c, repeat, quantize):
    rng = np.random.default_rng(2026102402 + repeat)
    nv, na, nt = (3, 2, 4) if repeat != 1 else (5, 3, 5)
    s = nv+na+nt
    permutation = rng.permutation(s)
    vi, ai, ti = permutation[:nv], permutation[nv:nv+na], permutation[nv+na:]
    tags = np.empty(s, np.int64)
    tags[vi], tags[ai], tags[ti] = 0, 1, 2
    positions = rng.normal(0, 2, (s, 3)).astype(np.float32)
    def values(shape):
        return rng.normal(0, .5, shape).astype(np.float32)
    return {'video_hidden_states': values((1, nv, c['latents_dim']*int(np.prod(c['patch_size'])))),
            'audio_hidden_states': values((1, na, c['audio_latents_dim'])),
            'encoder_hidden_states': quantize(values((1, nt, c['text_dim']))),
            'timestep': np.array([.125+.1*repeat, .75+.05*repeat], np.float32),
            'timestep_indices': ((np.arange(s)+repeat) % 2).astype(np.int64),
            'token_tags': tags, 'position_ids': np.arange(s, dtype=np.int64),
            'rotary_cos_sin_cache': quantize(continuous_cache(positions, c['rope_freq_dim'], c['rope_theta'])),
            'video_indices': vi.astype(np.int64), 'audio_indices': ai.astype(np.int64), 'text_indices': ti.astype(np.int64)}


def model_reference(c, weights, x, quantize):
    """Independent NumPy full forward, with rounding at model op boundaries."""
    q = quantize
    fp32 = lambda a: np.asarray(a, np.float32)
    heads, d = c['num_attention_heads'], c['attention_head_dim']
    def linear(name, value):
        result = value @ weights[name+'.weight'].T
        if name+'.bias' in weights:
            result = result + weights[name+'.bias']
        return (fp32 if preserve_fp32(name+'.weight') else q)(result)
    def norm(name, value):
        return q(rms_reference(value, weights[name+'.weight']))
    def mlp(prefix, value):
        up = linear(prefix+'.net.0.proj.up_proj', value)
        gate = linear(prefix+'.net.0.proj.gate_proj', value)
        return linear(prefix+'.net.2', q(up*silu_reference(gate)))
    def attention(prefix, value, cache=None):
        s = value.shape[1]
        query, key, val = [linear(prefix+'.to_'+p, value).reshape(s, heads, d) for p in ('q', 'k', 'v')]
        query, key = norm(prefix+'.norm_q', query), norm(prefix+'.norm_k', key)
        if cache is not None:
            query = q(rotary_reference(query, cache, x['position_ids'], c['rope_freq_dim']))
            key = q(rotary_reference(key, cache, x['position_ids'], c['rope_freq_dim']))
        out = q(attention_reference(query, key, val)).reshape(1, s, heads*d)
        return linear(prefix+'.to_out.0', out)
    video = q(linear('proj_in', x['video_hidden_states']))
    audio = q(linear('audio_proj_in', x['audio_hidden_states']))
    text = linear('context_embedder', x['encoder_hidden_states'])
    prefix = 'token_refiner.refiner_blocks.0'
    text = q(text + attention(prefix+'.attn', norm(prefix+'.norm1', text)))
    text = q(text + mlp(prefix+'.ff', norm(prefix+'.norm2', text)))
    text = norm('token_refiner.final_norm', text)
    hidden = np.zeros((1, x['position_ids'].size, c['hidden_size']), np.float32)
    for name, value in [('video', video), ('audio', audio), ('text', text)]:
        hidden[:, x[name+'_indices']] = value
    temb = timestep_reference(x['timestep'], c['timestep_input_dim'])
    temb = linear('time_embedder.linear_1', temb)
    temb = linear('time_embedder.linear_2', silu_reference(temb))
    activated = q(silu_reference(temb))
    prefix = 'transformer_blocks.0'
    table = linear(prefix+'.adaln_proj.linear', activated).reshape(-1, 6*c['hidden_size'])
    selected = table[3*x['timestep_indices']+x['token_tags']][None]
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = np.split(selected, 6, axis=-1)
    normalized = adaln_reference(norm(prefix+'.norm1', hidden), scale_a, shift_a, q)
    attn = attention(prefix+'.attn', normalized, x['rotary_cos_sin_cache'])
    hidden = q(hidden + q(gate_a*attn))
    normalized = adaln_reference(norm(prefix+'.norm2', hidden), scale_m, shift_m, q)
    hidden = q(hidden + q(gate_m*mlp(prefix+'.ff', normalized)))
    output_modulation = linear('norm_out.linear', activated)[x['timestep_indices']][None]
    shift, scale = np.split(output_modulation, 2, axis=-1)
    normalized = adaln_reference(norm('norm_out.norm', hidden), scale, shift, q)
    return {'video': linear('proj_out', normalized)[:, x['video_indices']],
            'audio': linear('audio_proj_out', normalized)[:, x['audio_indices']]}


def run_device(report, report_path, run_dir):
    import torch
    import infinicore as ic
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from infinilm.lib import _infinilm  # canonical existing import; do not load another alias
    from safetensors.torch import save_file
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Requires the Hygon competition Pod normal runtime Python')
    device = ic.device('cuda', 0)
    if device._underlying.type != type(device._underlying).Type.HYGON or ic.use_ntops:
        raise RuntimeError('Expected HYGON native InfiniCore backend')
    torch.cuda.set_device(0)
    ic.set_device(device)
    types = {torch.float32: ic.float32, torch.float16: ic.float16,
             torch.bfloat16: ic.bfloat16, torch.int64: ic.int64}
    def wrap(owner):
        return ic.strided_from_blob(owner.data_ptr(), list(owner.shape), list(owner.stride()),
                                    dtype=types[owner.dtype], device=device)
    def read(value):
        if not isinstance(value, ic.Tensor):
            value = ic.Tensor(value)
        dtype = ic.utils.to_torch_dtype(value.dtype)
        output = torch.empty(tuple(value.shape), dtype=dtype, device='cuda')
        wrap(output).copy_(value)
        ic.sync_device()
        torch.cuda.synchronize()
        cpu = output.cpu()
        return cpu.numpy().copy() if dtype == torch.int64 else cpu.float().numpy().copy()
    work_home = Path(os.environ['WORK_HOME'])
    sources = source_metadata(work_home)
    for rel in ('python/infinilm/modeling_utils.py', 'python/infinilm/infer_engine.py',
                'csrc/models/minimax_h3/minimax_h3_transformer.cpp',
                'csrc/models/minimax_h3/minimax_h3_transformer.hpp'):
        sources[rel] = {'sha256': file_hash(work_home / 'Projects/InfiniLM' / rel)}
    report['environment'] = {'python': sys.executable, 'torch': torch.__version__, 'hip': torch.version.hip,
        'device': torch.cuda.get_device_name(0), 'native_device': str(device._underlying.type),
        'libraries': loaded_libraries(), 'sources': sources,
        'infinilm_extension': {'path': _infinilm.__file__, 'sha256': file_hash(Path(_infinilm.__file__))}}
    print(json.dumps(report['environment'], indent=2), flush=True)
    report['status'] = 'running'
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)
        def q(value):
            return torch.from_numpy(np.asarray(value, np.float32).copy()).to(dtype).float().numpy()
        c = make_config(dtype_name)
        weights = make_weights(c, q)
        format_outputs = []
        for layout in ('diffusers', 'legacy'):
            record = {'name': f'{layout}/{dtype_name}', 'status': 'running', 'stage': 'fixtures', 'runs': []}
            report['cases'].append(record)
            def checkpoint(stage):
                record['stage'] = stage
                write_report(report_path, report)
                print(f'{record["name"]}: {stage}', flush=True)
            checkpoint('fixtures')
            directory = run_dir / record['name'].replace('/', '-')
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
            tensors = {}
            for name, value in checkpoint_layout(weights, c, layout).items():
                # Infer checkpoint dtype from its naming, matching the loader's
                # preserve_fp32_prefixes. Synthetic values already quantized.
                keep = name.startswith(FP32_PREFIXES + ('video_patch_proj.', 'audio_patch_proj.',
                                                       'final_layer.video_out.', 'final_layer.audio_out.'))
                tensors[name] = torch.from_numpy(value.copy()).to(torch.float32 if keep else dtype).contiguous()
            save_file(tensors, str(directory/'model.safetensors'))
            record.update(config=c, fixture_dir=str(directory),
                          checkpoint_sha256=file_hash(directory/'model.safetensors'))
            checkpoint('create_native_engine')
            engine = InferEngine(str(directory), device=device, distributed_config=DistConfig(tp_size=1),
                                 enable_graph_compiling=False, weight_load_mode='sync', pre_transpose=False)
            keys = set(engine.state_dict_keyname())
            if keys != set(weights):
                raise AssertionError(f'Native parameter keys differ: missing={sorted(set(weights)-keys)}, extra={sorted(keys-set(weights))}')
            checkpoint('load_checkpoint_with_existing_loader')
            load_model_state_dict_by_file(engine, str(directory), dtype=engine.dtype)
            checkpoint('read_back_native_parameters')
            states = engine.state_dict()
            if len(states) != 1 or set(states[0]) != set(weights):
                raise AssertionError('Expected one complete native rank state_dict')
            for name, value in states[0].items():
                native = value if isinstance(value, ic.Tensor) else ic.Tensor(value)
                expected_dtype = ic.float32 if preserve_fp32(name) else types[dtype]
                if native.dtype != expected_dtype:
                    raise AssertionError(f'{name}: native parameter dtype mismatch')
                if not np.array_equal(read(native), weights[name]):
                    raise AssertionError(f'{name}: native parameter value/shape mismatch')
            record['parameters_exact'] = len(weights)
            print(f'  parameters_exact={len(weights)}', flush=True)
            for repeat in range(3):
                x = make_model_inputs(c, repeat, q)
                run = {'repeat': repeat, 'sequence': int(x['position_ids'].size), 'status': 'running'}
                record['runs'].append(run)
                checkpoint(f'reference_{repeat}')
                expected = model_reference(c, weights, x, q)
                owners = {}
                for name, value in x.items():
                    target = torch.int64 if value.dtype == np.int64 else (
                        dtype if name in ('encoder_hidden_states', 'rotary_cos_sin_cache') else torch.float32)
                    owners[name] = torch.from_numpy(value.copy()).to(device='cuda', dtype=target)
                native_inputs = {name: wrap(owner) for name, owner in owners.items()}
                torch.cuda.synchronize()
                checkpoint(f'native_forward_{repeat}')
                output = engine.forward_raw(None, **native_inputs)
                # Worker stream must finish before the main stream copies output.
                torch.cuda.synchronize()
                actual = {'video': read(output['logits']), 'audio': read(output['hidden_states'])}
                for kind in ('video', 'audio'):
                    run[kind] = compare_arrays(actual[kind], expected[kind], *TOLERANCES[dtype_name])
                    if layout == 'legacy' and not np.array_equal(actual[kind], format_outputs[repeat][kind]):
                        raise AssertionError(f'{kind}: equivalent checkpoint layouts produced different output')
                for name, value in native_inputs.items():
                    if not np.array_equal(read(value), x[name]):
                        raise AssertionError(f'Input modified: {name}')
                if layout == 'diffusers':
                    format_outputs.append(actual)
                run.update(status='passed', inputs_unchanged=True,
                           layout_equivalence_exact=(layout == 'legacy'))
                write_report(report_path, report)
                print(f'  PASS repeat={repeat} S={run["sequence"]}'
                      f' video={run["video"]["max_abs"]:.6g} audio={run["audio"]["max_abs"]:.6g}'
                      f' inputs_unchanged=True', flush=True)
                np.savez(directory/f'outputs_{repeat}.npz', video=actual['video'], audio=actual['audio'],
                         reference_video=expected['video'], reference_audio=expected['audio'])
            record.update(status='passed', stage='complete')
            write_report(report_path, report)
            print(f'PASS {record["name"]}', flush=True)
            del states, engine
            gc.collect()
    report['status'] = 'passed'
    write_report(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', help='show synthetic scope without GPU imports')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'config_example': make_config('float32'), 'dtypes': DTYPES,
                          'layouts': ['diffusers', 'legacy'], 'sequences': [9, 13, 9],
                          'native_forwards': 18, 'official_model': False}, indent=2))
        return 0
    directory = Path(os.environ['LOG_DIR']) / f'h3_native_integration_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    directory.mkdir(parents=True)
    path = directory/'report.json'
    report = {'status': 'initializing', 'cases': [], 'synthetic_model': True, 'tp_size': 1,
              'graph_tested': False, 'performance_tested': False, 'production_cache_generation_tested': False,
              'scope': 'Existing native MiniMaxH3Transformer on tiny synthetic config/weights; no Qwen3-VL, VAE or video generation',
              'tolerances': TOLERANCES,
              'script_hashes': {name: file_hash(Path(__file__).with_name(name)) for name in (
                  'verify_h3_native_integration.py', 'verify_h3_block_components.py', 'verify_h3_rotary_mha.py')}}
    write_report(path, report)
    print(f'report: {path}', flush=True)
    try:
        run_device(report, path, directory)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = 'failed'
        write_report(path, report)
        traceback.print_exc()
        print(f'FAILED; report={path}', flush=True)
        return 1
    print(f'PASSED: 6 native models, 18 forwards; report={path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
