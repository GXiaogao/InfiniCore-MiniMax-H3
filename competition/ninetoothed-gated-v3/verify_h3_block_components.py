#!/usr/bin/env python3
"""Synthetic H3 modulation/residual/MLP op composition, not a full model test.

Requires verify_h3_rotary_mha.py beside this file for reporting utilities.
Device calculation uses InfiniCore. NumPy supplies independent FP32 references;
Torch is used only for test storage, transfers and dtype conversion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from verify_h3_rotary_mha import (
    compare_arrays, file_hash, loaded_libraries, run_cases, source_metadata, write_report,
)

EPS = 1e-5
LOCAL_TOL = {'float16': (.002, .002), 'bfloat16': (.02, .02), 'float32': (.00002, .0001)}
CHAIN_TOL = {'float16': (.01, .01), 'bfloat16': (.05, .05), 'float32': (.0002, .0002)}


def silu_reference(x):
    x = np.asarray(x, np.float32)
    return x / (np.float32(1) + np.exp(-x))


def rms_reference(x, weight):
    x = np.asarray(x, np.float32)
    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(EPS))) * weight


def adaln_reference(x, scale, shift, quantize):
    return quantize(quantize(x + quantize(x * scale)) + shift)


def swiglu_reference(up, gate):
    return np.asarray(up, np.float32) * silu_reference(gate)


def linear_reference(x, weight, bias, quantize):
    result = np.asarray(x, np.float32) @ np.asarray(weight, np.float32).T
    if bias is not None:
        result = result + bias
    return quantize(result)


def modulation_reference(projection, time_indices, tags, hidden):
    table = projection.reshape(-1, 6 * hidden)
    selected = table[3 * time_indices + tags][None]
    return tuple(selected[..., i*hidden:(i+1)*hidden] for i in range(6))


def chain_reference(data, quantize):
    """Independent sequence with explicit rounding at native operator boundaries."""
    h = data['hidden'].shape[-1]
    activated = quantize(silu_reference(data['temb']))
    projection = linear_reference(activated, data['mod_weight'], data['mod_bias'], quantize)
    a_shift, a_scale, a_gate, m_shift, m_scale, m_gate = modulation_reference(
        projection, data['time_indices'], data['tags'], h)
    norm1 = quantize(rms_reference(data['hidden'], data['norm1_weight']))
    adaln1 = adaln_reference(norm1, a_scale, a_shift, quantize)
    # The attention output is an explicit fixture, not recomputed from adaln1.
    residual1 = quantize(data['hidden'] + quantize(a_gate * data['attention_fixture']))
    norm2 = quantize(rms_reference(residual1, data['norm2_weight']))
    adaln2 = adaln_reference(norm2, m_scale, m_shift, quantize)
    up_gate = linear_reference(adaln2, data['up_gate_weight'], None, quantize)
    up, gate = np.split(up_gate, 2, axis=-1)
    swiglu = quantize(swiglu_reference(up, gate))
    down = linear_reference(swiglu, data['down_weight'], None, quantize)
    residual2 = quantize(residual1 + quantize(m_gate * down))
    output_projection = linear_reference(activated, data['output_weight'], data['output_bias'], quantize)
    output_table = output_projection[data['time_indices']][None]
    shift, scale = np.split(output_table, 2, axis=-1)
    normalized = quantize(rms_reference(residual2, data['norm_out_weight']))
    return {'adaln_msa': adaln1, 'attention_residual': residual1,
            'adaln_mlp': adaln2, 'mlp_residual': residual2,
            'output': adaln_reference(normalized, scale, shift, quantize)}


def make_cases():
    specs = [('minimal', 1, 64, 128, 16, 1),
             ('mixed_modalities', 17, 96, 192, 32, 2),
             ('longer_sequence', 129, 128, 256, 32, 2),
             ('zero_gates', 9, 64, 128, 16, 2),
             ('small_hidden', 7, 64, 128, 16, 2)]
    keys = ('scenario', 'sequence', 'hidden', 'intermediate', 'time_dim', 'time_count')
    return [dict(zip(keys, spec), dtype=dtype, name=f'{spec[0]}/{dtype}')
            for dtype in ('float16', 'bfloat16', 'float32') for spec in specs]


def make_inputs(case, rng, repeat):
    s, h, i, t, n = [case[k] for k in ('sequence', 'hidden', 'intermediate', 'time_dim', 'time_count')]
    def rand(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)
    data = {
        'hidden': rand((1, s, h), 1e-4 if case['scenario'] == 'small_hidden' else .5),
        'attention_fixture': rand((1, s, h), .25), 'temb': rand((n, t), .7),
        'mod_weight': rand((18*h, t), .3 / np.sqrt(t)), 'mod_bias': rand((18*h,), .15),
        'up_gate_weight': rand((2*i, h), .7 / np.sqrt(h)),
        'down_weight': rand((h, i), .7 / np.sqrt(i)),
        'output_weight': rand((2*h, t), .3 / np.sqrt(t)), 'output_bias': rand((2*h,), .15),
        'time_indices': ((np.arange(s) + repeat) % n).astype(np.int64),
        'tags': ((np.arange(s) // n + repeat) % 3).astype(np.int64),
    }
    for name in ('norm1_weight', 'norm2_weight', 'norm_out_weight'):
        data[name] = np.ones(h, np.float32) + rand((h,), .1)
    if case['scenario'] == 'zero_gates':
        for slot in (2, 5):
            data['mod_weight'].reshape(3, 6, h, t)[:, slot] = 0
            data['mod_bias'].reshape(3, 6, h)[:, slot] = 0
    return data


def device_executor(torch, ic, device):
    import infinicore.nn.functional as fn
    rng = np.random.default_rng(20261024)
    types = {getattr(torch, name): getattr(ic, name)
             for name in ('float16', 'bfloat16', 'float32', 'int64')}

    def wrap(tensor):
        return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                    dtype=types[tensor.dtype], device=device)

    def read(tensor, dtype):
        owner = torch.empty(tuple(tensor.shape), dtype=dtype, device='cuda')
        wrap(owner).copy_(tensor)
        ic.sync_device()
        torch.cuda.synchronize()
        value = owner.cpu()
        return value.numpy().copy() if dtype == torch.int64 else value.float().numpy().copy()

    def execute(case, record):
        h, i, s, n = [case[k] for k in ('hidden', 'intermediate', 'sequence', 'time_count')]
        dtype = getattr(torch, case['dtype'])
        local_tol = LOCAL_TOL[case['dtype']]
        def quantize(x):
            return torch.from_numpy(np.asarray(x, np.float32).copy()).to(dtype).float().numpy()
        owners = {}
        for repeat in range(2):
            inputs = make_inputs(case, rng, repeat)
            data = {}
            for name, value in inputs.items():
                tdtype = torch.int64 if value.dtype == np.int64 else (torch.float32 if name == 'temb' else dtype)
                cpu = torch.from_numpy(value).to(tdtype)
                if name not in owners:
                    owners[name] = torch.empty(tuple(value.shape), dtype=tdtype, device='cuda')
                owners[name].copy_(cpu)
                data[name] = cpu.numpy().copy() if tdtype == torch.int64 else cpu.float().numpy().copy()
            torch.cuda.synchronize()
            native = {name: wrap(owner) for name, owner in owners.items()}
            expected_chain = chain_reference(data, quantize)
            actual = {}
            metrics = {'repeat': repeat, 'stage': 'starting', 'local': {}, 'chain': {}}
            record['repeats'].append(metrics)

            def step(name, operation, expected, out_dtype=dtype, exact=False):
                metrics['stage'] = name
                tensor = operation()
                result = read(tensor, out_dtype)
                reference = expected()
                if exact:
                    if not np.array_equal(result, reference):
                        raise AssertionError(f'{name}: exact comparison failed')
                    metrics['local'][name] = {'exact': True}
                else:
                    tolerance = LOCAL_TOL['float32'] if out_dtype == torch.float32 else local_tol
                    metrics['local'][name] = compare_arrays(result, reference, *tolerance)
                actual[name] = result
                return tensor

            activated = step('temb_silu_fp32', lambda: fn.silu(native['temb']),
                             lambda: silu_reference(data['temb']), torch.float32)
            def cast_activated():
                if dtype == torch.float32:
                    return activated
                return ic.cast(activated, out=ic.empty(activated.shape, dtype=types[dtype], device=device))
            activated = step('temb_cast', cast_activated,
                             lambda: quantize(actual['temb_silu_fp32']), exact=True)
            proj = step('mod_projection', lambda: fn.linear(activated, native['mod_weight'], native['mod_bias']),
                        lambda: linear_reference(actual['temb_cast'], data['mod_weight'], data['mod_bias'], quantize))
            times2 = ic.add(native['time_indices'], native['time_indices'])
            times3 = ic.add(times2, native['time_indices'])
            indices = step('adaln_indices', lambda: ic.add(times3, native['tags']),
                           lambda: 3*data['time_indices'] + data['tags'], torch.int64, exact=True)
            table = proj.view([n*3, 6*h])
            selected = step('mod_embedding', lambda: fn.embedding(indices, table).unsqueeze(0),
                            lambda: actual['mod_projection'].reshape(n*3, 6*h)[actual['adaln_indices']][None], exact=True)
            chunks = []
            for slot in range(6):
                part = selected.narrow(2, slot*h, h)
                if list(part.stride())[-2:] != [6*h, 1]:
                    raise AssertionError(f'modulation slot {slot}: wrong stride {part.stride()}')
                if part.data_ptr() - selected.data_ptr() != slot*h*owners['hidden'].element_size():
                    raise AssertionError(f'modulation slot {slot}: wrong pointer offset')
                step(f'mod_slice_{slot}', lambda part=part: part,
                     lambda slot=slot: actual['mod_embedding'][..., slot*h:(slot+1)*h], exact=True)
                chunks.append(part)
            record['modulation_slice_stride'] = list(chunks[0].stride())

            def norm(name, tensor, value, weight):
                return step(name, lambda: fn.rms_norm(tensor, [h], native[weight], EPS),
                            lambda: quantize(rms_reference(value, data[weight])))

            def adaln(name, tensor, scale, shift, value, scale_value, shift_value):
                scaled = step(name+'_mul', lambda: ic.mul(tensor, scale), lambda: quantize(value*scale_value))
                added = step(name+'_add', lambda: ic.add(tensor, scaled), lambda: quantize(value+actual[name+'_mul']))
                return step(name, lambda: ic.add(added, shift), lambda: quantize(actual[name+'_add']+shift_value))

            normalized1 = norm('norm1', native['hidden'], data['hidden'], 'norm1_weight')
            adaln('adaln_msa', normalized1, chunks[1], chunks[0], actual['norm1'],
                  actual['mod_slice_1'], actual['mod_slice_0'])
            gate_attn = step('gate_attention', lambda: ic.mul(chunks[2], native['attention_fixture']),
                             lambda: quantize(actual['mod_slice_2']*data['attention_fixture']))
            residual1 = step('attention_residual', lambda: ic.add(native['hidden'], gate_attn),
                             lambda: quantize(data['hidden']+actual['gate_attention']))
            normalized2 = norm('norm2', residual1, actual['attention_residual'], 'norm2_weight')
            mlp_input = adaln('adaln_mlp', normalized2, chunks[4], chunks[3], actual['norm2'],
                             actual['mod_slice_4'], actual['mod_slice_3'])
            projection = step('up_gate', lambda: fn.linear(mlp_input, native['up_gate_weight']),
                              lambda: linear_reference(actual['adaln_mlp'], data['up_gate_weight'], None, quantize))
            up, gate = projection.narrow(2, 0, i), projection.narrow(2, i, i)
            if list(up.stride())[-2:] != [2*i, 1] or list(gate.stride())[-2:] != [2*i, 1]:
                raise AssertionError('Packed up/gate row stride is incorrect')
            record['up_gate_slice_stride'] = list(up.stride())
            swiglu = step('swiglu', lambda: fn.swiglu(up, gate),
                          lambda: quantize(swiglu_reference(actual['up_gate'][..., :i], actual['up_gate'][..., i:])))
            down = step('down', lambda: fn.linear(swiglu, native['down_weight']),
                        lambda: linear_reference(actual['swiglu'], data['down_weight'], None, quantize))
            gated_mlp = step('gate_mlp', lambda: ic.mul(chunks[5], down),
                             lambda: quantize(actual['mod_slice_5']*actual['down']))
            residual2 = step('mlp_residual', lambda: ic.add(residual1, gated_mlp),
                             lambda: quantize(actual['attention_residual']+actual['gate_mlp']))
            output_proj = step('output_projection', lambda: fn.linear(activated, native['output_weight'], native['output_bias']),
                               lambda: linear_reference(actual['temb_cast'], data['output_weight'], data['output_bias'], quantize))
            output_table = step('output_embedding', lambda: fn.embedding(native['time_indices'], output_proj).unsqueeze(0),
                                lambda: actual['output_projection'][data['time_indices']][None], exact=True)
            shift, scale = output_table.narrow(2, 0, h), output_table.narrow(2, h, h)
            if list(scale.stride())[-2:] != [2*h, 1]:
                raise AssertionError('Output modulation stride is incorrect')
            normalized_out = norm('norm_out', residual2, actual['mlp_residual'], 'norm_out_weight')
            adaln('output', normalized_out, scale, shift, actual['norm_out'],
                  actual['output_embedding'][..., h:], actual['output_embedding'][..., :h])
            metrics['stage'] = 'independent_chain_reference'
            for name, expected in expected_chain.items():
                metrics['chain'][name] = compare_arrays(actual[name], expected, *CHAIN_TOL[case['dtype']])
            if case['scenario'] == 'zero_gates':
                for name in ('attention_residual', 'mlp_residual'):
                    if not np.array_equal(actual[name], data['hidden']):
                        raise AssertionError(f'{name}: zero gates must preserve residual exactly')
                metrics['zero_gate_residual_exact'] = True
            metrics['stage'] = 'input_integrity'
            for name, tensor in native.items():
                if not np.array_equal(read(tensor, owners[name].dtype), data[name]):
                    raise AssertionError(f'Input modified: {name}')
            metrics.update(inputs_unchanged=True, stage='complete')
            print(f'  repeat={repeat} stages={len(metrics["local"])}'
                  f' adaln={metrics["chain"]["adaln_msa"]["max_abs"]:.6g}'
                  f' mlp_residual={metrics["chain"]["mlp_residual"]["max_abs"]:.6g}'
                  f' output={metrics["chain"]["output"]["max_abs"]:.6g}'
                  ' inputs_unchanged=True', flush=True)
    return execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-cases', action='store_true')
    args = parser.parse_args()
    cases = make_cases()
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return 0
    log_dir = Path(os.environ['LOG_DIR'])
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f'h3_block_components_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.json'
    report = {'scope': 'Synthetic H3 primitive composition; supplied attention output; no full native block or model',
              'full_native_block_tested': False, 'performance_tested': False, 'graph_tested': False,
              'status': 'initializing', 'cases': [], 'local_tolerances': LOCAL_TOL,
              'chain_tolerances': CHAIN_TOL, 'eps': EPS,
              'script_sha256': file_hash(Path(__file__)),
              'helper_sha256': file_hash(Path(__file__).with_name('verify_h3_rotary_mha.py'))}
    write_report(path, report)
    print(f'report: {path}', flush=True)
    try:
        import torch
        import infinicore as ic
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('Requires the Hygon competition Pod and its normal runtime Python')
        if ic.use_ntops:
            raise RuntimeError('This verifier requires the existing native InfiniCore backend (use_ntops=False)')
        device = ic.device('cuda', 0)
        native = device._underlying
        if native.type != type(native).Type.HYGON:
            raise RuntimeError(f'Expected native HYGON, got {native.type}')
        ic.set_device(device)
        torch.cuda.set_device(0)
        work_home = Path(os.environ['WORK_HOME'])
        sources = source_metadata(work_home)
        for name in ('minimax_h3_transformer.cpp', 'minimax_h3_transformer.hpp'):
            source = work_home / 'Projects/InfiniLM/csrc/models/minimax_h3' / name
            sources[name] = {'sha256': file_hash(source)}
        report['environment'] = {'python': sys.executable, 'torch': torch.__version__,
                                 'hip': torch.version.hip, 'numpy': np.__version__,
                                 'device': torch.cuda.get_device_name(0), 'native_device': str(native.type),
                                 'infinicore': ic.__file__, 'libraries': loaded_libraries(), 'sources': sources}
        print(json.dumps(report['environment'], indent=2), flush=True)
        report['status'] = 'running'
        run_cases(cases, device_executor(torch, ic, device), report, path)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_report(path, report)
        traceback.print_exc()
        print(f'FAILED; report={path}', flush=True)
        return 1
    print(f'PASSED: {len(cases)} cases, 2 changing-input runs each; report={path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
