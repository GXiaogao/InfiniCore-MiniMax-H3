#!/usr/bin/env python3
"""Hygon H3 cache-layout -> MRoPE -> noncausal MHA development regression.

Synthetic cases, not official model dimensions or a benchmark. Cache generation
is test-input preparation, NOT a claimed InfiniCore implementation of that step.
Torch only owns/transfers test buffers; NumPy computes independent CPU references.
All tested rotation and attention calls go through the public InfiniCore API.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


def continuous_cache(positions, freq_dim, theta=10000.0):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or freq_dim <= 0:
        raise ValueError('Expected continuous positions [S,3] and positive F')
    exponent = np.arange(freq_dim, dtype=np.float32) / np.float32(freq_dim)
    inv_freq = np.float32(1) / np.power(np.float32(theta), exponent)
    angles = (positions[:, :, None] * inv_freq).reshape(len(positions), 3 * freq_dim)
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)


def rotary_reference(x, cache, rows, freq_dim):
    """NeoX pairs in the first 6F channels, using already quantized inputs."""
    x = np.asarray(x, dtype=np.float32)
    half = 3 * freq_dim
    if x.ndim != 3 or 2 * half > x.shape[-1]:
        raise ValueError('Expected [S,H,D] with 6F <= D')
    selected = np.asarray(cache, dtype=np.float32)[rows]
    cosine, sine = selected[:, None, :half], selected[:, None, half:2 * half]
    out = x.copy()
    lo, hi = x[:, :, :half], x[:, :, half:2 * half]
    out[:, :, :half] = lo * cosine - hi * sine
    out[:, :, half:2 * half] = hi * cosine + lo * sine
    return out


def attention_reference(q, k, v):
    """B=1 noncausal attention in FP32; consecutive Q heads share one KV head."""
    q, k, v = [np.asarray(x, dtype=np.float32).transpose(1, 0, 2) for x in (q, k, v)]
    if q.shape[0] % k.shape[0]:
        raise ValueError('Q head count must be divisible by KV head count')
    groups = q.shape[0] // k.shape[0]
    k, v = [np.repeat(x, groups, axis=0) for x in (k, v)]
    scores = (q @ k.transpose(0, 2, 1)) * np.float32(q.shape[-1] ** -0.5)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return (probabilities @ v).transpose(1, 0, 2)


def compare_arrays(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if actual.shape != expected.shape:
        raise AssertionError(f'Shape mismatch: {actual.shape} != {expected.shape}')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('Non-finite output or reference')
    error = np.abs(actual - expected)
    allowed = atol + rtol * np.abs(expected)
    metrics = {'max_abs': float(error.max()), 'mean_abs': float(error.mean()),
               'outside_tolerance': int(np.count_nonzero(error > allowed)),
               'atol': atol, 'rtol': rtol}
    if metrics['outside_tolerance']:
        raise AssertionError(f'Numerical mismatch: {metrics}')
    return metrics


def write_report(path, report):
    # Encode first: serialization errors must not leave a truncated final file.
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def run_cases(cases, execute, report, path):
    report['expected_cases'] = len(cases)
    for case in cases:
        record = {'spec': case, 'status': 'running', 'repeats': []}
        report['cases'].append(record)
        write_report(path, report)
        print(f'RUN {case["name"]}', flush=True)
        try:
            execute(case, record)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            report['status'] = 'failed'
            write_report(path, report)
            raise
        record['status'] = 'passed'
        write_report(path, report)
        print(f'PASS {case["name"]}', flush=True)
    report['status'] = 'passed'
    write_report(path, report)


def make_cases():
    # name, S, Hq, Hkv, D, F, packed QKV rows, cache-row lookup, position dtype
    specs = [
        ('h3_minimal', 1, 2, 2, 64, 8, False, 'identity', 'int64'),
        ('h3_full_rotary', 7, 2, 2, 96, 16, False, 'identity', 'int64'),
        ('h3_partial_rotary', 17, 4, 4, 128, 16, False, 'identity', 'int64'),
        ('h3_query_boundary', 129, 2, 2, 128, 16, False, 'identity', 'int64'),
        ('extension_packed_gqa', 17, 4, 2, 64, 8, True, 'reverse', 'int32'),
        ('extension_packed_rows', 9, 4, 4, 128, 16, True, 'permuted', 'int64'),
    ]
    keys = ('scenario', 'sequence', 'q_heads', 'kv_heads', 'head_dim', 'freq_dim',
            'packed', 'rows', 'position_dtype')
    return [dict(zip(keys, spec), dtype=dtype, name=f'{spec[0]}/{dtype}')
            for dtype in ('float16', 'bfloat16', 'float32') for spec in specs]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(work_home):
    result = {}
    for name in ('InfiniCore', 'InfiniLM'):
        repo = work_home / 'Projects' / name
        command = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                 capture_output=True, text=True, timeout=15)
        result[name] = {'path': str(repo), 'commit': command.stdout.strip(),
                        'git_exit_code': command.returncode}
    for name in ('mha.cc', 'mha_hygon_infiniop.cc'):
        path = work_home / 'Projects/InfiniCore/src/infinicore/ops/multi_head_attention' / name
        result[name] = {'sha256': file_hash(path) if path.is_file() else None}
    return result


def loaded_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'libinfinicore_cpp_api.so' in line:
            paths.add(line.split(maxsplit=5)[-1])
    if not paths:
        raise RuntimeError('Could not identify the loaded InfiniCore C++ library')
    return [{'path': name, 'sha256': file_hash(Path(name))} for name in sorted(paths)]


def device_executor(torch, ic, device):
    type_map = {getattr(torch, name): getattr(ic, name)
                for name in ('float16', 'bfloat16', 'float32', 'int32', 'int64')}
    rng = np.random.default_rng(20261023)

    def wrap(tensor):
        # from_torch in this baseline loses strides; retain all owners below.
        return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                    dtype=type_map[tensor.dtype], device=device)

    def read(tensor, dtype):
        target = torch.empty(tuple(tensor.shape), dtype=dtype, device='cuda')
        wrap(target).copy_(tensor)
        ic.sync_device()
        torch.cuda.synchronize()
        return target.cpu()

    def numpy(tensor):
        return tensor.detach().cpu().float().numpy()

    def execute(case, record):
        s, hq, hk, d, f = [case[k] for k in ('sequence', 'q_heads', 'kv_heads', 'head_dim', 'freq_dim')]
        dtype = getattr(torch, case['dtype'])
        rotation_tol = {'float16': .001, 'bfloat16': .02, 'float32': .00001}[case['dtype']]
        attention_tol = {'float16': .01, 'bfloat16': .025, 'float32': .0002}[case['dtype']]
        if case['packed']:
            owner = torch.empty((s, (hq + 2 * hk) * d), dtype=dtype, device='cuda')
            q, k, v = owner[:, :hq*d], owner[:, hq*d:(hq+hk)*d], owner[:, (hq+hk)*d:]
        else:
            q, k, v = [torch.empty((s, h*d), dtype=dtype, device='cuda') for h in (hq, hk, hk)]
        cache = torch.empty((s, 6*f), dtype=dtype, device='cuda')
        rows = np.arange(s, dtype=np.int64)
        if case['rows'] == 'reverse':
            rows = rows[::-1].copy()
        elif case['rows'] == 'permuted':
            rows = np.roll(rows, 3)
        positions = torch.tensor(rows, dtype=getattr(torch, case['position_dtype']), device='cuda')
        iq, ik, iv = wrap(q), wrap(k), wrap(v.view(1, s, hk, d))
        icache, ipos = wrap(cache), wrap(positions)
        cosine, sine = icache.narrow(1, 0, 3*f), icache.narrow(1, 3*f, 3*f)
        if list(cosine.stride()) != [6*f, 1] or list(sine.stride()) != [6*f, 1]:
            raise AssertionError('Cache slices lost their expected [6F,1] strides')
        if sine.data_ptr() - cosine.data_ptr() != 3*f * cache.element_size():
            raise AssertionError('sin slice does not start 3F elements after cos')
        record['layout'] = {'q_stride': list(q.stride()), 'k_stride': list(k.stride()),
                            'v_stride': list(v.stride()), 'cos_stride': list(cosine.stride()),
                            'sin_stride': list(sine.stride()), 'rotary_dim': 6*f,
                            'sin_offset_elements': 3*f, 'position_ids': rows.tolist()}
        for repeat in range(2):
            measurements = {'repeat': repeat}
            record['repeats'].append(measurements)
            # New values in the same allocations catch stale input/cache reuse.
            cpu_inputs = [torch.from_numpy((rng.standard_normal((s, h, d)) * .75).astype(np.float32)).to(dtype)
                          for h in (hq, hk, hk)]
            t = np.arange(s, dtype=np.float32)
            continuous = np.column_stack((t * .375 + repeat * .2,
                                          (t % 5) * .625 - .5 + repeat * .3,
                                          (t % 7) * -.25 + .75 - repeat * .4))
            cpu_cache = torch.from_numpy(continuous_cache(continuous, f)).to(dtype)
            for target, source in zip((q, k, v), cpu_inputs):
                target.copy_(source.reshape(target.shape))
            cache.copy_(cpu_cache)
            torch.cuda.synchronize()
            qn, kn, vn = [numpy(x) for x in cpu_inputs]
            cn = numpy(cpu_cache)
            # MRoPE outputs are rounded to the model dtype before MHA sees them.
            expected_q = numpy(torch.from_numpy(rotary_reference(qn, cn, rows, f)).to(dtype))
            expected_k = numpy(torch.from_numpy(rotary_reference(kn, cn, rows, f)).to(dtype))
            measurements['stage'] = 'mrope'
            qr, kr = ic.mrope(iq, ik, cosine, sine, ipos, d, 6*f, f, f, f, False)
            ic.sync_device()
            actual_q = numpy(read(qr, dtype)).reshape(s, hq, d)
            actual_k = numpy(read(kr, dtype)).reshape(s, hk, d)
            for name, actual, expected, original in (
                    ('mrope_q', actual_q, expected_q, qn), ('mrope_k', actual_k, expected_k, kn)):
                measurements[name] = compare_arrays(actual, expected, rotation_tol, rotation_tol)
                if not np.array_equal(actual[:, :, 6*f:], original[:, :, 6*f:]):
                    raise AssertionError(f'{name}: unrotated tail was changed')
            measurements['tail_exact'] = True
            measurements['stage'] = 'mha'
            output = ic.mha(qr.view([1, s, hq, d]), kr.view([1, s, hk, d]), iv,
                            scale=d ** -.5, is_causal=False)
            ic.sync_device()
            actual = numpy(read(output, dtype)).reshape(s, hq, d)
            measurements['mha_from_actual_rotary'] = compare_arrays(
                actual, attention_reference(actual_q, actual_k, vn), attention_tol, attention_tol)
            measurements['mha_from_reference_rotary'] = compare_arrays(
                actual, attention_reference(expected_q, expected_k, vn), attention_tol, attention_tol)
            for tensor, original in zip((q, k, v, cache), (*cpu_inputs, cpu_cache)):
                if not torch.equal(tensor.cpu().reshape(original.shape), original):
                    raise AssertionError('An out-of-place operation changed its input')
            measurements['inputs_unchanged'] = True
            measurements['stage'] = 'complete'
            print(f'  repeat={repeat} Q={measurements["mrope_q"]["max_abs"]:.6g}'
                  f' K={measurements["mrope_k"]["max_abs"]:.6g}'
                  f' attention={measurements["mha_from_reference_rotary"]["max_abs"]:.6g}'
                  ' tail_exact=True', flush=True)
    return execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-cases', action='store_true', help='list synthetic cases without loading GPU libraries')
    args = parser.parse_args()
    cases = make_cases()
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return 0
    log_dir = Path(os.environ['LOG_DIR'])
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f'h3_rotary_mha_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.json'
    report = {'scope': 'Synthetic H3 cache layout, MRoPE and noncausal MHA; no model weights',
              'cache_generation_tested_as_infinicore': False, 'graph_tested': False,
              'status': 'initializing', 'cases': [], 'script_sha256': file_hash(Path(__file__))}
    write_report(path, report)
    print(f'report: {path}', flush=True)
    try:
        import torch
        import infinicore as ic
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('Requires the Hygon competition Pod, using the normal runtime Python')
        device = ic.device('cuda', 0)
        native = device._underlying
        if native.type != type(native).Type.HYGON:
            raise RuntimeError(f'Expected native HYGON, got {native.type}')
        ic.set_device(device)
        report['environment'] = {'python': sys.executable, 'torch': torch.__version__,
                                 'numpy': np.__version__, 'hip': torch.version.hip,
                                 'device': torch.cuda.get_device_name(0), 'native_device': str(native.type),
                                 'infinicore': ic.__file__, 'libraries': loaded_libraries(),
                                 'sources': source_metadata(Path(os.environ['WORK_HOME']))}
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
