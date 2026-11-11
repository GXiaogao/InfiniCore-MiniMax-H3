#!/usr/bin/env python3
"""Stage-gated device checks. JIT PASS never means C++/InfiniCore integration PASS."""
import argparse
from datetime import datetime
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
import uuid
import zipfile

from common import CANDIDATE, EXPECTED, disable_fusion_source, run_child, sha256, stage_passed, write_json

STAGES = (('environment', 0), ('jit-add', 9), ('jit-gated-residual', 30))
DTYPES = ('float32', 'float16', 'bfloat16')
GATE_CASES = ((1, 1, 'contiguous', 0), (7, 64, '6h', 2),
              (9, 65, '6h', 5), (17, 513, '6h', 2), (5, 127, '6h', 5),
              (2, 64, 'strided', 2), (1, 65, 'singleton', 5),
              (7, 64, 'broadcast', 0), (3, 128, 'rounding', 2),
              (7, 64, '6h-repeat', 5))


def environment():
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Requires Linux Python 3.11 in the competition Pod; CPU/local checks cannot pass')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use the separate candidate venv interpreter')
    if os.environ.get('HSA_OVERRIDE_GFX_VERSION') or os.environ.get('TRITON_OVERRIDE_ARCH'):
        raise RuntimeError('Architecture overrides are not allowed')
    packages = {}
    for name, expected in EXPECTED.items():
        dist = metadata.distribution(name)
        origin = Path(dist.locate_file('')).resolve()
        if dist.version != expected or not origin.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f'Wrong candidate package: {name}={dist.version}; path={origin}')
        packages[name] = {'version': dist.version, 'path': str(origin)}
    import torch
    if torch.__version__.split('+')[0] != '2.9.0' or not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Expected usable vendor HIP Torch 2.9.0')
    props = torch.cuda.get_device_properties(0)
    import triton
    target = triton.runtime.driver.active.get_current_target()
    if target.backend not in ('hip', 'hcu') or str(target.arch).split(':')[0] != 'gfx906' or target.warp_size != 64:
        raise RuntimeError(f'Wrong actual target: {target}')
    actual_arch = getattr(props, 'gcnArchName', '')
    if actual_arch.split(':')[0] != 'gfx906':
        raise RuntimeError(f'Torch and Triton architecture mismatch: {actual_arch}; {target}')
    import ninetoothed.aot as aot
    aot_source = Path(aot.__file__).read_text(encoding='utf-8')
    result = {'python': sys.executable, 'python_version': platform.python_version(),
              'host': platform.node(), 'os': platform.platform(), 'glibc': platform.libc_ver(),
              'packages': packages, 'torch': torch.__version__, 'torch_path': torch.__file__,
              'hip': torch.version.hip, 'device': str(props), 'target': str(target),
              'hipcc': shutil.which('hipcc'), 'nvcc': shutil.which('nvcc'),
              'triton_libhip_path': os.environ.get('TRITON_LIBHIP_PATH'),
              'hip_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                      if 'libamdhip64' in line}),
              'aot_static_blockers': [text for needle, text in (
                  ('cuCtxGetId', 'CUDA-specific context lookup'),
                  ('"-lcuda"', 'CUDA-specific library linkage'),
                  ('param_types.append(f"{stride_type}:16")', 'Unverified stride alignment hints'))
                                      if needle in aot_source]}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def launch_with_explicit_rounding(handle, folder):
    original = Path(handle._source)
    source = original.read_text(encoding='utf-8')
    # The generated kernel is not changed. Only the host launch receives the
    # compiler's no-contraction option, which make(0.26.0) does not expose.
    kernel_name = handle._kernel.__name__
    patched = disable_fusion_source(source, kernel_name)
    folder.mkdir(parents=True, exist_ok=True)
    original_copy = folder / (kernel_name + '_original.py')
    generated = folder / (kernel_name + '.py')
    original_copy.write_text(source, encoding='utf-8', newline='\n')
    generated.write_text(patched, encoding='utf-8', newline='\n')
    spec = importlib.util.spec_from_file_location('h3_generated_' + kernel_name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    launch = getattr(module, handle._launch.__name__)
    return launch, {'kernel': kernel_name, 'original': str(original_copy),
                    'original_sha256': sha256(original_copy), 'launch_source': str(generated),
                    'launch_sha256': sha256(generated), 'enable_fp_fusion': False}


def check_values(actual, expected):
    import torch
    value = actual.detach().float().cpu()
    ref = expected.float().cpu()
    if not torch.isfinite(value).all() or not torch.isfinite(ref).all():
        raise AssertionError('Nonfinite or unwritten output')
    error = (value - ref).abs()
    metrics = {'max_abs': error.max().item(), 'mean_abs': error.mean().item(),
               'mismatch_count': int(torch.count_nonzero(value != ref)),
               'atol': 0, 'rtol': 0}
    if metrics['mismatch_count']:
        raise AssertionError(f'Rounding-preserving result mismatch: {metrics}')
    return metrics


def run_add(report, folder, report_path):
    import torch
    import numpy as np
    from gated_residual import make_add
    torch.manual_seed(20261026)
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)
        launch, proof = launch_with_explicit_rounding(make_add(dtype_name), folder / 'generated')
        report['kernels'].append(proof)
        for n, stride in ((1, 1), (513, 1), (513, 2)):
            label = f'{dtype_name}_n{n}_s{stride}'
            report['current_case'] = label
            write_json(report_path, report)
            a_host = torch.randn(n * stride).to(dtype)
            b_host = torch.randn(n * stride).to(dtype)
            a_base, b_base = a_host.cuda(), b_host.cuda()
            a, b = a_base[::stride], b_base[::stride]
            out_base = torch.full((n * stride + 2,), -123, device='cuda', dtype=dtype)
            out = out_base[1:1 + n * stride:stride]
            out.fill_(float('nan'))
            mask = torch.ones(out_base.shape, dtype=torch.bool)
            mask[1:1 + n * stride:stride] = False
            expected = (a_host[::stride].float() + b_host[::stride].float()).to(dtype)
            launch(a, b, out)
            torch.cuda.synchronize()
            metrics = check_values(out, expected)
            if not torch.equal(a_base.cpu(), a_host) or not torch.equal(b_base.cpu(), b_host):
                raise AssertionError('Input mutated')
            if not torch.all(out_base.cpu()[mask] == -123):
                raise AssertionError('Output guard overwritten')
            np.savez(folder / (label + '.npz'), actual=out.float().cpu().numpy(), expected=expected.float().numpy())
            report['cases'].append({'name': label, 'passed': True, 'error': metrics,
                                    'inputs_unchanged': True, 'guards_unchanged': True})
            write_json(report_path, report)
            print('PASS', label, metrics, flush=True)


def run_gated(report, folder, report_path, launch_factory=launch_with_explicit_rounding, make_handle=None):
    import torch
    import numpy as np
    if make_handle is None:
        from gated_residual import make as make_handle
    torch.manual_seed(20261026)
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)
        launch, proof = launch_factory(make_handle(dtype_name), folder / 'generated')
        report['kernels'].append(proof)
        for rows, width, layout, offset in GATE_CASES:
            label = f'{dtype_name}_{layout}_{rows}x{width}'
            report['current_case'] = label
            write_json(report_path, report)
            input_step = 2 if layout == 'strided' else 1
            residual_host = (torch.randn(rows, width * input_step) * .25).to(dtype)
            branch_host = (torch.randn(rows, width * input_step) * .25).to(dtype)
            gate_host = (torch.randn(1 if layout == 'broadcast' else rows, 6 * width) * .25).to(dtype)
            if layout == 'rounding':
                eps = torch.finfo(dtype).eps
                residual_host.fill_(-(1 + 2 * eps))
                gate_host.fill_(1 + eps)
                branch_host.fill_(1 + eps)
            residual_base, branch_base, gate_base = residual_host.cuda(), branch_host.cuda(), gate_host.cuda()
            residual, branch = residual_base[:, ::input_step], branch_base[:, ::input_step]
            gate = gate_base[:, offset * width:(offset + 1) * width]
            gate_ref = gate_host[:, offset * width:(offset + 1) * width]
            if layout == 'broadcast':
                gate = gate.expand(rows, width)
                gate_ref = gate_ref.expand(rows, width)
            elif layout == 'singleton':
                gate = gate.as_strided((1, width), (0, 1))
            out_base = torch.full((rows, width * 2 + 3), -123, dtype=dtype, device='cuda')
            out = out_base[:, 1:1 + 2 * width:2]
            out.fill_(float('nan'))
            guard = torch.ones(out_base.shape, dtype=torch.bool)
            guard[:, 1:1 + 2 * width:2] = False
            product = (gate_ref.float() * branch_host[:, ::input_step].float()).to(dtype)
            from rounding_reference import torch_add
            expected = torch_add(residual_host[:, ::input_step], product, dtype)
            launch(residual, gate, branch, out)
            torch.cuda.synchronize()
            metrics = check_values(out, expected)
            for gpu, host in ((residual_base, residual_host), (branch_base, branch_host), (gate_base, gate_host)):
                if not torch.equal(gpu.cpu(), host):
                    raise AssertionError('Input or parent storage mutated')
            if not torch.all(out_base.cpu()[guard] == -123):
                raise AssertionError('Output guard overwritten')
            np.savez(folder / (label + '.npz'), actual=out.float().cpu().numpy(), expected=expected.float().numpy())
            report['cases'].append({'name': label, 'passed': True, 'error': metrics,
                'shape': list(out.shape), 'strides': {'residual': list(residual.stride()),
                'gate': list(gate.stride()), 'branch': list(branch.stride()), 'output': list(out.stride())},
                'gate_storage_offset': gate.storage_offset(), 'inputs_unchanged': True, 'guards_unchanged': True})
            write_json(report_path, report)
            print('PASS', label, metrics, flush=True)


def worker(args):
    folder = args.folder / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'result.json'
    report = {'stage': args.stage, 'run_id': args.run_id, 'passed': False, 'cases': [], 'kernels': []}
    try:
        home = folder / 'cache-home'
        home.mkdir(exist_ok=True)
        os.environ['HOME'] = str(home)
        os.environ['TRITON_CACHE_DIR'] = str(folder / 'triton-cache')
        report['environment'] = environment()
        write_json(report_path, report)
        if args.stage == 'jit-add':
            run_add(report, folder, report_path)
        elif args.stage == 'jit-gated-residual':
            run_gated(report, folder, report_path)
        report['passed'] = True
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        return 1
    finally:
        write_json(report_path, report)
        print('STAGE RESULT:', report_path, 'passed=', report['passed'], flush=True)


def run_all(args):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()) + '_' + uuid.uuid4().hex[:6]
    folder = args.logs.resolve() / ('h3_ninetoothed_v2_' + run_id)
    folder.mkdir(parents=True)
    summary = {'schema': 2, 'candidate': CANDIDATE, 'run_id': run_id, 'passed': False,
               'scope': 'JIT add and original NineToothed gated residual',
               'aot_validated': False, 'infinicore_integrated': False, 'h3_regression_passed': False,
               'stages': [], 'script_hashes': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    summary_path = folder / 'summary.json'
    write_json(summary_path, summary)
    print('REPORT:', summary_path, flush=True)
    ready = True
    for stage, count in STAGES:
        if not ready:
            summary['stages'].append({'stage': stage, 'status': 'NOT_RUN', 'passed': False, 'exit_code': None})
            continue
        log_path = folder / (stage + '.log')
        command = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                   '--stage', stage, '--folder', str(folder), '--run-id', run_id]
        print('RUN:', stage, flush=True)
        with log_path.open('w', encoding='utf-8') as output:
            rc = run_child(command, output, args.timeout)
        text = log_path.read_text(encoding='utf-8', errors='replace')
        print('\n'.join(text.splitlines()[-140:]), flush=True)
        result_path = folder / stage / 'result.json'
        value = None
        if result_path.is_file():
            try:
                value = json.loads(result_path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                pass
        ready = stage_passed(rc, value, stage, run_id, count)
        summary['stages'].append({'stage': stage, 'status': 'PASS' if ready else 'FAIL', 'passed': ready,
                                  'exit_code': rc, 'expected_cases': count,
                                  'result': str(result_path.relative_to(folder)), 'log': log_path.name})
        write_json(summary_path, summary)
    summary['passed'] = len(summary['stages']) == len(STAGES) and all(x['passed'] for x in summary['stages'])
    write_json(summary_path, summary)
    # Preserve compile inputs, outputs and failure context for the following
    # HIP/C++ ABI investigation. Exclude driver helper shared libraries only.
    members = [p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.so', '.pyc')]
    write_json(folder / 'files.json', [{'path': str(p.relative_to(folder)), 'sha256': sha256(p),
                                      'size': p.stat().st_size} for p in members])
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
        for p in members + [folder / 'files.json']:
            package.write(p, arcname=folder.name + '/' + p.relative_to(folder).as_posix())
    print('EVIDENCE ZIP:', archive, flush=True)
    print('JIT CHECKS PASSED' if summary['passed'] else 'JIT CHECKS FAILED', flush=True)
    print('AOT / InfiniCore integration / H3 regression: NOT VALIDATED', flush=True)
    return 0 if summary['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_home = os.environ.get('WORK_HOME')
    parser.add_argument('--logs', type=Path, default=Path(default_home) / 'logs' if default_home else None)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--stage', choices=[s for s, _ in STAGES])
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.stage:
        if args.folder is None or not args.run_id:
            parser.error('--stage requires --folder and --run-id')
        return worker(args)
    if args.logs is None:
        parser.error('Source setup_env.sh first or supply --logs')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    return run_all(args)


if __name__ == '__main__':
    sys.exit(main())
