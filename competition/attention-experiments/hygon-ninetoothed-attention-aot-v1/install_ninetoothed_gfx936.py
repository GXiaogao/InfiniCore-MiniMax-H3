#!/usr/bin/env python3
"""Install an isolated offline NineToothed candidate and require real GPU JIT."""
import argparse
from email.parser import Parser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

WHEELS = {
    'ninetoothed-0.26.0-py3-none-any.whl': ('ninetoothed', '0.26.0', '7f15a35b308f00659c63508ff14f964f431234397e5dc97f73e2e4008ab51f08'),
    'sympy-1.14.0-py3-none-any.whl': ('sympy', '1.14.0', 'e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5'),
    'mpmath-1.3.0-py3-none-any.whl': ('mpmath', '1.3.0', 'a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c'),
    'numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl':
        ('numpy', '1.26.4', '666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5'),
}
TRITON_FILE = 'triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
TRITON_SHA = 'e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581'
TRITON_CACHE_SHA = '0355c3e3452fa4500122244b8fff8864f22bc05c417a3d6f5dada9f2e92f8040'
BASE_PROBE = '''import json,torch,numpy
print(json.dumps(dict(torch_version=torch.__version__,torch_file=torch.__file__,hip=torch.version.hip,
                     numpy_version=numpy.__version__,numpy_file=numpy.__file__)))'''


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def verify_wheels(folder):
    paths = []
    for filename, (name, version, digest) in WHEELS.items():
        path = folder / filename
        if not path.is_file():raise FileNotFoundError(path)
        if sha(path) != digest:raise ValueError('wheel checksum mismatch: ' + filename)
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:raise ValueError('invalid wheel metadata')
            info = Parser().parsestr(archive.read(metadata[0]).decode())
        if info['Name'] != name or info['Version'] != version:raise ValueError('wheel name/version mismatch')
        paths.append(path)
    return paths


def patch_cache(path):
    data = path.read_bytes();text = data.decode()
    old = 'CACHE_DIR = pathlib.Path.home() / ".ninetoothed"'
    if text.count(old) != 1:raise ValueError('unrecognized or already patched NineToothed cache source')
    backup = path.with_suffix('.py.before-cache-patch')
    if backup.exists():raise FileExistsError(backup)
    backup.write_bytes(data)
    text = text.replace(old, 'CACHE_DIR = pathlib.Path(__import__("os").environ["NINETOOTHED_CACHE_DIR"])')
    path.write_text(text, encoding='utf-8', newline='\n')
    return {'original_sha256': hashlib.sha256(data).hexdigest(), 'patched_sha256': sha(path),
            'scope': 'isolated-cache-location-only', 'source': str(path)}


def patch_triton_cache(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRITON_CACHE_SHA:
        raise ValueError('unrecognized or already patched vendor Triton cache source')
    text = data.decode()
    old = '''        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)'''
    if text.count(old) != 1 or text.count('        os.removedirs(temp_dir)') != 1:
        raise ValueError('vendor Triton cache anchor differs')
    text = text.replace(old, '        temp_path = os.path.join(self.cache_dir, f".tmp.pid_{pid}_{rnd_id}.{filename}")')
    text = text.replace('        os.removedirs(temp_dir)\n', '')
    backup = path.with_suffix('.py.before-cache-patch')
    if backup.exists():raise FileExistsError(backup)
    backup.write_bytes(data);path.write_text(text, encoding='utf-8', newline='\n')
    return {'original_sha256': TRITON_CACHE_SHA, 'patched_sha256': sha(path),
            'scope': 'private-vendor-cache-same-directory-atomic-replace', 'source': str(path)}


def extractor():
    spec = importlib.util.spec_from_file_location('nt_wheel_extractor', Path(__file__).with_name('install_infinilm_private_runtime.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install_candidate(wheels, root):
    if root.exists():raise FileExistsError('candidate already exists: ' + str(root))
    target = root / 'site-packages'
    extractor().install_wheels_direct(wheels, target)
    evidence = patch_cache(target / 'ninetoothed/generation.py')
    write_json(root / 'candidate.json', {'wheels': WHEELS, 'cache_patch': evidence,
        'gpu_validation_at_creation': 'NOT_RUN',
        'validation_status_source': 'state/ninetoothed-gfx936.json after successful probe'})
    return evidence


def worker_environment(candidate, vendor, cache, baseline=None):
    env = dict(os.environ if baseline is None else baseline)
    env.pop('PYTHONHOME', None);env.pop('VIRTUAL_ENV', None)
    env.update(PYTHONPATH=os.pathsep.join([str(candidate / 'site-packages'), str(vendor)]),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key, name in [('TRITON_CACHE_DIR', 'triton'), ('NINETOOTHED_CACHE_DIR', 'ninetoothed'), ('TMPDIR', 'tmp')]:
        folder = cache / name;folder.mkdir(parents=True, exist_ok=True);env[key] = str(folder)
    return env


def validate_probe(value, baseline):
    cases = value.get('cases', [])
    expected = {f'{dtype}-n{n}-stride{stride}' for dtype in ('float32','float16','bfloat16')
                for n, stride in ((1,1),(513,1),(513,2))}
    if (value.get('result') != 'JIT_VERIFIED' or value.get('arch') != 'gfx936'
            or value.get('baseline') != baseline or len(cases) != 9
            or value.get('isolated_numpy', {}).get('version') != '1.26.4'
            or {case.get('name') for case in cases} != expected or value.get('cache_copy_errors')
            or not all(case.get('passed') is True for case in cases)):
        raise ValueError('incomplete or invalid NineToothed GPU JIT report')


def run_with_heartbeat(command, env):
    stop = threading.Event()
    def pulse():
        elapsed = 0
        while not stop.wait(20):
            elapsed += 20;print(f'stage=jit-probe status=RUNNING elapsed_seconds={elapsed}', flush=True)
    thread = threading.Thread(target=pulse, daemon=True);thread.start()
    try:subprocess.run(command, env=env, check=True, timeout=900)
    finally:stop.set();thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-home', 'run-dir', 'local-root', 'candidate'):parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv);args.run_dir.mkdir(parents=True, exist_ok=True)
    report = {'version': '1.2-isolated-gfx936-ninetoothed-cache-rename', 'result': 'FAILED', 'last_stage': 'preflight',
              'state_published': False, 'integration_status': 'NOT_TESTED', 'quality_status': 'UNASSESSED'}
    start = time.monotonic()
    def checkpoint(stage):
        report['last_stage'] = stage;report['elapsed_seconds'] = time.monotonic() - start
        write_json(args.run_dir / 'ninetoothed-install.json', report)
        print('stage=' + stage, flush=True)
    try:
        checkpoint('preflight')
        if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):raise RuntimeError('requires Linux Python3.11 on competition server')
        wheels = verify_wheels(args.work_home / 'packages/ninetoothed-gfx936-inputs')
        vendor_wheel = args.work_home / 'packages/minimax-h3-hygon-runtime-inputs' / TRITON_FILE
        if sha(vendor_wheel) != TRITON_SHA:raise ValueError('vendor Triton wheel checksum mismatch')
        clean = dict(os.environ);clean.pop('PYTHONPATH', None);clean.pop('PYTHONHOME', None)
        clean.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        baseline = json.loads(subprocess.check_output([sys.executable, '-c', BASE_PROBE], env=clean, text=True, timeout=90))
        report['baseline'] = baseline;write_json(args.run_dir / 'base-stack.json', baseline)
        print('base_stack=' + json.dumps(baseline), flush=True)
        checkpoint('isolated-candidate-install')
        report['cache_patch'] = install_candidate(wheels, args.candidate)
        vendor = args.local_root / 'vendor-site-packages'
        extractor().install_wheels_direct([vendor_wheel], vendor)
        cache_source = vendor/'triton/runtime/cache.py'
        report['triton_cache_patch'] = patch_triton_cache(cache_source)
        write_json(args.run_dir/'triton-cache-patch.json', report['triton_cache_patch'])
        evidence_dir = args.run_dir/'compat';evidence_dir.mkdir()
        shutil.copyfile(cache_source, evidence_dir/'triton-cache-patched.py')
        shutil.copyfile(cache_source.with_suffix('.py.before-cache-patch'), evidence_dir/'triton-cache-original.py')
        env = worker_environment(args.candidate, vendor, args.local_root / 'cache')
        checkpoint('gpu-jit-probe')
        run_with_heartbeat([sys.executable, '-u', str(Path(__file__).with_name('ninetoothed_gfx936_probe.py')),
            '--candidate', str(args.candidate), '--vendor', str(vendor), '--run-dir', str(args.run_dir),
            '--baseline', str(args.run_dir / 'base-stack.json')], env)
        probe = json.loads((args.run_dir / 'ninetoothed-probe.json').read_text())
        validate_probe(probe, baseline)
        after = json.loads(subprocess.check_output([sys.executable, '-c', BASE_PROBE], env=clean, text=True, timeout=90))
        if after != baseline:raise RuntimeError('base Torch/NumPy identity changed')
        checkpoint('publish-verified-state')
        state = {'schema_version': 1, 'candidate': str(args.candidate), 'runtime_mode': 'isolated-candidate-job-local-vendor-triton',
                 'wheels': WHEELS, 'triton_wheel': str(vendor_wheel), 'triton_sha256': TRITON_SHA,
                 'cache_patch': report['cache_patch'], 'baseline': baseline, 'probe': probe,
                 'isolated_numpy': probe['isolated_numpy'],
                 'triton_cache_patch': report['triton_cache_patch'],
                 'run_dir': str(args.run_dir), 'scope': 'standalone-add-jit-not-infinicore-integration'}
        write_json(args.work_home / 'state/ninetoothed-gfx936.json', state)
        report.update(result='JIT_VERIFIED', state_published=True, candidate=str(args.candidate))
        checkpoint('complete')
        print('NINETOOTHED INSTALL/JIT VERIFIED; InfiniCore integration=NOT_TESTED', flush=True)
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        write_json(args.run_dir / 'ninetoothed-install.json', report)


if __name__ == '__main__':raise SystemExit(main())
