"""Export NineToothed's generated function and call it through a host-only C++ adapter."""
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess

from abi import validate_abi
from common import sha256, write_json

PATCHED_DRIVER_SHA256 = '0c09ac376412d1ec376652a44f72ecbc5e9497c63904cd9544f626142a691bda'


class TensorArg(ct.Structure):
    _fields_ = [('pointer', ct.c_void_p), ('rows', ct.c_int64), ('width', ct.c_int64),
                ('stride0', ct.c_int64), ('stride1', ct.c_int64)]


def build_adapter(folder):
    import triton.backends.amd.driver as driver
    source_dir = Path(__file__).resolve().parent
    driver_dir = Path(driver.__file__).resolve().parent
    if sha256(driver_dir / 'driver.c') != PATCHED_DRIVER_SHA256:
        raise RuntimeError('Requires the already verified Triton 3.8 DTK property repair')
    hip_path = Path(driver._get_path_to_hip_runtime_dylib()).resolve(strict=True)
    loaded = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if ('libamdhip64' in line or 'libgalaxyhip' in line) and line.split()[-1].startswith('/')}
    if loaded != {hip_path}:
        raise RuntimeError(f'HIP library mismatch: selected={hip_path}, loaded={loaded}')
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ is required for the host launcher')
    binary = folder / 'libh3_gated_host.so'
    command = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra',
               '-I' + str(driver_dir / 'include'), str(source_dir / 'native_launcher.cpp'),
               '-ldl', '-o', str(binary)]
    write_json(folder / 'host_build.json', {'command': command, 'hip_path': str(hip_path),
                'driver_c_sha256': sha256(driver_dir / 'driver.c'),
                'source_sha256': sha256(source_dir / 'native_launcher.cpp')})
    print('HOST BUILD:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, timeout=120)
    return binary, hip_path


def export_kernel(handle, dtype_name, folder):
    import triton
    from triton.compiler import ASTSource
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(handle._source)
    shutil.copyfile(source, folder / 'ninetoothed_generated.py')
    kernel = handle._kernel
    names = list(kernel.arg_names)
    suffixes = ('_pointer', '_size_0', '_size_1', '_stride_0', '_stride_1')
    if len(names) != 20 or any(not name.endswith(suffixes[i % 5]) for i, name in enumerate(names)):
        raise RuntimeError(f'Unexpected NineToothed generated signature: {names}')
    pointer_type = {'float32': '*fp32', 'float16': '*fp16', 'bfloat16': '*bf16'}[dtype_name]
    # No constexpr dimensions, automatic value specialization or alignment assumptions.
    signature = {name: pointer_type if i % 5 == 0 else 'i64' for i, name in enumerate(names)}
    target = triton.runtime.driver.active.get_current_target()
    options = dict(num_warps=4, num_stages=1, num_ctas=1, enable_fp_fusion=False)
    write_json(folder / 'compile_request.json', {'signature': signature, 'constexprs': {},
                'attrs': {}, 'options': options, 'target': str(target), 'source_sha256': sha256(source)})
    compiled = triton.compile(ASTSource(fn=kernel, signature=signature, constexprs={}, attrs={}),
                              target=target, options=options)
    metadata = compiled.metadata._asdict()
    metadata['target'] = dict(backend=target.backend, arch=target.arch, warp_size=target.warp_size)
    write_json(folder / 'metadata.json', metadata)
    for extension in ('hsaco', 'ttir', 'ttgir', 'llir', 'amdgcn'):
        data = compiled.asm[extension]
        path = folder / ('kernel.' + extension)
        path.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
    abi = validate_abi(compiled.asm['amdgcn'], metadata)
    info = {'dtype': dtype_name, 'kernel': metadata['name'], 'abi': abi,
            'hsaco_sha256': sha256(folder / 'kernel.hsaco'),
            'generated_sha256': sha256(folder / 'ninetoothed_generated.py'),
            'options': options, 'directory': str(folder)}
    write_json(folder / 'export.json', info)
    return info


class Adapter:
    def __init__(self, binary, hip_path):
        self.lib = ct.CDLL(str(binary))
        self.hip_path = os.fsencode(hip_path)
        self.handles = []
        self.lib.h3_error.argtypes = []
        self.lib.h3_error.restype = ct.c_char_p
        self.lib.h3_open.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p]
        self.lib.h3_open.restype = ct.c_void_p
        self.lib.h3_launch.argtypes = [ct.c_void_p, ct.POINTER(TensorArg), ct.c_void_p]
        self.lib.h3_launch.restype = ct.c_int
        for name in ('h3_calls', 'h3_last_stream'):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = [ct.c_void_p], ct.c_uint64
        self.lib.h3_close.argtypes, self.lib.h3_close.restype = [ct.c_void_p], ct.c_int

    def error(self):
        return RuntimeError(self.lib.h3_error().decode('utf-8', errors='replace'))

    def open(self, export):
        handle = self.lib.h3_open(self.hip_path, os.fsencode(Path(export['directory']) / 'kernel.hsaco'),
                                  export['kernel'].encode('ascii'))
        if not handle:
            raise self.error()
        self.handles.append(handle)
        return handle

    def launch(self, handle, dtype_name, tensors):
        import torch
        first = tensors[0]
        if len(tensors) != 4 or first.ndim != 2 or first.device.type != 'cuda':
            raise ValueError('Expected four 2D HIP tensors')
        for t in tensors:
            if t.ndim != 2 or t.shape != first.shape or t.device != first.device or t.dtype != getattr(torch, dtype_name):
                raise ValueError('Tensor device, shape or dtype mismatch')
        if first.device.index != torch.cuda.current_device():
            raise ValueError('Current device does not own the tensors')
        args = (TensorArg * 4)(*(TensorArg(t.data_ptr(), *t.shape, *t.stride()) for t in tensors))
        stream = torch.cuda.current_stream(first.device).cuda_stream
        if not stream:
            raise RuntimeError('This verification requires an explicit non-default stream')
        if self.lib.h3_launch(handle, args, stream) != 0:
            raise self.error()
        if self.lib.h3_last_stream(handle) != stream:
            raise RuntimeError('C++ adapter did not receive the selected stream')
        return stream

    def close(self):
        import torch
        torch.cuda.synchronize()
        while self.handles:
            handle = self.handles[-1]
            if self.lib.h3_close(handle) != 0:
                raise self.error()
            self.handles.pop()
