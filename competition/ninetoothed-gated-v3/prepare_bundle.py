#!/usr/bin/env python3
"""Build/export immutable kernel assets with the isolated candidate interpreter."""
import json
import os
from pathlib import Path
import sys
import time
from common import sha256,write_json


def main():
    home=Path(os.environ['WORK_HOME']).resolve()
    if Path(sys.prefix).resolve()!=home/'venvs/h3-ninetoothed-026-triton380':
        raise RuntimeError('Use the validated isolated candidate interpreter')
    root=home/'tools'/('h3-gated-rounding-v3-bundle-'+time.strftime('%Y%m%d_%H%M%S')+'-'+str(os.getpid()))
    root.mkdir(parents=True)
    cache=root/'cache-home';cache.mkdir()
    os.environ['HOME']=str(cache);os.environ['TRITON_CACHE_DIR']=str(root/'triton-cache')
    from jit_checks import environment
    env=environment()
    from native import build_adapter,export_kernel
    from gated_residual import make
    binary,hip=build_adapter(root)
    exports=[export_kernel(make(dtype),dtype,root/('export_'+dtype)) for dtype in ('float32','float16','bfloat16')]
    (root/'hip_library.txt').write_text(str(hip)+'\n',encoding='utf-8')
    files={p.relative_to(root).as_posix():sha256(p) for p in [binary,root/'hip_library.txt',*[root/('export_'+d)/'kernel.hsaco' for d in ('float32','float16','bfloat16')]]}
    write_json(root/'bundle.json',{'schema':1,'environment':env,'exports':exports,'files':files,
        'semantics':'FP32 mul nearest-even then add toward negative infinity; F16/BF16 unchanged',
        'revision':'v3: preserve directed signed zero through integer selection and bitcast',
        'source_hashes':{name:sha256(Path(__file__).with_name(name)) for name in ('gated_residual.py','native_launcher.cpp','native.py','abi.py')}})
    # Published only after every export has passed the ABI validator.
    pathfile=Path(os.environ['LOG_DIR'])/'bundle_path.txt'
    pathfile.write_text(str(root)+'\n',encoding='utf-8')
    print('BUNDLE:',root,flush=True)


if __name__=='__main__':main()
