"""Read-only gate: require exactly the already-installed integration source patch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    manifest=json.loads(Path(__file__).with_name('patch_manifest.json').read_text())
    root=Path(os.environ['WORK_HOME'])/'Projects'
    def digest(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for name,expected in manifest['commits'].items():
        actual=subprocess.check_output(['git','-C',str(root/name),'rev-parse','HEAD'],text=True).strip()
        if actual!=expected:raise RuntimeError('Unexpected baseline commit: '+name)
    for entry in manifest['files']+manifest['prerequisites']:
        path=root/entry['repo']/entry['path']
        if digest(path)!=entry.get('after',entry.get('sha256')):
            raise RuntimeError('Current source differs from validated integration: '+str(path))
    add=root/'InfiniCore/src/infiniop/ops/add/cuda/kernel.cuh'
    if 'return __fadd_rd(a, b);' not in add.read_text():
        raise RuntimeError('Original FP32 add source is not the expected directed-rounding implementation')
    print('Existing integration sources verified; FP32 add source SHA256:',digest(add),flush=True)


if __name__=='__main__':main()
