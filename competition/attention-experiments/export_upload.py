"""Verify and copy fixed source files from public Git to a fresh upload directory."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PACKAGES={'jit':'hygon-ninetoothed-attention-probe-v1','aot':'hygon-ninetoothed-attention-aot-v1','bf16':'hygon-nine-bf16-diagnostic-v1'}

def export(mode,output):
    source=ROOT/PACKAGES[mode]
    manifest=json.loads((source/'package-manifest.json').read_text(encoding='utf-8'))
    files=manifest['files']
    for name,expected in files.items():
        if Path(name).name!=name:raise ValueError('non-flat source path')
        file=source/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('source hash mismatch: '+name)
        if file.suffix=='.py':ast.parse(file.read_text(encoding='utf-8'))
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('choose a fresh destination: '+str(output))
    output.mkdir(parents=True)
    for name in [*files,'package-manifest.json']:shutil.copyfile(source/name,output/name)
    return dict(mode=mode,files=len(files)+1,output=str(output),source_hashes_verified=True,gpu_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=PACKAGES);p.add_argument('output',type=Path)
    args=p.parse_args();print(json.dumps(export(args.mode,args.output),indent=2))
