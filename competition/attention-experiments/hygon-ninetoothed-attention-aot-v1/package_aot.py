"""Build flat source-only offline upload; do not mutate the passed JIT package."""
import ast
import argparse
import json
from pathlib import Path
import zipfile

from attention_contract import sha256,verify_upstream,verify_candidate_attention,write_json

HERE=Path(__file__).resolve().parent
NAMES=['attention_contract.py','attention_worker.py','probe_primitives.py','run_attention_probe.py','llvm_compat.py',
       'upstream_attention.py','candidate_attention.py','install_ninetoothed_gfx936.py','install_infinilm_private_runtime.py',
       'aot_contract.py','aot_export.py','aot_launcher.cpp','aot_native.py','aot_worker.py','run_attention_aot.py',
       'run_hygon_attention_aot.sh','LICENSE-InfiniCore-MIT.txt','UPLOAD.md']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    for name in NAMES:
        p=HERE/name
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=name)
        if p.suffix in ('.py','.sh','.cpp') and b'\r\n' in p.read_bytes():raise ValueError('non-LF source: '+name)
    original=HERE.parent/'hygon-ninetoothed-attention-probe-v1'
    reused=NAMES[:9]+['LICENSE-InfiniCore-MIT.txt']
    for name in reused:
        if (HERE/name).read_bytes()!=(original/name).read_bytes():raise ValueError('reused helper/source drift: '+name)
    manifest=(args.output_dir if args.output_dir else HERE)/'package-manifest.json'
    write_json(manifest,dict(version='1.1-nine-attention-aot-dynamic-lds',
        validation_scope='source-upload-not-device-verified',upstream=verify_upstream(HERE/'upstream_attention.py'),
        candidate=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py'),
        files={name:sha256(HERE/name) for name in NAMES}))
    archive=(args.output_dir if args.output_dir else HERE.parent)/'hygon-ninetoothed-attention-aot-v1.1-upload.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in NAMES:bundle.write(HERE/name,name)
        bundle.write(manifest,'package-manifest.json')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name in NAMES:assert bundle.read(name)==(HERE/name).read_bytes()
        assert bundle.read('package-manifest.json')==manifest.read_bytes()
    archive.with_suffix('.zip.sha256').write_text(sha256(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),members=len(NAMES)+1,bytes=archive.stat().st_size,sha256=sha256(archive))))


if __name__=='__main__':main()
