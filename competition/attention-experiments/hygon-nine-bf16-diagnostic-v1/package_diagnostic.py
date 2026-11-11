"""Flat isolated diagnostic upload. Keeps all old AOT/JIT packages unchanged."""
import ast
import json
from pathlib import Path
import zipfile
from attention_contract import sha256,write_json,verify_candidate_attention

HERE=Path(__file__).resolve().parent
NAMES=['attention_contract.py','attention_worker.py','probe_primitives.py','run_attention_probe.py','llvm_compat.py',
       'upstream_attention.py','candidate_attention.py','install_ninetoothed_gfx936.py','install_infinilm_private_runtime.py',
       'aot_contract.py','aot_launcher.cpp','aot_native.py','bf16_diagnostic_contract.py','bf16_cast_dsl.py',
       'bf16_diagnostic_worker.py','run_bf16_diagnostic.py','run_hygon_bf16_diagnostic.sh',
       'LICENSE-InfiniCore-MIT.txt','UPLOAD.md']


def main():
    for name in NAMES:
        f=HERE/name
        if f.suffix=='.py':ast.parse(f.read_text(encoding='utf-8'),filename=name)
        if f.suffix in ('.py','.sh','.cpp') and b'\r\n' in f.read_bytes():raise ValueError('non-LF source '+name)
    old=HERE.parent/'hygon-ninetoothed-attention-aot-v1'
    for name in NAMES[:12]:
        if (HERE/name).read_bytes()!=(old/name).read_bytes():raise ValueError('copied helper/compute drift '+name)
    write_json(HERE/'package-manifest.json',dict(version='1.1-bf16-cast-loaded-value-diagnostic',
        scope='diagnostic-not-formal-correctness',candidate=verify_candidate_attention(HERE/'upstream_attention.py',HERE/'candidate_attention.py'),
        files={name:sha256(HERE/name) for name in NAMES}))
    archive=HERE.parent/'hygon-nine-bf16-diagnostic-v1.1-upload.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in NAMES+['package-manifest.json']:z.write(HERE/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(HERE/name).read_bytes()
    archive.with_suffix('.zip.sha256').write_text(sha256(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,members=len(NAMES)+1,sha256=sha256(archive))))


if __name__=='__main__':main()
