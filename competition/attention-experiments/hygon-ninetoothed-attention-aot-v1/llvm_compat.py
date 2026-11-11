"""Isolated LLVM dialect bridge for the pinned vendor parser, not a GPU kernel.

Drop only GEP nuw/nusw poison/optimization promises, retaining inbounds, types,
indices and all numerical instructions. Defined original pointer computations
keep their values. The original and lowered IR are recorded before translation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

PROFILE='gep-nowrap-legacy-parser-v1'
BACKEND_SHA='e7d31b37a0a59c73e1bba6a5400af3af8a15f18b9a7d1b17b4c3644dee2c8401'
ANCHOR='        amdgcn = llvm.translate_to_asm(src, amd.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,\n'
INSERT=('        from .h3_llvm_compat import lower_for_vendor\n'
        '        src = lower_for_vendor(src)\n')
SEGMENTS=re.compile(r'"(?:\\.|[^"\\])*"|;[^\n]*|[^";]+')
GEP_FLAGS=re.compile(r'\bgetelementptr(?P<flags>(?:[ \t]+(?:inbounds|nuw|nusw)\b)+)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lower_gep_flags(text):
    counts={'nuw':0,'nusw':0}
    def lower_match(match):
        flags=match.group('flags')
        for flag in counts:counts[flag]+=len(re.findall(r'\b'+flag+r'\b',flags))
        return 'getelementptr'+re.sub(r'[ \t]+(?:nuw|nusw)\b','',flags)
    pieces=[]
    for match in SEGMENTS.finditer(text):
        token=match.group()
        pieces.append(token if token.startswith(('"',';')) else GEP_FLAGS.sub(lower_match,token))
    return ''.join(pieces),counts


def atomic_write(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.llvm-compat-',delete=False) as output:
        output.write(data)
        temp=Path(output.name)
    try:os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def lower_for_vendor(text,evidence_dir=None):
    if evidence_dir is None:
        value=os.environ.get('H3_LLVM_COMPAT_LOG_DIR')
        if not value:raise RuntimeError('LLVM compatibility evidence directory is not set')
        evidence_dir=Path(value)
    evidence_dir=Path(evidence_dir)
    lowered,counts=lower_gep_flags(text)
    digest=sha(text.encode('utf-8'))
    original_file=digest+'.original.ll'
    lowered_file=digest+'.lowered.ll'
    atomic_write(evidence_dir/original_file,text.encode('utf-8'))
    atomic_write(evidence_dir/lowered_file,lowered.encode('utf-8'))
    report=dict(profile=PROFILE,original_file=original_file,lowered_file=lowered_file,
                original_sha256=digest,lowered_sha256=sha(lowered.encode('utf-8')),removed_flags=counts,
                scope='only-GEP-nuw-nusw-flags; inbounds/types/indices/arithmetic-retained')
    payload=(json.dumps(report,indent=2)+'\n').encode('utf-8')
    atomic_write(evidence_dir/(digest+'.json'),payload)
    atomic_write(evidence_dir.parent/'llvm-compat-last.json',payload)
    print('llvm_compat='+json.dumps(dict(profile=PROFILE,removed_flags=counts,ir_sha256=digest)),flush=True)
    return lowered


def patch_vendor_backend(vendor,module_file,evidence_dir):
    backend=Path(vendor)/'triton/backends/amd/compiler.py'
    original=backend.read_bytes()
    if sha(original)!=BACKEND_SHA:
        raise ValueError('unexpected vendor AMD compiler source; compatibility patch refused')
    text=original.decode('utf-8')
    if text.count(ANCHOR)!=1:raise ValueError('LLVM translation boundary differs')
    patched=text.replace(ANCHOR,INSERT+ANCHOR,1).encode('utf-8')
    evidence_dir=Path(evidence_dir)
    atomic_write(evidence_dir/'vendor-compiler-original.py',original)
    atomic_write(evidence_dir/'vendor-compiler-patched.py',patched)
    destination=backend.parent/'h3_llvm_compat.py'
    if destination.exists():raise FileExistsError(destination)
    shutil.copyfile(module_file,destination)
    atomic_write(backend,patched)
    return dict(profile=PROFILE,original_sha256=sha(original),patched_sha256=sha(patched),
                compatibility_module_sha256=sha(Path(module_file).read_bytes()),
                scope='job-local-vendor-LLVM-text-parser-boundary-only')
