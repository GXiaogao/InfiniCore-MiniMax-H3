"""Host-only validation shared by controller and GPU worker. No compute fallback."""
import hashlib
import base64
import csv
import json
import os
from pathlib import Path

UPSTREAM_COMMIT='f32e2fbb2a9b2ca912b5711f4f3836c403196a74'
UPSTREAM_PATH='src/infiniop/ops/flash_attention/ninetoothed/flash_attention.py'
UPSTREAM_SHA='b0039b78b9693b32fec36701424a9773b72aeb0688fe5d52c58faa99f428f1ca'
SCOPE='standalone-nine-attention-jit-not-formal-integration-or-performance'
NT_WHEEL='ninetoothed-0.26.0-py3-none-any.whl'
NT_WHEEL_ID=['ninetoothed','0.26.0','7f15a35b308f00659c63508ff14f964f431234397e5dc97f73e2e4008ab51f08']
NT_RECORD_SHA='e5e2f2ef55bac36dc81cfcffe5141054fa556feceb7351fee95ed8432c4939f1'
NT_GENERATION_ORIGINAL_SHA='8863013c70fb5bdc131f1078c7b1e0ddbfdd68336a625dfb2811c781e345fe3d'
NT_GENERATION_PATCHED_SHA='17f908e7ee7ddf9e87f2a2cb51d302d6d2b32211da7735998fbced276b8cbf4e'
CEIL_PROFILE='positive-dynamic-kv-ceiling-div-v1'
CEIL_OLD='for j in range(-(-actual_kv_len // key.dtype.shape[0])):'
CEIL_NEW='for j in range((actual_kv_len + key.dtype.shape[0] - 1) // key.dtype.shape[0]):'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def verify_manifest(folder,manifest):
    folder=Path(folder).resolve()
    for relative,digest in manifest.items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder) or sha256(path)!=digest:
            raise ValueError('package checksum/path mismatch: '+relative)


def verify_upstream(path):
    digest=sha256(path)
    if digest!=UPSTREAM_SHA:
        raise ValueError('unexpected upstream NineToothed attention expression SHA256')
    return dict(commit=UPSTREAM_COMMIT,path=UPSTREAM_PATH,sha256=digest,modified=False)


def candidate_attention_source(original):
    if hashlib.sha256(original).hexdigest()!=UPSTREAM_SHA:
        raise ValueError('unknown original DSL; loop-count correction refused')
    text=original.decode('utf-8')
    if text.count(CEIL_OLD)!=1:
        raise ValueError('expected one original dynamic K/V loop bound')
    return text.replace(CEIL_OLD,CEIL_NEW,1).encode('utf-8')


def verify_candidate_attention(original_path,candidate_path):
    original=Path(original_path).read_bytes()
    expected=candidate_attention_source(original)
    actual=Path(candidate_path).read_bytes()
    if actual!=expected:raise ValueError('candidate attention DSL differs beyond declared ceiling-division repair')
    return dict(profile=CEIL_PROFILE,modified=True,original_sha256=UPSTREAM_SHA,
                candidate_sha256=hashlib.sha256(actual).hexdigest(),
                original_expression=CEIL_OLD,candidate_expression=CEIL_NEW,
                compilation='NineToothed expression and generation chain')


def verify_candidate_source(candidate):
    candidate=Path(candidate).resolve()
    metadata=json.loads((candidate/'candidate.json').read_text(encoding='utf-8'))
    if metadata.get('wheels',{}).get(NT_WHEEL)!=NT_WHEEL_ID:
        raise ValueError('candidate NineToothed wheel identity differs')
    patch=metadata.get('cache_patch',{})
    if (patch.get('original_sha256')!=NT_GENERATION_ORIGINAL_SHA
            or patch.get('patched_sha256')!=NT_GENERATION_PATCHED_SHA):
        raise ValueError('candidate cache patch identity differs')
    site=candidate/'site-packages'
    record=site/'ninetoothed-0.26.0.dist-info/RECORD'
    if sha256(record)!=NT_RECORD_SHA:
        raise ValueError('candidate wheel RECORD is not the pinned original')
    verified={}
    expected_py=set()
    with record.open(encoding='utf-8',newline='') as stream:
        for name,expected,size in csv.reader(stream):
            if not expected:continue
            path=(site/name).resolve()
            if not path.is_relative_to(site.resolve()):raise ValueError('unsafe candidate RECORD path')
            actual=sha256(path)
            if name=='ninetoothed/generation.py':
                if actual!=NT_GENERATION_PATCHED_SHA:
                    raise ValueError('NineToothed generation/cache patch drift')
            else:
                encoded=base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=')
                if expected!='sha256='+encoded or path.stat().st_size!=int(size):
                    raise ValueError('candidate source drift: '+name)
            verified[name]=actual
            if name.startswith('ninetoothed/') and name.endswith('.py'):expected_py.add(name)
    actual_py={p.relative_to(site).as_posix() for p in (site/'ninetoothed').rglob('*.py')}
    if actual_py!=expected_py:raise ValueError('unexpected NineToothed Python sources')
    return dict(wheel=NT_WHEEL,record_sha256=NT_RECORD_SHA,files=verified,
                cache_patch_scope='isolated-job-cache-only')


def read_verified_state(work):
    work=Path(work).resolve()
    state=json.loads((work/'state/ninetoothed-gfx936.json').read_text(encoding='utf-8'))
    candidate=Path(state.get('candidate','')).resolve()
    expected=work/'python-envs/ninetoothed-gfx936'
    if (not candidate.is_relative_to(expected.resolve()) or candidate==expected.resolve()
            or not (candidate/'site-packages/ninetoothed').is_dir()):
        raise ValueError('verified NineToothed private candidate missing or outside expected directory')
    probe=state.get('probe',{})
    if (probe.get('result')!='JIT_VERIFIED' or probe.get('arch')!='gfx936'
            or state.get('scope')!='standalone-add-jit-not-infinicore-integration'):
        raise ValueError('NineToothed installation has no matching gfx936 JIT evidence')
    if (state.get('wheels',{}).get(NT_WHEEL)!=NT_WHEEL_ID
            or state.get('cache_patch',{}).get('patched_sha256')!=NT_GENERATION_PATCHED_SHA):
        raise ValueError('published candidate source identity differs')
    state['candidate_integrity']=verify_candidate_source(candidate)
    state['candidate']=str(candidate)
    return state


def validate_case(case):
    for key in ('batch','sq','sk','hq','hkv','dim'):
        if type(case.get(key)) is not int or case[key]<=0:
            raise ValueError('positive integer required: '+key)
    if case['hq']%case['hkv'] or case['dim'] not in (64,128):
        raise ValueError('unsupported GQA/head dimension in this development probe')
    if type(case.get('causal')) is not bool or (case['causal'] and case['sq']>case['sk']):
        raise ValueError('causal mode requires Sq <= Sk')
    if case.get('layout') not in ('bsnd','padded-bsnd'):
        raise ValueError('unsupported layout')


def causal_key_limit(row,sq,sk):
    if not 0<=row<sq or sq>sk:
        raise ValueError('invalid lower-right causal coordinates')
    return sk-sq+row+1


def case_specs():
    # Development shapes, not claimed to be the complete real-model workload.
    variants=[('single',1,1,1,1,1,64,False,'bsnd'),
              ('tail',1,17,33,2,2,64,False,'bsnd'),
              ('gqa',1,33,65,4,2,64,False,'bsnd'),
              ('causal-right',1,17,33,2,1,64,True,'bsnd'),
              ('padded',2,33,65,4,2,64,False,'padded-bsnd'),
              ('head128',1,16,16,2,2,128,False,'bsnd')]
    result=[]
    for dtype in ('float16','bfloat16'):
        for label,b,sq,sk,hq,hkv,dim,causal,layout in variants:
            value=dict(name=dtype+'-'+label,dtype=dtype,batch=b,sq=sq,sk=sk,hq=hq,
                       hkv=hkv,dim=dim,causal=causal,layout=layout,scale=dim**-0.5)
            validate_case(value)
            result.append(value)
    return result


def validate_report(report):
    target=report.get('target',{})
    expected={case['name'] for case in case_specs()}
    cases=report.get('cases',[])
    if (report.get('result')!='JIT_PROBE_PASSED' or target.get('backend') not in ('hip','hcu')
            or target.get('arch')!='gfx936' or len(cases)!=len(expected)
            or {case.get('name') for case in cases}!=expected
            or report.get('primitives')!={'dot':True,'reduction':True}
            or report.get('aot_status')!='NOT_TESTED'
            or report.get('integration_status')!='NOT_TESTED'):
        raise ValueError('incomplete or invalid standalone attention JIT evidence')
    for case in cases:
        if (case.get('rounds')!=2 or not all(case.get(key) is True for key in
                ('passed','finite','inputs_unchanged','guards_unchanged'))):
            raise ValueError('case incomplete/failed: '+str(case.get('name')))
