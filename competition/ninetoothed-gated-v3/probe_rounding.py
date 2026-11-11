#!/usr/bin/env python3
"""Run original InfiniCore and NineToothed on identical rounding-sensitive inputs."""
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
from common import sha256,write_json
from rounding_reference import add_down


def fixtures():
    # Exact FP32 values, deliberately discriminating downward from nearest rounding.
    a=np.array([1.,1.,-1.,-1.,1.,-1.,0.,-0.,0.,-0.,1.,-1.,
                2.**40,-2.**40,2.**-100,-2.**-100],np.float32)
    b=np.array([3*2.**-25,2.**-24,3*2.**-25,-3*2.**-25,-1.,1.,
                0.,0.,-0.,-0.,2.**-100,-2.**-100,1.,-1.,2.**-124,-2.**-124],np.float32)
    rng=np.random.default_rng(20261027)
    a=np.concatenate((a,rng.uniform(-1,1,1008).astype(np.float32))).reshape(4,256)
    b=np.concatenate((b,rng.uniform(-0.3,0.3,1008).astype(np.float32))).reshape(4,256)
    return a,b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--expected',choices=('nearest','down'),required=True)
    p.add_argument('--label',choices=('before','after'),required=True)
    args=p.parse_args()
    os.environ['INFINICORE_H3_GATED_BUNDLE']=str(args.bundle.resolve())
    root=Path(os.environ['LOG_DIR'])/('rounding_probe_'+args.label);root.mkdir()
    report={'passed':False,'expected_candidate':args.expected,'cases':[]}
    try:
        from verify_integration import check_bundle
        report['bundle']=check_bundle()
        import torch
        import infinicore as ic
        from infinicore.lib import _infinicore as native
        from verify_h3_rotary_mha import loaded_libraries
        device=ic.device('cuda',0)
        if not torch.version.hip or device._underlying.type!=type(device._underlying).Type.HYGON or ic.use_ntops:
            raise RuntimeError('Requires native HYGON framework runtime')
        if getattr(torch.cuda.get_device_properties(0),'gcnArchName','').split(':')[0]!='gfx906':
            raise RuntimeError('Requires actual gfx906')
        if os.environ.get('HSA_OVERRIDE_GFX_VERSION') or os.environ.get('TRITON_OVERRIDE_ARCH'):
            raise RuntimeError('Architecture overrides are not allowed')
        ic.set_device(device);torch.cuda.set_device(0)
        report['environment']={'torch':torch.__version__,'hip':torch.version.hip,'libraries':loaded_libraries(),
                               'extension_sha256':sha256(native.__file__)}
        a,b=fixtures();down=add_down(a,b);nearest=(a+b).astype(np.float32)
        if np.array_equal(down,nearest):raise AssertionError('Probe does not distinguish rounding modes')
        def wrap(x):
            return ic.strided_from_blob(x.data_ptr(),list(x.shape),list(x.stride()),dtype=ic.float32,device=device)
        def read(x):
            target=torch.empty(tuple(x.shape),device='cuda',dtype=torch.float32)
            wrap(target).copy_(x);ic.sync_device();torch.cuda.synchronize()
            return target.cpu().numpy().copy()
        for layout in ('contiguous','6h','strided'):
            owners=[torch.from_numpy(x.copy()).cuda() for x in (a,np.ones_like(a),b)]
            if layout=='6h':
                base=torch.zeros(4,6*256,device='cuda');base[:,5*256:]=1
                owners[1]=base[:,5*256:]
            if layout=='strided':
                owners=[x.repeat_interleave(2,dim=1)[:,::2] for x in owners]
            torch.cuda.synchronize();r,g,branch=map(wrap,owners)
            old=read(ic.add(r,ic.mul(g,branch)))
            before=native.gated_residual_hygon_launch_count()
            new=read(ic.gated_residual(r,g,branch))
            launches=native.gated_residual_hygon_launch_count()-before
            np.savez(root/(layout+'.npz'),residual=a,branch=b,original=old,candidate=new,
                     reference_down=down,reference_nearest=nearest)
            row={'layout':layout,'native_launches':launches,'original_equals_down':bool(np.array_equal(old.view(np.uint32),down.view(np.uint32))),
                 'original_equals_nearest':bool(np.array_equal(old,nearest)),
                 'candidate_equals_original':bool(np.array_equal(new.view(np.uint32),old.view(np.uint32))),
                 'candidate_max_abs_vs_original':float(np.max(np.abs(new.astype(float)-old.astype(float))))}
            expected=down if args.expected=='down' else nearest
            row['candidate_matches_expected']=bool(np.array_equal(new.view(np.uint32),expected.view(np.uint32)))
            different=new.view(np.uint32)!=expected.view(np.uint32)
            row['bitwise_mismatches']=int(np.count_nonzero(different))
            row['all_finite']=bool(np.isfinite(new).all() and np.isfinite(old).all())
            row['only_signed_zero_differences']=bool(different.any() and
                np.all((new[different]==0)&(expected[different]==0)))
            row['mismatch_examples']=[{
                'index':[int(x) for x in index],
                'actual_bits':f'0x{int(new.view(np.uint32)[tuple(index)]):08x}',
                'expected_bits':f'0x{int(expected.view(np.uint32)[tuple(index)]):08x}'
            } for index in np.argwhere(different)[:8]]
            report['cases'].append(row);write_json(root/'report.json',report)
            print(json.dumps(row),flush=True)
            if not row['original_equals_down'] or row['original_equals_nearest'] or launches!=1 or not row['candidate_matches_expected'] or not row['all_finite']:
                raise AssertionError('Rounding hypothesis/candidate not confirmed; stop without switching bundle')
        report['passed']=True
        return 0
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';traceback.print_exc();return 1
    finally:
        write_json(root/'report.json',report);print('ROUNDING PROBE:',root/'report.json','passed=',report['passed'],flush=True)


if __name__=='__main__':raise SystemExit(main())
