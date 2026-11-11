#!/usr/bin/env python3
"""Device regression for the HYGON InfiniOp MHA fallback (not a benchmark).

Run only AFTER inspect_h3_mha_registration.py passes, using the normal server
Python, not the rejected NineToothed toolchain venv. Torch is test-only: inputs,
CPU FP32 reference, output inspection. All tested attention calls use InfiniCore.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='store_true', help='also test graph recording/replay')
    parser.add_argument('--graph-only', action='store_true', help='run only the six graph cases')
    args = parser.parse_args()
    if (args.graph or args.graph_only) and 'INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS' in os.environ:
        raise RuntimeError('Unset INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS to validate device graph capture.')
    import torch
    import infinicore as ic

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('This regression requires the Hygon competition Pod.')
    # This InfiniCore version maps native HYGON to Python's "cuda" name.
    # The CLI flag --hygon is not a valid Python device type.
    infini_device = ic.device('cuda', 0)
    native_device = infini_device._underlying
    if native_device.type != type(native_device).Type.HYGON:
        raise RuntimeError(f'cuda:0 did not resolve to native HYGON: {native_device.type}')
    ic.set_device(infini_device)
    torch.manual_seed(20261023)
    types = [(torch.float16, ic.float16, 0.01),
             (torch.bfloat16, ic.bfloat16, 0.025),
             (torch.float32, ic.float32, 0.0002)]
    records = []
    log_dir = Path(os.environ['LOG_DIR'])
    log_dir.mkdir(parents=True, exist_ok=True)
    report = log_dir / f'h3_mha_extended_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.json'
    metadata = {'python': sys.executable, 'torch': torch.__version__, 'hip': torch.version.hip,
                'device': torch.cuda.get_device_name(0), 'infinicore': ic.__file__,
                'infinicore_native_device': str(native_device.type),
                'graph_requested': args.graph or args.graph_only,
                'device_graph_segments_disabled': os.environ.get('INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS')}
    print(json.dumps(metadata, indent=2), flush=True)

    def wrap(t):
        # The baseline from_torch() loses non-contiguous strides; use the
        # explicit public blob API and retain the Torch owners in the caller.
        dtype = next(i for ttype, i, _ in types if t.dtype == ttype)
        return ic.strided_from_blob(t.data_ptr(), list(t.shape), list(t.stride()),
                                    dtype=dtype, device=infini_device)

    def tensor(shape, dtype, packed=False):
        b, s, h, d = shape
        stride = (s * h * d * 3, h * d * 3, d, 1) if packed else (s*h*d, h*d, d, 1)
        value = torch.randn(shape, dtype=torch.float32).to(dtype)
        target = torch.empty_strided(shape, stride, dtype=dtype, device='cuda')
        target.copy_(value)
        return target

    def reference(q, k, v, scale, causal):
        q, k, v = [x.detach().cpu().float().permute(0, 2, 1, 3) for x in (q, k, v)]
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale
        if causal:
            nq, nk = q.shape[2], k.shape[2]
            masked = torch.arange(nk)[None, :] > torch.arange(nq)[:, None] + nk - nq
            scores.masked_fill_(masked, -torch.inf)
        return (scores.softmax(-1) @ v).permute(0, 2, 1, 3)

    def check(name, function):
        print(f'RUN {name}', flush=True)
        record = {'name': name}
        try:
            function()
            record['status'] = 'passed'
            print(f'PASS {name}', flush=True)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        records.append(record)
        report.write_text(json.dumps({'scope': 'MHA InfiniOp correctness', 'environment': metadata, 'cases': records},
                                     indent=2), encoding='utf-8')
        # Stop at the first failure: a native asynchronous error can poison the
        # device context, and continuing would make subsequent results misleading.
        if record['status'] != 'passed':
            raise RuntimeError(f'{name} failed; details: {report}')

    # B, Sq, Sk, Hq, Hkv, D, packed. Includes the original crashing shape,
    # differing sequence lengths, GQA and both sides of the 128-query boundary.
    shapes = [(1, 1, 1, 1, 1, 64, True), (2, 5, 5, 4, 2, 128, True),
              (1, 14, 19, 8, 2, 64, False), (1, 127, 131, 2, 1, 64, True),
              (1, 129, 137, 4, 2, 128, True), (1, 257, 263, 2, 1, 64, False),
              (1, 129, 1025, 2, 1, 64, False)]

    def numeric(spec, dtype, tolerance, causal, allocated=False, graph=False, zero_axes=()):
        b, nq, nk, hq, hk, d, packed = spec
        q = tensor((b, nq, hq, d), dtype, packed)
        k = tensor((b, nk, hk, d), dtype, packed)
        v = tensor((b, nk, hk, d), dtype, packed)
        out = tensor((b, nq, hq, d), dtype, True)
        iq, ik, iv, io = map(wrap, (q, k, v, out))
        # Exercise InfiniCore's actual native view construction, not Torch's
        # unsqueeze (which chooses different singleton strides).
        def singleton_view(value):
            for axis in zero_axes:
                value = value.squeeze(axis).unsqueeze(axis)
                if value.size(axis) != 1 or value.stride(axis) != 0:
                    raise AssertionError('Expected native singleton zero stride')
            return value
        iq, ik, iv, io = map(singleton_view, (iq, ik, iv, io))
        if zero_axes:
            print(f'  native strides: q={iq.stride()} k={ik.stride()} v={iv.stride()} out={io.stride()}', flush=True)
        scale = d ** -0.5
        torch.cuda.synchronize()
        graph_obj = None
        if graph:
            ic.start_graph_recording()
            ic.mha(iq, ik, iv, scale=scale, is_causal=causal, out=io)
            graph_obj = ic.stop_graph_recording()
        for repeat in range(2):
            if repeat:
                # Detect accidental reuse of old values, including graph replay.
                q.mul_(0.6)
                v.add_(0.25)
            torch.cuda.synchronize()
            expected = reference(q, k, v, scale, causal)
            if graph_obj is not None:
                graph_obj.run()
            elif allocated:
                result = ic.mha(iq, ik, iv, scale=scale, is_causal=causal)
                io.copy_(result)
            else:
                ic.mha(iq, ik, iv, scale=scale, is_causal=causal, out=io)
            ic.sync_device()
            torch.cuda.synchronize()
            actual = out.cpu().float()
            if not torch.isfinite(actual).all():
                raise AssertionError('non-finite output')
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            print(f'  repeat={repeat} max_abs={(actual-expected).abs().max().item():.6g}', flush=True)

    for dtype, _, tolerance in types:
        if args.graph_only:
            break
        for spec in shapes:
            for causal in (False, True):
                name = f'{dtype},shape={spec},causal={causal}'
                check(name, lambda s=spec, dt=dtype, tol=tolerance, c=causal:
                      numeric(s, dt, tol, c))
        check(f'{dtype},allocated_output', lambda dt=dtype, tol=tolerance:
              numeric((1, 7, 3, 4, 2, 64, False), dt, tol, False, allocated=True))
        for spec, axes in [((1,24,24,1,1,64,False),(0,)),
                           ((2,7,7,1,1,64,False),(2,)),
                           ((2,1,1,4,2,64,False),(1,)),
                           ((1,1,1,1,1,64,False),(0,1,2))]:
            for allocated in (False, True):
                check(f'{dtype},native_singleton_axes={axes},allocated={allocated}',
                      lambda s=spec, ax=axes, a=allocated, dt=dtype, tol=tolerance:
                      numeric(s,dt,tol,False,allocated=a,zero_axes=ax))

    def rejected(kind, expected_message):
        q = tensor((1, 5, 4, 64), torch.float16)
        k = tensor((1, 5, 2, 64), torch.float16)
        v = tensor((1, 5, 2, 64), torch.float16)
        kwargs = dict(scale=0.125)
        if kind == 'causal_long_q':
            k, v = k[:, :3], v[:, :3]
            kwargs['is_causal'] = True
        elif kind == 'alibi':
            alibi = torch.ones((1, 1, 1, 4), dtype=torch.float16, device='cuda')
            kwargs['alibi_slopes'] = wrap(alibi)
        elif kind == 'heads':
            q = q[:, :, :3]
        elif kind == 'scale':
            kwargs['scale'] = float('nan')
        elif kind == 'stride':
            q = tensor((1, 5, 4, 128), torch.float16)[..., ::2]
        elif kind == 'alias':
            kwargs['out'] = wrap(q)
        elif kind == 'dtype':
            k = k.float()
        elif kind.startswith('broadcast_'):
            axis = int(kind.rsplit('_',1)[1])
            if axis == 0:
                q = q.expand(2,-1,-1,-1)
            elif axis == 1:
                q = q[:,:1].expand(-1,5,-1,-1)
            else:
                q = q[:,:,:1].expand(-1,-1,4,-1)
        torch.cuda.synchronize()
        try:
            ic.mha(wrap(q), wrap(k), wrap(v), **kwargs)
            ic.sync_device()
        except (RuntimeError, ValueError) as exc:
            if expected_message not in str(exc):
                raise AssertionError(f'wrong rejection: {exc}') from exc
        else:
            raise AssertionError('unsupported input was silently accepted')

    for kind, message in [('causal_long_q', 'Sq <= Sk'), ('alibi', 'ALiBi'),
                          ('heads', 'divisible'), ('scale', 'finite'),
                          ('stride', 'last-dimension stride'), ('alias', 'overlap'),
                          ('dtype', 'dtype'), ('broadcast_0','zero only for singleton'),
                          ('broadcast_1','zero only for singleton'),
                          ('broadcast_2','zero only for singleton')]:
        if args.graph_only:
            break
        check(f'reject:{kind}', lambda k=kind, m=message: rejected(k, m))
    if args.graph or args.graph_only:
        for dtype, _, tolerance in types:
            check(f'graph:{dtype}', lambda dt=dtype, tol=tolerance:
                  numeric((1, 129, 137, 4, 2, 64, True), dt, tol, True, graph=True))
            check(f'graph:singleton:{dtype}', lambda dt=dtype, tol=tolerance:
                  numeric((1,24,24,1,1,64,False),dt,tol,False,graph=True,zero_axes=(0,)))
    print(f'PASSED: {len(records)} cases; graph_requested={args.graph or args.graph_only}; report={report}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
