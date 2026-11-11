"""Non-numerical host glue: compile/load host adapter, validate and pass HIP arguments."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess

from attention_contract import sha256,write_json
from aot_contract import bind_launch

HERE=Path(__file__).resolve().parent


def hip_runtime():
    import triton.backends.amd.driver as driver
    if sha256(Path(driver.__file__))!='e9ac88b272c49ff043500b28d67023d0c55faac68a1aafa493b24d620055e6c7':
        raise RuntimeError('vendor HIP launch/scratch policy differs from pinned driver')
    path=Path(driver._get_path_to_hip_runtime_dylib()).resolve(strict=True)
    mapped={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
            if ('libamdhip64' in line or 'libgalaxyhip' in line) and line.split()[-1].startswith('/')}
    if mapped!={path}:raise RuntimeError(f'HIP runtime selection/mapping differs: {path}, {mapped}')
    return path,Path(driver.__file__).resolve().parent


def build_host(folder):
    hip,driver=hip_runtime()
    compiler=shutil.which('g++')
    if not compiler:raise RuntimeError('g++ is required for host-only HIP launcher')
    binary=folder/'libnine_attention_host.so'
    command=[compiler,'-std=c++17','-O2','-fPIC','-shared','-Wall','-Wextra',
             '-I'+str(driver/'include'),str(HERE/'aot_launcher.cpp'),'-ldl','-o',str(binary)]
    record=dict(command=command,hip_path=str(hip),hip_sha256=sha256(hip),
                source_sha256=sha256(HERE/'aot_launcher.cpp'),
                vendor_driver_sha256=sha256(driver/'driver.py'),host_only=True)
    write_json(folder/'host-build.json',record)
    print('HOST BUILD: '+' '.join(command),flush=True)
    subprocess.run(command,check=True,timeout=120)
    return record


def validate_arguments(schema,args,current_device):
    if len(args)!=10:raise ValueError('ten runtime arguments required')
    q,k,v,length,mask,causal,scale,out,masked,variant=args
    tensors=[q,k,v,length,mask,out]
    if any(t.device.type!='cuda' or t.device.index!=current_device for t in tensors):
        raise ValueError('tensor must belong to current HIP device')
    dtype='torch.'+schema['dtype']
    if any(str(t.dtype)!=dtype for t in (q,k,v,mask,out)) or str(length.dtype)!='torch.int32':
        raise ValueError('tensor dtype differs from export')
    if any(t.ndim!=4 for t in (q,k,v,mask,out)) or tuple(length.shape)!=(1,):
        raise ValueError('unsupported tensor ranks')
    b,hq,sq,d=tuple(q.shape); bk,hkv,sk,dk=tuple(k.shape)
    if (not all(type(x) is int and x>0 for x in (b,hq,sq,d,bk,hkv,sk,dk))
            or b!=bk or d!=dk or schema.get('head_dim') not in (64,128) or d!=schema['head_dim'] or hq%hkv
            or tuple(v.shape)!=tuple(k.shape) or tuple(out.shape)!=tuple(q.shape)
            or tuple(mask.shape)!=(1,1,1,1) or (causal and sq>sk)):
        raise ValueError('unsupported shape/GQA/mask')
    ranges=[]
    for t in tensors:
        sizes=tuple(t.shape); strides=tuple(t.stride())
        if any(type(x) is not int or x<=0 for x in strides):raise ValueError('positive strides required')
        # BSND or padded BSND viewed as BHSD, including singleton dimensions.
        if t.ndim==4 and t is not mask:
            if (strides[3]!=1 or strides[1]<sizes[3] or strides[2]<sizes[1]*strides[1]
                    or strides[0]<sizes[2]*strides[2]):raise ValueError('unsupported/overlapping BSND layout')
        span=sum((size-1)*stride for size,stride in zip(sizes,strides))+1
        if span>=2**61:raise ValueError('tensor span overflow')
        start=t.data_ptr();ranges.append((start,start+span*(4 if t is length else 2)))
    output_range=ranges[-1]
    if any(output_range[0]<end and start<output_range[1] for start,end in ranges[:-1]):
        raise ValueError('output overlaps an input')
    bind_launch(schema,args)


class Adapter:
    def __init__(self,binary,hip):
        self.lib=ct.CDLL(str(binary));self.hip=os.fsencode(hip);self.handles=[]
        for name,types,result in [
            ('error',[],ct.c_char_p),('open',[ct.c_char_p,ct.c_char_p,ct.c_char_p,ct.c_uint],ct.c_void_p),
            ('launch',[ct.c_void_p,ct.POINTER(ct.c_uint64),ct.c_uint,ct.c_uint,ct.c_void_p],ct.c_int),
            ('calls',[ct.c_void_p],ct.c_uint64),('last_stream',[ct.c_void_p],ct.c_uint64),
            ('close',[ct.c_void_p],ct.c_int)]:
            fn=getattr(self.lib,'h3_attention_'+name);fn.argtypes=types;fn.restype=result

    def fail(self):return RuntimeError(self.lib.h3_attention_error().decode(errors='replace'))

    def open(self,folder,metadata,abi):
        handle=self.lib.h3_attention_open(self.hip,os.fsencode(folder/'kernel.hsaco'),
                        metadata['name'].encode('ascii'),abi['shared_bytes'])
        if not handle:raise self.fail()
        self.handles.append(handle)
        return handle

    def launch(self,handle,schema,args):
        import torch
        validate_arguments(schema,args,torch.cuda.current_device())
        grid,values=bind_launch(schema,args)
        slots=(ct.c_uint64*len(values))(*values)
        stream=torch.cuda.current_stream().cuda_stream
        if not stream:raise ValueError('non-default explicit stream required')
        if self.lib.h3_attention_launch(handle,slots,len(values),grid,stream):raise self.fail()
        if self.lib.h3_attention_last_stream(handle)!=stream:raise RuntimeError('explicit stream lost in host adapter')
        return stream

    def calls(self,handle):return self.lib.h3_attention_calls(handle)

    def close(self):
        import torch
        torch.cuda.synchronize()
        while self.handles:
            if self.lib.h3_attention_close(self.handles[-1]):raise self.fail()
            self.handles.pop()
