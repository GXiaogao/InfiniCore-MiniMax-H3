"""Host-only launch/ABI contract. No generated code execution, GPU imports or math fallback."""
import ast
from collections import Counter
import math
import operator
from pathlib import Path
import re
import struct

from attention_contract import case_specs, sha256

SCOPE = 'standalone-nine-attention-aot-host-launch-not-framework-model-or-performance'
ROLES = ('q','k','v','length','mask','causal','scale','out','masked','variant')
OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul, ast.FloorDiv:operator.floordiv}


def _encode(node, names):
    if isinstance(node,ast.Name) and node.id in names:
        return ['input',names[node.id]]
    if isinstance(node,ast.Constant) and type(node.value) is int and 0<=node.value<=128:
        return ['int',node.value]
    if isinstance(node,ast.BinOp) and type(node.op) in OPS:
        return [type(node.op).__name__,_encode(node.left,names),_encode(node.right,names)]
    if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id in names
            and node.func.attr in ('size','stride') and len(node.args)==1 and not node.keywords
            and isinstance(node.args[0],ast.Constant) and type(node.args[0].value) is int
            and 0<=node.args[0].value<4):
        role=names[node.func.value.id]
        if role not in ('q','k','v','length','mask','out'):
            raise ValueError('size/stride requires tensor role')
        return [node.func.attr,role,node.args[0].value]
    raise ValueError('unsupported generated launch expression: '+ast.dump(node))


def launch_schema(source,kernel_name,dtype,causal):
    if dtype not in ('float16','bfloat16') or type(causal) is not bool:
        raise ValueError('unsupported specialization')
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
    kernels=[n for n in functions if n.name==kernel_name]
    launches=[n for n in functions if n.name=='launch_'+kernel_name]
    if len(kernels)!=1 or len(launches)!=1:raise ValueError('ambiguous generated functions')
    kernel,launch=kernels[0],launches[0]
    if (len(launch.args.args)!=10 or launch.args.defaults or launch.args.vararg or launch.args.kwarg
            or launch.args.kwonlyargs or len(launch.body)!=4 or launch.decorator_list):
        raise ValueError('unexpected no-cache launch profile')
    names=dict(zip([x.arg for x in launch.args.args],ROLES))
    constants={}
    for statement,role,value in zip(launch.body[:3],('causal','masked','variant'),(causal,False,2)):
        if (not isinstance(statement,ast.Assign) or len(statement.targets)!=1
                or not isinstance(statement.targets[0],ast.Name) or not isinstance(statement.value,ast.Name)
                or names.get(statement.value.id)!=role):
            raise ValueError('unexpected constexpr binding')
        flag=statement.targets[0].id
        if flag in names:raise ValueError('constexpr shadows runtime input')
        names[flag]=role; constants[flag]=value
    expression=launch.body[-1]
    if not isinstance(expression,ast.Expr) or not isinstance(expression.value,ast.Call):
        raise ValueError('expected one generated launch')
    call=expression.value
    if (not isinstance(call.func,ast.Subscript) or not isinstance(call.func.value,ast.Name)
            or call.func.value.id!=kernel_name or not isinstance(call.func.slice,ast.Lambda)):
        raise ValueError('unknown launch target/grid')
    grid=call.func.slice
    if (len(grid.args.args)!=1 or not isinstance(grid.body,ast.Tuple) or len(grid.body.elts)!=1
            or grid.args.defaults or grid.args.vararg or grid.args.kwarg):
        raise ValueError('only one-dimensional generated grid supported')
    if {x.arg:ast.literal_eval(x.value) for x in call.keywords}!={'num_warps':4,'num_stages':1}:
        raise ValueError('unexpected launch options')
    encoded_grid=_encode(grid.body.elts[0],names)
    args=kernel.args.args
    arg_names=[x.arg for x in args]
    if len(args)!=47 or len(set(arg_names))!=47 or len(call.args)!=47:
        raise ValueError('expected three constexpr and 44 dynamic parameters')
    signature={}; dynamic=[]
    for i,flag in enumerate(constants):
        if (arg_names[i]!=flag or not args[i].annotation
                or not ast.unparse(args[i].annotation).endswith('.constexpr')
                or not isinstance(call.args[i],ast.Name) or call.args[i].id!=flag):
            raise ValueError('constexpr signature differs from generated launch')
        signature[flag]='constexpr'
    pointer_type={'float16':'*fp16','bfloat16':'*bf16'}[dtype]
    index=3
    for role in ('q','k','v','mask','out','length'):
        sizes,strides=(1,1) if role=='length' else (3,4)
        root=arg_names[index].removesuffix('_pointer')
        suffixes=['_pointer']+[f'_size_{i}' for i in range(sizes)]+[f'_stride_{i}' for i in range(strides)]
        expressions=[['input',role]]+[['size',role,i] for i in range(sizes)]+[['stride',role,i] for i in range(strides)]
        for suffix,expected in zip(suffixes,expressions):
            name=arg_names[index]
            if name!=root+suffix or args[index].annotation or _encode(call.args[index],names)!=expected:
                raise ValueError('tensor argument order/shape differs: '+name)
            typ=('*i32' if role=='length' else pointer_type) if suffix=='_pointer' else 'i64'
            signature[name]=typ; dynamic.append(dict(name=name,type=typ,expr=expected)); index+=1
    if index!=46 or args[index].annotation or _encode(call.args[index],names)!=['input','scale']:
        raise ValueError('expected final dynamic fp64 scale')
    signature[arg_names[index]]='fp64'
    dynamic.append(dict(name=arg_names[index],type='fp64',expr=['input','scale']))
    return dict(kernel=kernel_name,dtype=dtype,causal=causal,arg_names=arg_names,
                signature=signature,constexprs=constants,dynamic=dynamic,grid=encoded_grid,
                profile='no-cache-no-mask-nine-0.26-attention-47args-v1')


def _value(expr,runtime):
    kind=expr[0]
    if kind=='input':return runtime[expr[1]]
    if kind=='int':return expr[1]
    if kind in ('size','stride'):return getattr(runtime[expr[1]],kind)(expr[2])
    op={x.__name__:fn for x,fn in OPS.items()}.get(kind)
    if not op:raise ValueError('unknown encoded launch operation')
    a,b=_value(expr[1],runtime),_value(expr[2],runtime)
    if type(a) is not int or type(b) is not int:raise ValueError('grid arithmetic must be integer')
    result=op(a,b)
    if abs(result)>=2**63:raise ValueError('launch arithmetic overflow')
    return result


def bind_launch(schema,arguments):
    if len(arguments)!=10:raise ValueError('expected ten runtime arguments')
    runtime=dict(zip(ROLES,arguments))
    if (runtime['causal'] is not schema['causal'] or runtime['masked'] is not False
            or type(runtime['variant']) is not int or runtime['variant']!=2):
        raise ValueError('runtime flags do not match exported specialization')
    grid=_value(schema['grid'],runtime)
    if type(grid) is not int or not 0<grid<2**31:raise ValueError('invalid grid')
    slots=[]
    for binding in schema['dynamic']:
        value=_value(binding['expr'],runtime); typ=binding['type']
        if typ.startswith('*'):
            value=value.data_ptr()
            if type(value) is not int or not 0<value<2**64:raise ValueError('invalid pointer')
        elif typ=='i64':
            if type(value) is not int or not 0<=value<2**63:raise ValueError('invalid dimension/stride')
        elif typ=='fp64':
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('positive finite scale required')
            value=struct.unpack('<Q',struct.pack('<d',value))[0]
        else:raise ValueError('unknown argument type')
        slots.append(value)
    return grid,slots


def validate_dynamic_lds(llir,shared):
    """Pinned HIP backend uses external LDS, allocated by the launch's sharedMemBytes."""
    if type(shared) is not int or not 0<=shared<=65536:
        raise ValueError('unsupported dynamic shared memory')
    if shared and len(re.findall(
            r'^@global_smem = external local_unnamed_addr addrspace\(3\) global \[0 x i8\], align 16\s*$',
            llir,re.M))!=1:
        raise ValueError('nonzero dynamic LDS requires pinned external global_smem declaration')
    return dict(profile='hip-external-dynamic-lds-v1',dynamic_shared_bytes=shared)


def validate_abi(assembly,metadata,schema):
    target=metadata.get('target',{})
    if (target.get('backend') not in ('hip','hcu') or target.get('arch')!='gfx936'
            or target.get('warp_size')!=64):raise ValueError('unsupported target ABI')
    if metadata.get('global_scratch_size',0)!=0:raise ValueError('nonzero global scratch unsupported')
    for key,want in dict(num_warps=4,num_ctas=1,profile_scratch_size=0,
                          launch_cooperative_grid=False,enable_fp_fusion=True).items():
        if metadata.get(key)!=want:raise ValueError('unsupported compiled metadata: '+key)
    shared=metadata.get('shared')
    if type(shared) is not int or not 0<=shared<=65536:raise ValueError('unsupported shared memory')
    blocks=re.findall(r'(?m)^\s*\.amdgpu_metadata\s*\n(.*?)^\s*\.end_amdgpu_metadata',assembly,re.S)
    if len(blocks)!=1:raise ValueError('expected one AMDGPU metadata block')
    block=blocks[0]
    def scalar(key):
        values=re.findall(r'(?m)^\s*'+re.escape(key)+r':\s*([^\n]+)',block)
        if len(values)!=1:raise ValueError('missing/ambiguous metadata: '+key)
        return values[0].strip().strip('"\'')
    kinds=['global_buffer' if item['type'].startswith('*') else 'by_value' for item in schema['dynamic']]
    kinds+=['global_buffer','global_buffer']
    for key,want in {'.kernarg_segment_size':str(len(kinds)*8),'.wavefront_size':'64',
                     '.max_flat_workgroup_size':'256','.group_segment_fixed_size':'0',
                     '.name':metadata.get('name'),'amdhsa.target':'amdgcn-amd-amdhsa--gfx936'}.items():
        actual=scalar(key)
        if actual!=want:raise ValueError(f'incompatible assembly metadata: {key}: actual={actual!r}, expected={want!r}')
    parts=re.split(r'(?m)^\s*- \.args:\s*\n',block)
    if len(parts)!=2:raise ValueError('missing/ambiguous kernel args')
    args=parts[1].split('    .group_segment_fixed_size:',1)[0]
    entries=re.split(r'(?m)^      - ',args)[1:]
    if len(entries)!=len(kinds):raise ValueError('unexpected argument count')
    for i,(entry,kind) in enumerate(zip(entries,kinds)):
        fields=dict(re.findall(r'(\.[a-z_]+):\s*([^\s]+)',entry))
        if (fields.get('.offset'),fields.get('.size'),fields.get('.value_kind'))!=(str(i*8),'8',kind):
            raise ValueError(f'incompatible kernel argument {i}: {fields}')
    return dict(argument_count=len(kinds),dynamic_count=len(kinds)-2,kernarg_bytes=len(kinds)*8,
                block_threads=256,shared_bytes=shared,static_shared_bytes=0,dynamic_shared_bytes=shared,
                shared_memory_profile='hip-external-dynamic-lds-v1',null_scratch_pointers=2)


def verify_export_files(folder,files):
    root=Path(folder).resolve()
    if not files:raise ValueError('empty export manifest')
    for name,expected in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=expected:
            raise ValueError('export missing/checksum/path mismatch: '+name)


def validate_aot_report(report):
    cases=report.get('cases',[]); expected=case_specs()
    if (report.get('result')!='AOT_PROBE_PASSED' or report.get('validation_scope')!=SCOPE
            or report.get('integration_status')!='NOT_TESTED' or report.get('performance_measurement_valid') is not False
            or report.get('target',{}).get('arch')!='gfx936' or report.get('fresh_process_launch') is not True
            or report.get('target',{}).get('backend') not in ('hip','hcu') or report.get('target',{}).get('warp_size')!=64
            or report.get('host_launch_count')!=24 or report.get('export_count')!=6
            or Counter(c.get('name') for c in cases)!=Counter(c['name'] for c in expected)):
        raise ValueError('incomplete AOT/host evidence')
    by_name={c['name']:c for c in expected}
    for case in cases:
        if (case.get('rounds')!=2 or case.get('host_calls')!=2
                or any(case.get(k)!=v for k,v in by_name[case['name']].items())
                or not all(case.get(key) is True for key in ('passed','finite','inputs_unchanged','guards_unchanged','explicit_stream_verified'))
                or len(case.get('metrics',[]))!=2):
            raise ValueError('incomplete case: '+str(case.get('name')))
        for round_number,metric in enumerate(case['metrics'],1):
            if metric.get('round')!=round_number:raise ValueError('incorrect metric round')
            for key in ('max_abs','relative_l2'):
                value=metric.get(key)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('invalid numerical metric: '+key)
