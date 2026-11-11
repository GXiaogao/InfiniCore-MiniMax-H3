import copy
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def fixture():
    roles = ['q','k','v','length','mask','causal','scale','out','masked','variant']
    flags = ['c_causal','c_masked','c_variant']
    names = flags[:]
    exprs = flags[:]
    for name in ['q','k','v','mask','out','length']:
        sizes = 1 if name == 'length' else 3
        strides = 1 if name == 'length' else 4
        names += [name+'_pointer'] + [name+f'_size_{i}' for i in range(sizes)] + [name+f'_stride_{i}' for i in range(strides)]
        exprs += [name] + [name+f'.size({i})' for i in range(sizes)] + [name+f'.stride({i})' for i in range(strides)]
    names += ['scale']; exprs += ['scale']
    args = [x+': triton.language.constexpr' if x in flags else x for x in names]
    text = f'def attention({", ".join(args)}):\n    pass\n'
    text += f'def launch_attention({", ".join(roles)}):\n'
    for flag, role in zip(flags, ['causal','masked','variant']):
        text += f'    {flag} = {role}\n'
    text += f'    attention[lambda meta: (q.size(0)*k.size(1)*((q.size(2)+31)//32),)]({", ".join(exprs)}, num_warps=4, num_stages=1)\n'
    return text, names


def metadata():
    return dict(name='attention',target=dict(backend='hip',arch='gfx936',warp_size=64),
                num_warps=4,num_ctas=1,shared=16384,global_scratch_size=0,
                profile_scratch_size=0,launch_cooperative_grid=False,enable_fp_fusion=True)


def assembly(kinds):
    args = ''.join(f'      - .offset: {i*8}\n        .size: 8\n        .value_kind: {kind}\n' for i,kind in enumerate(kinds+['global_buffer','global_buffer']))
    return '.amdgpu_metadata\n---\namdhsa.target: amdgcn-amd-amdhsa--gfx936\namdhsa.kernels:\n  - .args:\n'+args+f'    .group_segment_fixed_size: 0\n    .kernarg_segment_size: {(len(kinds)+2)*8}\n    .wavefront_size: 64\n    .max_flat_workgroup_size: 256\n    .name: attention\n...\n.end_amdgpu_metadata\n'


class FakeTensor:
    def __init__(self,shape,strides,ptr): self.shape,self.strides,self.ptr=shape,strides,ptr
    def size(self,i): return self.shape[i]
    def stride(self,i): return self.strides[i]
    def data_ptr(self): return self.ptr


class ContractTests(unittest.TestCase):
    def test_reported_gfx936_dynamic_lds_artifact(self):
        import json
        api=self.api()
        folder=HERE/'results/20261111T004840Z-336595/exports/nine_attention_float16_d64_c0'
        schema=json.loads((folder/'launch-schema.json').read_text())
        meta=json.loads((folder/'metadata.json').read_text())
        result=api.validate_abi((folder/'kernel.amdgcn').read_text(),meta,schema)
        self.assertEqual(result['static_shared_bytes'],0)
        self.assertEqual(result['dynamic_shared_bytes'],4096)
        self.assertEqual(result['shared_bytes'],4096)
        self.assertEqual(result['argument_count'],46)
        api.validate_dynamic_lds((folder/'kernel.llir').read_text(),result['dynamic_shared_bytes'])

    def test_dynamic_lds_requires_real_external_declaration(self):
        api=self.api()
        self.assertTrue(hasattr(api,'validate_dynamic_lds'),'dynamic LDS evidence check missing')
        llir='@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8], align 16\n'
        api.validate_dynamic_lds(llir,4096)
        for bad in ['',llir.replace('external ','',1),llir.replace('[0 x i8]','[4096 x i8]'),llir.replace('addrspace(3)','addrspace(1)')]:
            with self.assertRaises(ValueError):api.validate_dynamic_lds(bad,4096)

    def api(self):
        spec=importlib.util.find_spec('aot_contract')
        self.assertIsNotNone(spec, 'AOT launch contract has not been implemented')
        import aot_contract
        return aot_contract

    def test_generated_binding_order_types_and_grid(self):
        api=self.api(); text,names=fixture()
        schema=api.launch_schema(text,'attention','float16',False)
        self.assertEqual(schema['arg_names'],names)
        self.assertEqual(schema['constexprs'],dict(c_causal=False,c_masked=False,c_variant=2))
        self.assertEqual(len(schema['dynamic']),44)
        self.assertEqual(schema['signature']['length_pointer'],'*i32')
        self.assertEqual(schema['signature']['scale'],'fp64')
        q=FakeTensor((2,4,33,64),(33*4*64,64,4*64,1),0x1000)
        k=FakeTensor((2,2,65,64),(65*2*64,64,2*64,1),0x2000)
        runtime=[q,k,k,FakeTensor((1,),(1,),0x3000),q,False,1/math.sqrt(64),q,False,2]
        grid,slots=api.bind_launch(schema,runtime)
        self.assertEqual(grid,8)
        self.assertEqual(slots[0],0x1000)
        self.assertEqual(slots[-1],struct.unpack('<Q',struct.pack('<d',.125))[0])
        runtime[5]=True
        with self.assertRaises(ValueError): api.bind_launch(schema,runtime)

    def test_actual_ninetoothed_generated_launch(self):
        api=self.api()
        source=HERE.parent/'hygon-ninetoothed-attention-probe-v1/results/20261110T025453Z-839257/generated/attention_float16_d64_causal0.py'
        schema=api.launch_schema(source.read_text(), 'attention_float16_d64_causal0','bfloat16',False)
        self.assertEqual(len(schema['dynamic']),44)
        self.assertEqual(sum(x['type']=='*bf16' for x in schema['dynamic']),5)

    def test_unknown_or_tampered_launch_rejected(self):
        api=self.api(); text,_=fixture()
        for bad in [text.replace('q.size(0)','__import__("os").system("oops")'),
                    text.replace('num_warps=4','num_warps=8'),
                    text.replace('c_masked = masked','c_masked = causal'),
                    text.replace('q_pointer, q_size_0','q_pointer, q_stride_0'),
                    text.replace('q, q.size(0)','k, q.size(0)')]:
            with self.subTest(bad=bad[-80:]):
                with self.assertRaises(ValueError): api.launch_schema(bad,'attention','float16',False)

    def test_compiled_abi_and_rejection(self):
        api=self.api(); text,_=fixture()
        schema=api.launch_schema(text,'attention','float16',False)
        kinds=['global_buffer' if x['type'].startswith('*') else 'by_value' for x in schema['dynamic']]
        asm=assembly(kinds); meta=metadata()
        result=api.validate_abi(asm,meta,schema)
        self.assertEqual(result['argument_count'],46)
        self.assertEqual(result['kernarg_bytes'],368)
        self.assertEqual(result['shared_bytes'],16384)
        # This pinned vendor3.5 driver sets global_scratch=0 without emitting a metadata field.
        legacy=copy.deepcopy(meta); del legacy['global_scratch_size']
        self.assertEqual(api.validate_abi(asm,legacy,schema)['null_scratch_pointers'],2)
        for change in [('global_scratch_size',8),('shared',65537),('num_warps',8),('target',dict(backend='hip',arch='gfx906',warp_size=64))]:
            bad=copy.deepcopy(meta); bad[change[0]]=change[1]
            with self.assertRaises(ValueError): api.validate_abi(asm,bad,schema)
        for bad in [asm.replace('.offset: 8\n','.offset: 4\n',1),asm.replace('.kernarg_segment_size: 368','.kernarg_segment_size: 360'),asm.replace('.wavefront_size: 64','.wavefront_size: 32'),asm.replace('.group_segment_fixed_size: 0','.group_segment_fixed_size: 16384')]:
            with self.assertRaises(ValueError): api.validate_abi(bad,meta,schema)

    def test_invalid_scalar_and_missing_cases_are_not_success(self):
        api=self.api(); text,_=fixture()
        schema=api.launch_schema(text,'attention','float16',False)
        t=FakeTensor((1,1,1,64),(64,64,64,1),0x1000)
        args=[t,t,t,FakeTensor((1,),(1,),0x2000),t,False,float('nan'),t,False,2]
        with self.assertRaises(ValueError): api.bind_launch(schema,args)
        with self.assertRaises(ValueError): api.validate_aot_report(dict(result='AOT_PROBE_PASSED',cases=[]))

    def test_export_artifact_integrity_and_traversal(self):
        api=self.api()
        self.assertTrue(hasattr(api,'verify_export_files'),'export integrity checker missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); file=root/'kernel.hsaco'; file.write_bytes(b'ELF-test')
            from attention_contract import sha256
            files={'kernel.hsaco':sha256(file)}
            api.verify_export_files(root,files)
            file.write_bytes(b'changed')
            with self.assertRaises(ValueError):api.verify_export_files(root,files)
            with self.assertRaises(ValueError):api.verify_export_files(root,{'../outside':'0'*64})

    def test_native_launcher_contract_exists_and_rejects_wrong_shape(self):
        self.assertIsNotNone(importlib.util.find_spec('aot_native'),'host launch module missing')
        from aot_native import validate_arguments
        class Device:
            type='cuda';index=0
        class Tensor(FakeTensor):
            device=Device(); dtype='torch.float16'
            @property
            def ndim(self):return len(self.shape)
            def stride(self,i=None):return self.strides if i is None else self.strides[i]
        text,_=fixture(); schema=self.api().launch_schema(text,'attention','float16',False)
        schema['head_dim']=64
        q=Tensor((1,2,17,64),(2176,64,128,1),0x10000)
        k=Tensor((1,1,33,64),(2112,64,64,1),0x20000)
        v=Tensor(k.shape,k.strides,0x30000)
        out=Tensor(q.shape,q.strides,0x40000)
        length=Tensor((1,),(1,),0x50000); length.dtype='torch.int32'
        mask=Tensor((1,1,1,1),(1,1,1,1),0x60000)
        args=[q,k,v,length,mask,False,.125,out,False,2]
        validate_arguments(schema,args,0)
        out.shape=(1,2,16,64)
        with self.assertRaises(ValueError): validate_arguments(schema,args,0)
        out.shape=q.shape;out.ptr=q.ptr
        with self.assertRaises(ValueError): validate_arguments(schema,args,0)

    def test_complete_report_validated_and_inconsistent_claims_rejected(self):
        api=self.api()
        from attention_contract import case_specs
        report=dict(result='AOT_PROBE_PASSED',validation_scope=api.SCOPE,
            integration_status='NOT_TESTED',performance_measurement_valid=False,
            target=dict(backend='hip',arch='gfx936',warp_size=64),fresh_process_launch=True,
            host_launch_count=24,export_count=6,cases=[])
        for case in case_specs():
            report['cases'].append(dict(**case,rounds=2,host_calls=2,passed=True,finite=True,
                inputs_unchanged=True,guards_unchanged=True,explicit_stream_verified=True,
                metrics=[dict(round=i,max_abs=0.001,relative_l2=.002) for i in (1,2)]))
        api.validate_aot_report(report)
        for field,badvalue in [('dtype','float32'),('sq',100),('host_calls',0),('guards_unchanged',False)]:
            bad=copy.deepcopy(report);bad['cases'][0][field]=badvalue
            with self.assertRaises(ValueError):api.validate_aot_report(bad)
        bad=copy.deepcopy(report);bad['cases'][0]['metrics'][0]['max_abs']=float('nan')
        with self.assertRaises(ValueError):api.validate_aot_report(bad)


if __name__=='__main__':unittest.main()
