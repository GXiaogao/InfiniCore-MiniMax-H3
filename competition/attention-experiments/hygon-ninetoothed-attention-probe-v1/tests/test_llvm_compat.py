import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]


class LLVMCompatibilityTests(unittest.TestCase):
    def setUp(self):
        path=PACKAGE/'llvm_compat.py'
        self.assertTrue(path.is_file(),'LLVM compatibility lowering is not implemented')
        spec=importlib.util.spec_from_file_location('nine_llvm_compat_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_reported_gep_syntax_preserves_address_expression(self):
        original='  %67 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %66\n'
        lowered,counts=self.m.lower_gep_flags(original)
        self.assertEqual(lowered,'  %67 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i64 %66\n')
        self.assertEqual(counts,{'nuw':1,'nusw':0})

    def test_arithmetic_strings_comments_and_names_are_not_rewritten(self):
        original=('%nuw = add nuw i64 %x, 1\n'
                  '; getelementptr inbounds nuw i8, ptr %comment, i64 0\n'
                  '@label = constant [25 x i8] c"getelementptr nuw test\\00"\n'
                  '%p = getelementptr nusw i8, ptr %nuw, i64 1 ; keep nuw here\n')
        lowered,counts=self.m.lower_gep_flags(original)
        self.assertEqual(lowered,original.replace('getelementptr nusw i8','getelementptr i8'))
        self.assertEqual(counts,{'nuw':0,'nusw':1})

    def test_combined_flags_constant_expression_and_idempotence(self):
        original='%p = getelementptr nuw nusw inbounds i8, ptr %x, i64 2\n'
        lowered,counts=self.m.lower_gep_flags(original)
        self.assertEqual(lowered,'%p = getelementptr inbounds i8, ptr %x, i64 2\n')
        again,second=self.m.lower_gep_flags(lowered)
        self.assertEqual(again,lowered)
        self.assertEqual(second,{'nuw':0,'nusw':0})

    def test_evidence_retains_original_and_lowered_ir(self):
        with tempfile.TemporaryDirectory() as d:
            text='%p = getelementptr inbounds nuw i8, ptr %x, i64 2\n'
            evidence=Path(d)/'evidence'
            lowered=self.m.lower_for_vendor(text,evidence)
            reports=list(evidence.glob('*.json'))
            self.assertEqual(len(reports),1)
            report=json.loads(reports[0].read_text())
            self.assertEqual((evidence/report['original_file']).read_text(),text)
            self.assertEqual((evidence/report['lowered_file']).read_text(),lowered)
            self.assertEqual(report['original_sha256'],hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(report['removed_flags'],{'nuw':1,'nusw':0})

    def test_unknown_backend_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as d:
            vendor=Path(d)
            backend=vendor/'triton/backends/amd/compiler.py';backend.parent.mkdir(parents=True)
            backend.write_bytes(b'unrecognized backend')
            with self.assertRaises(ValueError):self.m.patch_vendor_backend(vendor,PACKAGE/'llvm_compat.py',vendor/'evidence')
            self.assertEqual(backend.read_bytes(),b'unrecognized backend')
            self.assertFalse((backend.parent/'h3_llvm_compat.py').exists())

    def test_real_pinned_backend_only_changes_translation_boundary(self):
        wheel=Path(__import__('os').environ.get('H3_TEST_TRITON_WHEEL','__missing_vendor_wheel__'))
        if not wheel.is_file():self.skipTest('fixed vendor wheel unavailable')
        with tempfile.TemporaryDirectory() as d,zipfile.ZipFile(wheel) as archive:
            root=Path(d)
            backend=root/'triton/backends/amd/compiler.py';backend.parent.mkdir(parents=True)
            original=archive.read('triton/backends/amd/compiler.py');backend.write_bytes(original)
            result=self.m.patch_vendor_backend(root,PACKAGE/'llvm_compat.py',root/'evidence')
            after=backend.read_text(encoding='utf-8')
            self.assertEqual(after,original.decode().replace(self.m.ANCHOR,self.m.INSERT+self.m.ANCHOR,1))
            self.assertEqual(result['original_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual((backend.parent/'h3_llvm_compat.py').read_bytes(),(PACKAGE/'llvm_compat.py').read_bytes())


if __name__=='__main__':unittest.main()
