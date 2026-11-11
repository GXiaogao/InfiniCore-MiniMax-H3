import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self):
        file = PACKAGE / 'attention_contract.py'
        self.assertTrue(file.is_file(), 'attention contract is not implemented')
        spec = importlib.util.spec_from_file_location('attention_contract_test', file)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def make_candidate(self,work):
        candidate=work/'python-envs/ninetoothed-gfx936/run'
        site=candidate/'site-packages'
        wheel=Path(__import__('os').environ.get('H3_TEST_NINETOOOTHED_WHEEL','__missing_nine_wheel__'))
        if not wheel.is_file():self.skipTest('set H3_TEST_NINETOOOTHED_WHEEL to the pinned Nine 0.26.0 wheel')
        with zipfile.ZipFile(wheel) as archive:archive.extractall(site)
        generation=site/'ninetoothed/generation.py'
        text=generation.read_text(encoding='utf-8').replace(
            'CACHE_DIR = pathlib.Path.home() / ".ninetoothed"',
            'CACHE_DIR = pathlib.Path(__import__("os").environ["NINETOOTHED_CACHE_DIR"])')
        generation.write_text(text,encoding='utf-8',newline='\n')
        patch={'original_sha256':'8863013c70fb5bdc131f1078c7b1e0ddbfdd68336a625dfb2811c781e345fe3d',
               'patched_sha256':'17f908e7ee7ddf9e87f2a2cb51d302d6d2b32211da7735998fbced276b8cbf4e'}
        wheels={'ninetoothed-0.26.0-py3-none-any.whl':[
            'ninetoothed','0.26.0','7f15a35b308f00659c63508ff14f964f431234397e5dc97f73e2e4008ab51f08']}
        (candidate/'candidate.json').write_text(json.dumps(dict(wheels=wheels,cache_patch=patch)))
        return candidate,patch

    def test_missing_state_fails_without_creating_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                self.m.read_verified_state(Path(d))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_state_requires_actual_jit_and_candidate_inside_work(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            candidate,patch = self.make_candidate(work)
            good = dict(candidate=str(candidate), probe=dict(result='JIT_VERIFIED', arch='gfx936'),
                        scope='standalone-add-jit-not-infinicore-integration',cache_patch=patch,
                        wheels=json.loads((candidate/'candidate.json').read_text())['wheels'])
            state = work/'state/ninetoothed-gfx936.json'
            state.parent.mkdir()
            state.write_text(json.dumps(good))
            self.assertEqual(self.m.read_verified_state(work)['candidate'], str(candidate.resolve()))
            for value in ({**good, 'candidate':str(work.parent)},
                          {**good, 'probe':dict(result='IMPORT_OK',arch='gfx936')},
                          {**good, 'probe':dict(result='JIT_VERIFIED',arch='gfx906')}):
                state.write_text(json.dumps(value))
                with self.assertRaises(ValueError):self.m.read_verified_state(work)

    def test_restored_cache_source_or_other_candidate_source_drift_is_rejected(self):
        self.assertTrue(hasattr(self.m,'verify_candidate_source'),'candidate source integrity not verified')
        with tempfile.TemporaryDirectory() as d:
            candidate,patch=self.make_candidate(Path(d))
            self.m.verify_candidate_source(candidate)
            generation=candidate/'site-packages/ninetoothed/generation.py'
            preserved=generation.read_bytes()
            generation.write_bytes(preserved.replace(
                b'pathlib.Path(__import__("os").environ["NINETOOTHED_CACHE_DIR"])',
                b'pathlib.Path.home() / ".ninetoothed"'))
            with self.assertRaises(ValueError):self.m.verify_candidate_source(candidate)
            generation.write_bytes(preserved)
            jit=candidate/'site-packages/ninetoothed/jit.py'
            jit.write_bytes(jit.read_bytes()+b'\n# unexpected drift\n')
            with self.assertRaises(ValueError):self.m.verify_candidate_source(candidate)

    def test_unknown_upstream_expression_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'flash_attention.py'
            p.write_text('unrecognized compute source')
            with self.assertRaises(ValueError):self.m.verify_upstream(p)

    def test_causal_mask_is_lower_right_and_gqa_shape_validated(self):
        self.assertEqual(self.m.causal_key_limit(0, 2, 4), 3)
        self.assertEqual(self.m.causal_key_limit(1, 2, 4), 4)
        valid = dict(batch=1, sq=17, sk=33, hq=4, hkv=2, dim=64, causal=True, layout='bsnd')
        self.m.validate_case(valid)
        for change in (dict(hq=3),dict(sq=34),dict(dim=7),dict(batch=0),dict(layout='unknown')):
            with self.assertRaises(ValueError):self.m.validate_case({**valid,**change})

    def test_partial_duplicate_failed_case_or_wrong_target_cannot_pass(self):
        names = [c['name'] for c in self.m.case_specs()]
        self.assertEqual(len(names),12)
        cases=[dict(name=n,passed=True,rounds=2,finite=True,inputs_unchanged=True,guards_unchanged=True) for n in names]
        good=dict(result='JIT_PROBE_PASSED',target=dict(backend='hip',arch='gfx936'),cases=cases,
                  primitives=dict(dot=True,reduction=True),aot_status='NOT_TESTED',integration_status='NOT_TESTED')
        self.m.validate_report(good)
        for bad in ({**good,'cases':cases[:-1]}, {**good,'cases':cases[:-1]+cases[:1]},
                    {**good,'cases':[{**cases[0],'guards_unchanged':False}]+cases[1:]},
                    {**good,'target':dict(backend='cuda',arch='gfx936')},
                    {**good,'primitives':dict(dot=True,reduction=False)},
                    {**good,'aot_status':'PASSED'}):
            with self.assertRaises(ValueError):self.m.validate_report(bad)

    def test_manifest_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'x.py').write_bytes(b'original')
            manifest={'x.py':self.m.sha256(folder/'x.py')}
            self.m.verify_manifest(folder,manifest)
            (folder/'x.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):self.m.verify_manifest(folder,manifest)


if __name__=='__main__':unittest.main()
