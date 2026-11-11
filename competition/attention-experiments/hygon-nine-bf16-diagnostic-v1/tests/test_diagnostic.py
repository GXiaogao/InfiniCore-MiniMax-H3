import importlib.util
from pathlib import Path
import sys
import unittest
import json
import os
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TEST_BASH'),'Git Bash required')
    def test_staging_and_readonly_work_preserve_failure(self):
        from package_diagnostic import NAMES
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);project=root/'project';project.mkdir()
            for name in NAMES:shutil.copyfile(HERE/name,project/name)
            (project/'package-manifest.json').write_text('{}')
            home=root/'afs/minimax-h3';home.mkdir(parents=True)
            (home/'setup_infinifabric_env.sh').write_text('readonly WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"\n')
            binaries=root/'bin';binaries.mkdir()
            (binaries/'python3').write_text('#!/bin/sh\necho expected-worker-failure\nexit 39\n',newline='\n')
            env=dict(os.environ,IFAB_AFS_HOME=str(home.parent))
            code='export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"\nexport PATH="$(cygpath -u "$1"):/usr/bin:/bin:$PATH"\nbash "$2"'
            if os.name!='nt':code='export PATH="$1:/usr/bin:/bin:$PATH"\nbash "$2"'
            result=subprocess.run([os.environ['TEST_BASH'],'-c',code,'test',str(binaries),str(project/'run_hygon_bf16_diagnostic.sh')],env=env,cwd=root,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,39,result.stdout+result.stderr)
            logs=list(home.glob('logs/*/summary.txt'));self.assertEqual(len(logs),1)
            self.assertIn('exit_code=39',logs[0].read_text())
            staged=list(home.glob('scripts/nine-bf16-diagnostic-*'));self.assertEqual(len(staged),1)
            self.assertEqual({f.name for f in staged[0].iterdir()},set(NAMES)|{'package-manifest.json'})

    def test_nonfinite_preview_remains_serializable(self):
        from bf16_diagnostic_contract import json_preview
        preview=json_preview([1.,float('nan'),float('inf'),-float('inf')])
        self.assertEqual(preview,[1.,'nan','inf','-inf'])
        json.dumps(preview,allow_nan=False)

    def test_mismatches_collected_without_claiming_correctness(self):
        self.assertIsNotNone(importlib.util.find_spec('bf16_diagnostic_contract'))
        from bf16_diagnostic_contract import validate_diagnostic
        report=dict(result='DIAGNOSTIC_COMPLETE',correctness_passed=False,
                    conversion=[dict(name=n,passed=False) for n in ('fp64-direct','fp64-via-fp32','fp32-direct')],
                    attention=[dict(round=i,aot_passed=False,jit_passed=True,inputs_unchanged=True,guards_unchanged=True) for i in (1,2)],target=dict(arch='gfx936'))
        validate_diagnostic(report)
        report['correctness_passed']=True
        with self.assertRaises(ValueError):validate_diagnostic(report)
        report['correctness_passed']=False;report['attention'].pop()
        with self.assertRaises(ValueError):validate_diagnostic(report)


if __name__=='__main__':unittest.main()
