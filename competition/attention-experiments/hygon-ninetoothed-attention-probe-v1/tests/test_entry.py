import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE=Path(__file__).resolve().parents[1]
BASH=os.environ.get('TEST_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH,'Git Bash needed for shell behavior')
class EntryTests(unittest.TestCase):
    def test_extra_args_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([BASH,str(PACKAGE/'run_hygon_ninetoothed_attention.sh'),'--override'],
                                  cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_readonly_work_and_driver_failure_preserve_exit_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            project=folder/'project';project.mkdir()
            shutil.copyfile(PACKAGE/'run_hygon_ninetoothed_attention.sh',project/'entry.sh')
            manifest=json.loads((PACKAGE/'package-manifest.json').read_text())
            for name in list(manifest['files'])+['package-manifest.json']:
                shutil.copyfile(PACKAGE/name,project/name)
            work=folder/'afs/minimax-h3';work.mkdir(parents=True)
            (work/'setup_infinifabric_env.sh').write_text('readonly WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"\n')
            (project/'run_attention_probe.py').write_text('stub, not executed by Python')
            binaries=folder/'bin';binaries.mkdir()
            (binaries/'python3').write_text('#!/bin/sh\necho intentional-worker-failure\nexit 39\n',newline='\n')
            env=dict(os.environ,IFAB_AFS_HOME=work.parent.as_posix(),PATH=binaries.as_posix()+os.pathsep+os.environ['PATH'])
            code='export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"\nexport PATH="$(cygpath -u "$1"):/usr/bin:/bin:$PATH"\nbash "$2"'
            if os.name!='nt':code='export PATH="$1:/usr/bin:/bin:$PATH"\nbash "$2"'
            result=subprocess.run([BASH,'-c',code,'test',binaries.as_posix(),(project/'entry.sh').as_posix()],
                                  env=env,cwd=folder,capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,39,result.stdout+result.stderr)
            logs=list((work/'logs').rglob('run.log'))
            self.assertEqual(len(logs),1)
            self.assertIn('intentional-worker-failure',logs[0].read_text())
            self.assertIn('exit_code=39',(logs[0].parent/'summary.txt').read_text())
            staged=list((work/'scripts').glob('nine-attention-probe-*'))
            self.assertEqual(len(staged),1,'helpers must execute from persistent scripts directory')
            self.assertTrue((staged[0]/'package-manifest.json').is_file())


class ChildTimeoutTests(unittest.TestCase):
    def test_timeout_kills_child_and_does_not_return_success(self):
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('attention_controller_test',PACKAGE/'run_attention_probe.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with self.assertRaises(subprocess.TimeoutExpired):
                module.run_child([sys.executable,'-c','import time; time.sleep(30)'],dict(os.environ),0.1,'timeout-test')
        finally:sys.path.remove(str(PACKAGE))

    def test_outer_deadline_signal_is_reportable_instead_of_abrupt_exit(self):
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('attention_signal_test',PACKAGE/'run_attention_probe.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertTrue(hasattr(module,'JobInterrupted'),'outer timeout lacks cleanup/report handling')
            with self.assertRaises(module.JobInterrupted):module.handle_signal(15,None)
        finally:sys.path.remove(str(PACKAGE))
