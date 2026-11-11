import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
BASH=os.environ.get('TEST_BASH') or shutil.which('bash')


class DeliveryTests(unittest.TestCase):
    def test_controller_requires_complete_export_and_launch(self):
        self.assertIsNotNone(importlib.util.find_spec('run_attention_aot'),'AOT controller missing')
        from run_attention_aot import validate_exports
        for value in [{},{'result':'AOT_EXPORT_COMPLETE','exports':[],'gpu_attention_launches':0}]:
            with self.assertRaises(ValueError):validate_exports(value)

    def test_worker_deadline_kills_child(self):
        from run_attention_aot import run_child
        with self.assertRaises(subprocess.TimeoutExpired):
            run_child([sys.executable,'-c','import time;time.sleep(20)'],dict(os.environ),.1,'local-timeout-test')

    @unittest.skipUnless(BASH,'Git Bash needed')
    def test_extra_shell_args_rejected_before_any_writes(self):
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([BASH,str(HERE/'run_hygon_attention_aot.sh'),'--override'],
                                  cwd=d,text=True,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_flat_upload_archive_integrity(self):
        self.assertTrue((HERE/'package_aot.py').is_file(),'AOT packager missing')
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,str(HERE/'package_aot.py'),'--output-dir',directory],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            archive=Path(directory)/'hygon-ninetoothed-attention-aot-v1.1-upload.zip'
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                names=set(bundle.namelist())
                self.assertTrue(all('/' not in name for name in names))
                manifest=json.loads(bundle.read('package-manifest.json'))
                self.assertEqual(names,set(manifest['files'])|{'package-manifest.json'})
                from attention_contract import verify_manifest
                verify_manifest(HERE,manifest['files'])

    @unittest.skipUnless(BASH,'Git Bash needed')
    def test_setup_failure_is_not_swallowed_by_tee(self):
        entry=HERE/'run_hygon_attention_aot.sh'
        self.assertTrue(entry.is_file(),'AOT shell entry missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);home=root/'afs/minimax-h3';home.mkdir(parents=True)
            (home/'setup_infinifabric_env.sh').write_text('return 17\n',encoding='utf-8')
            env=dict(os.environ,IFAB_AFS_HOME=str(root/'afs'))
            code='export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"\nbash "$1"'
            if os.name!='nt':code='bash "$1"'
            result=subprocess.run([BASH,'-c',code,'test',str(entry)],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,17,result.stdout+result.stderr)
            summaries=list(home.glob('logs/*/summary.txt'))
            self.assertEqual(len(summaries),1)
            self.assertIn('exit_code=17',summaries[0].read_text())


if __name__=='__main__':unittest.main()
