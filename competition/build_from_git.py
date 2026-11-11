#!/usr/bin/env python3
"""Build committed public sources in job-local storage; never modify the checked-out sources.

This adapter replaces archive extraction/source rewriting with committed Git trees.
Its host profiles follow the historical InfiniFabric builders. Device rebuild pending.
"""
import argparse
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import traceback
import zipfile


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('unexpected build source anchor: '+old)
    return text.replace(old,new,1)


def configure_text(text,component,profile):
    if (component,profile) not in [('InfiniCore','native'),('InfiniCore','python'),('InfiniLM','extension')]:
        raise ValueError('unknown component/profile')
    text=once(text,'add_requires("pybind11")\n','')
    if component=='InfiniCore':text=once(text,'add_requires("boost", {configs = {stacktrace = true}})\n','')
    if profile!='native':
        text=once(text,'\n    add_packages("pybind11")\n',
                  '\n    add_includedirs(os.getenv("PYBIND11_INCLUDE"), os.getenv("PYTHON_INCLUDE_DIR"), { public = false })\n')
    if profile=='python':text=once(text,'    add_deps("infinicore_cpp_api")','    add_links("infinicore_cpp_api")')
    return text


def command(args,cwd=None,env=None,capture=False):
    print('+ '+' '.join(map(str,args)),flush=True)
    result=subprocess.run(list(map(str,args)),cwd=cwd,env=env,check=True,
                          stdout=subprocess.PIPE if capture else None,text=capture)
    return result.stdout.strip() if capture else None


def export_tree(repo,destination,revision):
    data=subprocess.check_output(['git','-C',str(repo),'archive',revision])
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:archive.extractall(destination,filter='data')


def build(profile,log):
    source=Path(__file__).resolve().parents[1]
    component=json.loads((source/'competition/source-provenance.json').read_text(encoding='utf-8'))['component']
    configure_text((source/'xmake.lua').read_text(encoding='utf-8'),component,profile)  # Fail before side effects.
    work=Path(os.environ['IFAB_AFS_HOME']).resolve()/'minimax-h3'
    if Path.cwd().resolve().is_relative_to(work):raise ValueError('run in the job workspace, not AFS')
    revision=command(['git','-C',source,'rev-parse','HEAD'],capture=True)
    if command(['git','-C',source,'status','--porcelain','--untracked-files=no'],capture=True):
        raise ValueError('source checkout has tracked changes')
    prefix=work/'.infini-public-20261111'
    site=work/'python-envs/public-git-20261111/site-packages'
    local=Path.cwd()/('.public-source-build-'+profile+'-'+str(os.getpid()))
    project=local/component
    export_tree(source,project,revision)
    dependencies=['third_party/spdlog']+(['third_party/json'] if component=='InfiniLM' else [])
    for dependency in dependencies:
        expected=command(['git','-C',source,'rev-parse','HEAD:'+dependency],capture=True)
        actual=command(['git','-C',source/dependency,'rev-parse','HEAD'],capture=True)
        if actual!=expected:raise ValueError('submodule revision differs: '+dependency)
        export_tree(source/dependency,project/dependency,actual)
    env=dict(os.environ)
    env.update(INFINI_ROOT=str(prefix),PYTHON=sys.executable,XMAKE_GLOBALDIR=str(local/'global'),
               XMAKE_STORAGE_DIR=str(local/'global/.xmake'),XMAKE_TMPDIR=str(local/'tmp'),
               XMAKE_RAMDIR=str(local/'tmp'),TMPDIR=str(local/'tmp'),XMAKE_STATS='false',XMAKE_THEME='plain')
    for folder in [local/'global/.xmake',local/'tmp',prefix,site]:folder.mkdir(parents=True,exist_ok=True)
    packages=work/'packages'
    with tarfile.open(packages/'xmake-repo-cache.tar.gz') as archive:archive.extractall(local/'global',filter='data')
    xmake=project/'xmake.lua';xmake.write_text(configure_text(xmake.read_text(encoding='utf-8'),component,profile),encoding='utf-8',newline='\n')
    if component=='InfiniCore' and profile=='native':
        path=project/'xmake/hygon.lua'
        old='    add_files("../src/infiniop/devices/cpu/*.cc", "../src/infiniop/ops/*/cpu/*.cc", "../src/infiniop/reduce/cpu/*.cc")\n'
        path.write_text(once(path.read_text(encoding='utf-8'),old,''),encoding='utf-8',newline='\n')
    if profile!='native':
        with zipfile.ZipFile(packages/'pybind11-v3.1.0.zip') as archive:archive.extractall(local/'pybind')
        include=local/'pybind/pybind11-3.1.0/include'
        if not (include/'pybind11/pybind11.h').is_file():raise ValueError('offline pybind header missing')
        env.update(PYBIND11_INCLUDE=str(include),PYTHON_INCLUDE_DIR=sysconfig.get_paths()['include'])
    cuda=Path('/opt/dtk/cuda/cuda')
    env.update(CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),HYGON_CUDA_HOME=str(cuda))
    env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH']
    torch_lib=command([sys.executable,'-c','from pathlib import Path;import torch;print(Path(torch.__file__).resolve().parent/"lib")'],env=env,capture=True)
    env['LD_LIBRARY_PATH']=str(prefix/'lib')+':'+torch_lib+':'+env.get('LD_LIBRARY_PATH','')
    env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','')
    command(['xmake','g','--network=private','--pkg_searchdirs='+str(packages)],cwd=project,env=env)
    flags=['xmake','f','-y','-m','release']
    if component=='InfiniCore':
        flags+=['--hygon-dcu=y','--hygon-arch=gfx936','--cuda='+str(cuda),'--cpu=n','--omp=n',
                '--aten=y','--ccl=y','--graph=n','--infiniops=n','--ninetoothed=n','--flash-attn=']
    else:
        abi=command([sys.executable,'-c','import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))'],env=env,capture=True)
        flags+=['--cxx11-abi='+abi]
    command(flags,cwd=project,env=env)
    target='infinicore_cpp_api' if profile=='native' else ('_infinicore' if profile=='python' else '_infinilm')
    jobs=int(os.environ.get('PUBLIC_BUILD_JOBS','16'))
    if not 1<=jobs<=256:raise ValueError('PUBLIC_BUILD_JOBS outside1..256')
    command(['xmake','build','-j',str(jobs),target],cwd=project,env=env)
    if profile=='native':
        for target in ('infinirt','infiniop','infiniccl','infinicore_cpp_api'):
            command(['xmake','install',target],cwd=project,env=env)
    else:
        package='infinicore' if profile=='python' else 'infinilm'
        extension=list((project/'build/linux/x86_64/release').glob('_'+package+'*.so'))
        if len(extension)!=1:raise ValueError('built extension absent/ambiguous')
        destination=site/package
        shutil.copytree(project/'python'/package,destination,dirs_exist_ok=True)
        (destination/'lib').mkdir(exist_ok=True);saved=destination/'lib'/extension[0].name
        shutil.copyfile(extension[0],saved);saved.chmod(0o644)
        command(['ldd',saved],env=env)
        command([sys.executable,'-c','import sys;sys.path.insert(0,sys.argv[1]);__import__(sys.argv[2]);print("extension import OK")',
                 str(destination/'lib'),'_'+package],env=env)
    return dict(result='BUILD_AND_INSTALL_COMPLETE',component=component,profile=profile,source_commit=revision,
                install_prefix=str(prefix),python_site=str(site),local_project=str(project),log=str(log),
                scope='source build only; full model/quality/performance not validated')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('profile',choices=['native','python','extension']);args=parser.parse_args()
    work=Path(os.environ['IFAB_AFS_HOME'])/'minimax-h3'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    logs=Path(os.environ['PUBLIC_BUILD_RUN_DIR']) if os.environ.get('PUBLIC_BUILD_LOG_CHILD') else work/'logs'/('public-git-'+args.profile+'-'+stamp)
    if not logs.resolve().is_relative_to((work/'logs').resolve()):raise ValueError('log directory outside work/logs')
    logs.mkdir(parents=True,exist_ok=True)
    log=logs/'run.log'
    if not os.environ.get('PUBLIC_BUILD_LOG_CHILD'):
        env=dict(os.environ,PUBLIC_BUILD_LOG_CHILD='1',PUBLIC_BUILD_RUN_DIR=str(logs))
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),args.profile],env=env,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:print(line,end='',flush=True);stream.write(line);stream.flush()
            rc=child.wait()
        (logs/'summary.txt').write_text(f'exit_code={rc}\nlog={log}\n',encoding='utf-8')
        return rc
    try:
        result=build(args.profile,log)
        (logs/'build-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True);return 0
    except Exception:
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
