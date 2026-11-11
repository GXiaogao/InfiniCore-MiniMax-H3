"""Offline integrity and fail-closed process helpers; no GPU imports."""
import ast
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import zipfile

CANDIDATE = 'ninetoothed026-triton380-gfx906'
MANIFEST = 'h3_toolchain_v2_manifest.json'
EXPECTED = {'triton': '3.8.0', 'ninetoothed': '0.26.0',
            'numpy': '1.26.4', 'sympy': '1.14.0', 'mpmath': '1.3.0'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')
    tmp.replace(path)


def filename_only(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', name):
        raise ValueError(f'Expected wheel basename: {name!r}')
    return name


def validate_manifest(value):
    if value.get('schema') != 2 or value.get('candidate') != CANDIDATE:
        raise ValueError('Wrong candidate manifest')
    entries = value.get('wheels', [])
    if len(entries) != len(EXPECTED) or {e['name'] for e in entries} != set(EXPECTED):
        raise ValueError('Manifest must list each of the five pinned packages exactly once')
    names = set()
    for entry in entries:
        name = filename_only(entry['file'])
        if name in names or entry['version'] != EXPECTED[entry['name']]:
            raise ValueError('Duplicate filename or wrong pinned version')
        names.add(name)
        digest = entry['sha256']
        if not re.fullmatch('[0-9a-f]{64}', digest) or entry.get('publisher_sha256') != digest:
            raise ValueError('Local hash must match published PyPI SHA256')
    return entries


def check_wheel(root, entry):
    path = Path(root) / filename_only(entry['file'])
    if not path.is_file() or sha256(path) != entry['sha256']:
        raise ValueError(f'Missing or corrupt wheel: {path}')
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError(f'Invalid wheel metadata: {path}')
        info = Parser().parsestr(archive.read(names[0]).decode('utf-8'))
    if info['Name'].lower().replace('_', '-') != entry['name'].replace('_', '-') or info['Version'] != entry['version']:
        raise ValueError(f'Wheel metadata does not match pin: {path}')
    return path


def stage_passed(rc, value, stage, run_id, count):
    if rc != 0 or not isinstance(value, dict):
        return False
    cases = value.get('cases')
    return (value.get('passed') is True and value.get('stage') == stage
            and value.get('run_id') == run_id and isinstance(cases, list)
            and len(cases) == count
            and all(isinstance(row, dict) and row.get('passed') is True for row in cases))


def run_child(command, output, timeout, env=None):
    with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                          env=env, start_new_session=os.name == 'posix') as child:
        try:
            return child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == 'posix':
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                child.kill()
            child.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return 124


def disable_fusion_source(source, kernel_name):
    """Only add a compiler launch option to NineToothed-generated Python.

    NineToothed 0.26 make() does not expose this option. Do not alter the
    generated compute body or invent its argument list. Fail on schema drift.
    """
    tree = ast.parse(source)
    found = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == kernel_name):
            continue
        if any(k.arg == 'enable_fp_fusion' for k in node.keywords):
            raise ValueError('Unexpected existing fusion option')
        node.keywords.append(ast.keyword(arg='enable_fp_fusion', value=ast.Constant(False)))
        found += 1
    if found != 1:
        raise ValueError(f'Expected one generated launch of {kernel_name}, found {found}')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
