#!/usr/bin/env python3
"""Validate pinned InfiniLM runtime wheels for job-local staging."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.utils import canonicalize_name


INSTALL_VERSION = "1.6-infinilm-private-runtime-job-local-staging"

EXPECTED_WHEELS = {
    "accelerate-1.15.0-py3-none-any.whl": "97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec",
    "annotated_doc-0.0.5-py3-none-any.whl": "117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101",
    "av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl": "8a032e8d8ebc73dec079364b9b4a6837638a2d106e8472314e685ffbf163e700",
    "click-8.4.2-py3-none-any.whl": "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76",
    "decord-0.6.0-py3-none-manylinux2010_x86_64.whl": "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976",
    "diffusers-0.41.0-py3-none-any.whl": "ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990",
    "h11-0.16.0-py3-none-any.whl": "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86",
    "hf_xet-1.5.2-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": "db78c39c83d6279daddc98e2238f373ab8980685556d42472b4ec51abcf03e8c",
    "httpcore-1.0.9-py3-none-any.whl": "2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55",
    "httpx-0.28.1-py3-none-any.whl": "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad",
    "huggingface_hub-1.32.0-py3-none-any.whl": "b0c7c80561969d9cdacdd55fce67ba9584cca0b9d4ea80957a3a5c1445fac5c8",
    "imageio-2.34.2-py3-none-any.whl": "a0bb27ec9d5bab36a9f4835e51b21d2cb099e1f78451441f94687ff3404b79f8",
    "janus-2.0.0-py3-none-any.whl": "7e6449d34eab04cd016befbd7d8c0d8acaaaab67cb59e076a69149f9031745f9",
    "lazy_loader-0.4-py3-none-any.whl": "342aa8e14d543a154047afb4ba8ef17f5563baad3fc610d7b15b213b0f119efc",
    "markdown_it_py-4.2.0-py3-none-any.whl": "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a",
    "mdurl-0.1.2-py3-none-any.whl": "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8",
    "msgspec-0.22.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": "a6db3806b3b76ca78064255eac6fa101a8a64fe6f698d80fbaf81fdfa21217d4",
    "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "377d08a7e48a1405b5e84afcbe4798464ce7ee17081c1c23619c8b398ff18295",
    "pygments-2.21.0-py3-none-any.whl": "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9",
    "regex-2026.9.29-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": "612b709381c0355b70d89cdb51b7f670591ed5cbbc0e3b5337488019dc667b65",
    "rich-15.0.0-py3-none-any.whl": "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb",
    "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774",
    "scikit_image-0.26.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl": "74aa5518ccea28121f57a95374581d3b979839adc25bb03f289b1bc9b99c58af",
    "scipy-1.11.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "530f9ad26440e85766509dbf78edcfe13ffd0ab7fec2560ee5c36ff74d6269ff",
    "shellingham-1.5.4-py2.py3-none-any.whl": "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686",
    "tifffile-2024.8.30-py3-none-any.whl": "8bc59a8f02a2665cd50a910ec64961c5373bee0b8850ec89d3b7b485bf7be7ad",
    "tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87",
    "transformers-5.20.2-py3-none-any.whl": "7ae75dbc1fb3922aff1b6c97b9e5479d6cf26fd86a8866bfb71324190479f807",
    "typer-0.27.3-py3-none-any.whl": "e50022f28b82a86313e54501317a1db64bf8f8d036ff8cfe5ca7e47675454aff",
    "xxhash-4.0.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": "f8e86a0b5e28418162dfa68d6dc4bee873ea2efbd6ee67789903a0ee37a81c14",
}

EXPECTED_DISTRIBUTIONS = {
    "accelerate": "1.15.0",
    "annotated-doc": "0.0.5",
    "av": "18.1.0",
    "click": "8.4.2",
    "decord": "0.6.0",
    "diffusers": "0.41.0",
    "h11": "0.16.0",
    "hf-xet": "1.5.2",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.32.0",
    "imageio": "2.34.2",
    "janus": "2.0.0",
    "lazy-loader": "0.4",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "msgspec": "0.22.0",
    "opencv-python-headless": "4.10.0.84",
    "pygments": "2.21.0",
    "regex": "2026.9.29",
    "rich": "15.0.0",
    "safetensors": "0.8.0",
    "scikit-image": "0.26.0",
    "scipy": "1.11.4",
    "shellingham": "1.5.4",
    "tifffile": "2024.8.30",
    "tokenizers": "0.23.3",
    "transformers": "5.20.2",
    "typer": "0.27.3",
    "xxhash": "4.0.2",
}

PROTECTED_STACK = {
    "torch",
    "numpy",
}

VALIDATION_CODE = r'''import importlib.metadata as metadata
import json

expected = {
    "accelerate": "1.15.0",
    "annotated-doc": "0.0.5",
    "av": "18.1.0",
    "click": "8.4.2",
    "decord": "0.6.0",
    "diffusers": "0.41.0",
    "h11": "0.16.0",
    "hf-xet": "1.5.2",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.32.0",
    "imageio": "2.34.2",
    "janus": "2.0.0",
    "lazy-loader": "0.4",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "msgspec": "0.22.0",
    "opencv-python-headless": "4.10.0.84",
    "pygments": "2.21.0",
    "regex": "2026.9.29",
    "rich": "15.0.0",
    "safetensors": "0.8.0",
    "scikit-image": "0.26.0",
    "scipy": "1.11.4",
    "shellingham": "1.5.4",
    "tifffile": "2024.8.30",
    "tokenizers": "0.23.3",
    "transformers": "5.20.2",
    "typer": "0.27.3",
    "xxhash": "4.0.2",
}
versions = {name: metadata.version(name) for name in expected}
assert versions == expected, (versions, expected)

import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()

import infinicore
from infinicore.lib import _infinicore
assert _infinicore.get_device_count(_infinicore.Device.Type.HYGON) == 8

import janus
import lazy_loader
import annotated_doc
import av
import click
import decord
import cv2
import h11
import hf_xet
import httpcore
import huggingface_hub
import imageio
import markdown_it
import mdurl
import pygments
import regex
import rich
import safetensors
import shellingham
import skimage
import scipy
import tifffile
import tokenizers
import transformers
import typer
import msgspec
import xxhash
import httpx

import infinilm
from infinilm.lib import _infinilm
assert hasattr(_infinilm, "InferEngine")
assert hasattr(_infinilm, "DistConfig")
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder, MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner

from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration
from safetensors.torch import load_file

print("PRIVATE_RUNTIME_JSON=" + json.dumps({
    "versions": versions,
    "torch": torch.__version__,
    "torch_device_count": torch.cuda.device_count(),
    "infinicore": str(infinicore.__file__),
    "infinilm": str(infinilm.__file__),
}, sort_keys=True))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument("--local-target", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheels(wheel_dir: Path) -> list[Path]:
    paths = []
    for filename, expected in EXPECTED_WHEELS.items():
        path = wheel_dir / filename
        if not path.is_file():
            raise RuntimeError(f"required wheel is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"wheel checksum mismatch: {filename}; expected={expected}; actual={actual}")
        paths.append(path)
        print(f"wheel_checksum=OK sha256={actual} file={filename}")
    return paths


def target_distributions(target: Path) -> dict[str, str]:
    result = {}
    for distribution in importlib.metadata.distributions(path=[str(target)]):
        name = canonicalize_name(distribution.metadata["Name"])
        result[name] = distribution.version
    return result


def wheel_destination(member_name: str, local_target: Path) -> Path:
    member = PurePosixPath(member_name)
    parts = member.parts
    if not parts or member.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"unsafe wheel member path: {member_name!r}")
    if parts[0].endswith(".data"):
        if len(parts) < 3:
            raise RuntimeError(f"invalid wheel data member: {member_name!r}")
        scheme = parts[1]
        remainder = parts[2:]
        prefixes = {
            "purelib": (),
            "platlib": (),
            "scripts": ("bin",),
            "headers": ("include",),
            "data": ("data",),
        }
        if scheme not in prefixes:
            raise RuntimeError(f"unsupported wheel data scheme {scheme!r}: {member_name!r}")
        parts = (*prefixes[scheme], *remainder)
    destination = (local_target / Path(*parts)).resolve()
    target_root = local_target.resolve()
    if destination != target_root and target_root not in destination.parents:
        raise RuntimeError(f"wheel member escapes target: {member_name!r}")
    return destination


def safe_extract_wheel(wheel: Path, local_target: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            destination = wheel_destination(member.filename, local_target)
            unix_mode = (member.external_attr >> 16) & 0o177777
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"wheel symlinks are not accepted: {wheel.name}:{member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"wheel file collision: {wheel.name}:{member.filename} -> {destination}")
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            file_mode = unix_mode & 0o777
            destination.chmod(file_mode or 0o644)
            extracted += 1
    return extracted


def install_wheels_direct(wheels: list[Path], local_target: Path) -> None:
    if local_target.exists() and any(local_target.iterdir()):
        raise RuntimeError(f"local target is not empty: {local_target}")
    local_target.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        extracted = safe_extract_wheel(wheel, local_target)
        print(f"wheel_unpack=OK files={extracted} file={wheel.name}")


def check_target_contents(target: Path) -> dict[str, str]:
    distributions = target_distributions(target)
    protected_stack_contamination = sorted(PROTECTED_STACK & set(distributions))
    if protected_stack_contamination:
        raise RuntimeError(f"protected_stack_contamination={protected_stack_contamination}")
    if distributions != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(f"unexpected private distributions: actual={distributions}; expected={EXPECTED_DISTRIBUTIONS}")
    return distributions


def validate_runtime(target: Path, run_dir: Path, label: str) -> dict[str, Any]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(target) + (os.pathsep + existing if existing else "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", VALIDATION_CODE],
        text=True,
        capture_output=True,
        env=environment,
        timeout=240,
    )
    (run_dir / f"validate-{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / f"validate-{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.rstrip().splitlines()[-15:])
        raise RuntimeError(f"runtime validation failed at {label}; exit_code={completed.returncode}; error={tail}")
    return {"label": label, "returncode": completed.returncode, "stdout": completed.stdout.rstrip()}


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    (run_dir / "runtime-install-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


STAGE_STARTED_AT: dict[str, float] = {}


def stage_start(result: dict[str, Any], stage: str) -> None:
    result["last_stage"] = stage
    STAGE_STARTED_AT[stage] = time.monotonic()
    print(
        f"stage={stage} status=START time={datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )


def stage_done(result: dict[str, Any], stage: str) -> None:
    started_at = STAGE_STARTED_AT.pop(stage, time.monotonic())
    elapsed_seconds = time.monotonic() - started_at
    print(f"stage={stage} status=DONE elapsed_seconds={elapsed_seconds:.3f}", flush=True)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "install_version": INSTALL_VERSION,
        "python": sys.version,
        "python_executable": sys.executable,
        "result": "FAILED",
        "last_stage": "initialization",
    }
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(f"Python 3.11 is required, got {sys.version}")

        stage_start(result, "wheel-verification")
        wheels = verify_wheels(args.wheel_dir.resolve())
        stage_done(result, "wheel-verification")

        stage_start(result, "local-wheel-unpack")
        local_target = args.local_target.resolve()
        install_wheels_direct(wheels, local_target)
        result["private_distributions"] = check_target_contents(local_target)
        stage_done(result, "local-wheel-unpack")

        stage_start(result, "local-runtime-validation")
        result["local_validation"] = validate_runtime(local_target, run_dir, "local")
        stage_done(result, "local-runtime-validation")

        wheel_dir = args.wheel_dir.resolve()
        result["runtime_mode"] = "job-local-wheel-staging"
        result["wheel_dir"] = str(wheel_dir)

        stage_start(result, "state-manifest")
        args.state_dir.mkdir(parents=True, exist_ok=True)
        state_dir = args.state_dir.resolve()
        manifest = state_dir / "infinilm-runtime-private.json"
        manifest_temp = manifest.with_name(f"{manifest.name}.tmp.{os.getpid()}")
        manifest_payload = {
            "schema_version": 1,
            "install_version": INSTALL_VERSION,
            "runtime_mode": "job-local-wheel-staging",
            "wheel_dir": str(wheel_dir),
            "wheel_count": len(EXPECTED_WHEELS),
            "expected_wheels": EXPECTED_WHEELS,
            "expected_distributions": EXPECTED_DISTRIBUTIONS,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "validation_exit_code": result["local_validation"]["returncode"],
        }
        manifest_temp.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_temp.chmod(0o644)
        os.replace(manifest_temp, manifest)
        result["state_manifest"] = str(manifest)
        stage_done(result, "state-manifest")

        stage_start(result, "state-marker")
        marker = state_dir / "infinilm-runtime-private.path"
        marker_temp = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
        marker_temp.write_text(str(wheel_dir) + "\n", encoding="utf-8")
        marker_temp.chmod(0o644)
        os.replace(marker_temp, marker)
        stage_done(result, "state-marker")

        result["result"] = "PASSED"
        result["last_stage"] = "complete"
        result["state_marker"] = str(marker)
        write_result(run_dir, result)
        print("private_runtime_mode=job-local-wheel-staging")
        print(f"wheel_bundle={wheel_dir}")
        print(f"state_manifest={manifest}")
        print(f"state_marker={marker}")
        print("RESULT: PASSED; exit_code=0; last_stage=complete")
        return 0
    except Exception as error:
        result["error"] = str(error)
        write_result(run_dir, result)
        print(f"ERROR: stage {result['last_stage']} failed: {error}", file=sys.stderr)
        print(f"RESULT: FAILED; exit_code=1; last_stage={result['last_stage']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
