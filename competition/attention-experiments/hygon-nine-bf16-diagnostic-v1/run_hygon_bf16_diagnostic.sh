#!/usr/bin/env bash
set -Eeuo pipefail
umask 0022
if [[ $# -ne 0 ]]; then echo 'ERROR: no program arguments accepted' >&2; exit 2; fi
: "${IFAB_AFS_HOME:?IFAB_AFS_HOME is missing}"
TASK_WORK="${IFAB_AFS_HOME}/minimax-h3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${TASK_WORK}/logs/infinifabric-nine-bf16-diagnostic-${TASK_ID}"
LOCAL_ROOT="${PWD}/.nine-bf16-diagnostic-${TASK_ID}"
mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
run_task() {
    printf 'version=1.1-bf16-cast-loaded-value-diagnostic\nrun_dir=%s\nscope=bf16-diagnostic-not-formal-correctness\n' "${RUN_DIR}"
    source "${TASK_WORK}/setup_infinifabric_env.sh"
    TASK_SCRIPT_DIR="${TASK_WORK}/scripts/nine-bf16-diagnostic-${TASK_ID}"
    mkdir -p "${TASK_SCRIPT_DIR}"
    for name in attention_contract.py attention_worker.py probe_primitives.py run_attention_probe.py llvm_compat.py upstream_attention.py candidate_attention.py install_ninetoothed_gfx936.py install_infinilm_private_runtime.py aot_contract.py aot_launcher.cpp aot_native.py bf16_diagnostic_contract.py bf16_cast_dsl.py bf16_diagnostic_worker.py run_bf16_diagnostic.py run_hygon_bf16_diagnostic.sh LICENSE-InfiniCore-MIT.txt UPLOAD.md package-manifest.json; do
        cp -- "${SCRIPT_DIR}/${name}" "${TASK_SCRIPT_DIR}/${name}"
    done
    printf 'staged_scripts=%s\n' "${TASK_SCRIPT_DIR}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    TASK_PYTHON="$(command -v python3)"
    timeout --signal=TERM --kill-after=15s 25m "${TASK_PYTHON}" -u "${TASK_SCRIPT_DIR}/run_bf16_diagnostic.py" \
        --work-home "${TASK_WORK}" --run-dir "${RUN_DIR}" --local-root "${LOCAL_ROOT}"
}
set +e
( set -Eeuo pipefail; run_task ) 2>&1 | tee "${RUN_DIR}/run.log"
statuses=("${PIPESTATUS[@]}")
set -e
rc="${statuses[0]}"
if [[ "${rc}" -eq 0 ]]; then rc="${statuses[1]}"; fi
printf 'exit_code=%s\nrun_dir=%s\nvalidation_scope=standalone-nine-bf16-diagnostic-host\nintegration_status=NOT_TESTED\nperformance_measurement_valid=false\n' \
    "${rc}" "${RUN_DIR}" > "${RUN_DIR}/summary.txt"
printf 'SUMMARY: %s/summary.txt\n' "${RUN_DIR}"
exit "${rc}"
