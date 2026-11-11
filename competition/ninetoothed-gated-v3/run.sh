#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
export LOG_DIR="$WORK_HOME/logs/h3_gated_rounding_v3_setup_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
echo "Setup logs: $LOG_DIR"
step() { local name="$1"; shift; "$@" 2>&1 | tee "$LOG_DIR/$name.log"; }
step sources python check_sources.py
ACTIVE="$WORK_HOME/logs/h3_gated_active_bundle.txt"
cp "$ACTIVE" "$LOG_DIR/previous_bundle.txt"
OLD_BUNDLE="$(cat "$ACTIVE")"
OLD_MODE="$(python -c 'import json,sys,pathlib; m=json.loads((pathlib.Path(sys.argv[1])/"bundle.json").read_text()); print("down" if m.get("semantics","").startswith("FP32 mul nearest-even then add toward negative infinity;") else "nearest")' "$OLD_BUNDLE")"
step before python -X faulthandler -u probe_rounding.py --bundle "$OLD_BUNDLE" --expected "$OLD_MODE" --label before
step export "$WORK_HOME/venvs/h3-ninetoothed-026-triton380/bin/python" -u prepare_bundle.py
export INFINICORE_H3_GATED_BUNDLE
INFINICORE_H3_GATED_BUNDLE="$(cat "$LOG_DIR/bundle_path.txt")"
step after python -X faulthandler -u probe_rounding.py --bundle "$INFINICORE_H3_GATED_BUNDLE" --expected down --label after
step integration python -X faulthandler -u verify_integration.py
python - "$ACTIVE" "$LOG_DIR/previous_bundle.txt" "$LOG_DIR/bundle_path.txt" <<'PY'
from pathlib import Path
import sys
active,previous,new=map(Path,sys.argv[1:])
if active.read_bytes()!=previous.read_bytes():
    raise RuntimeError('Active bundle changed concurrently; not publishing')
temporary=active.with_name(active.name+'.rounding-tmp')
with temporary.open('xb') as f:f.write(new.read_bytes())
temporary.replace(active)
print('Verified bundle selected:',active.read_text().strip())
PY
echo "ROUNDING REPAIR AND INTEGRATION PASSED; setup logs: $LOG_DIR"
