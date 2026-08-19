#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source tools/qnn_env.sh

qnn_python="${TNU_QNN_PYTHON:-$HOME/.venvs/tnu-qnn/bin/python}"
if [[ ! -x "$qnn_python" ]]; then
  echo "QNN Python environment not found: $qnn_python" >&2
  exit 1
fi

export EXECUTORCH_BUILDING_WHEEL=1
export FLATC_EXECUTABLE="${FLATC_EXECUTABLE:-$HOME/.venvs/tnu-qnn/lib/python3.10/site-packages/executorch/data/bin/flatc}"
if [[ ! -x "$FLATC_EXECUTABLE" ]]; then
  echo "ExecuTorch flatc not found or not executable: $FLATC_EXECUTABLE" >&2
  exit 1
fi
exec "$qnn_python" -m tools.export_spatial_sr_qnn "$@"
