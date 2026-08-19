#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/tools/qnn_env.sh"

executorch_source="${TNU_EXECUTORCH_SOURCE:-$HOME/.cache/executorch/executorch}"
qnn_python="${TNU_QNN_PYTHON:-$HOME/.venvs/tnu-qnn/bin/python}"
expected_commit="e2f18eb23c45bd22ca332b0b8b49a81de304b472"

if [[ ! -d "$executorch_source/.git" ]]; then
  echo "ExecuTorch v1.3.1 source checkout is missing: $executorch_source" >&2
  exit 1
fi
if [[ "$(git -C "$executorch_source" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "ExecuTorch source is not pinned to v1.3.1 commit $expected_commit" >&2
  exit 1
fi
if [[ ! -x "$qnn_python" ]]; then
  echo "QNN Python environment is missing: $qnn_python" >&2
  exit 1
fi

export PYTHON_EXECUTABLE="$qnn_python"
export PATH="$(dirname "$qnn_python"):$PATH"
cd "$executorch_source"
exec ./backends/qualcomm/scripts/build.sh \
  --skip_x86_64 \
  --release \
  --job_number "${TNU_BUILD_JOBS:-8}" \
  "$@"
