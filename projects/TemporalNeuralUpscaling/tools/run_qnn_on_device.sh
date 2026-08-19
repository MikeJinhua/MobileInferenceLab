#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/tools/qnn_env.sh"

executorch_source="${TNU_EXECUTORCH_SOURCE:-$HOME/.cache/executorch/executorch}"
qnn_python="${TNU_QNN_PYTHON:-$HOME/.venvs/tnu-qnn/bin/python}"
device_dir="${TNU_QNN_DEVICE_DIR:-/data/local/tmp/tnu_qnn}"
adb_command="${ADB:-$(command -v adb.exe || command -v adb || true)}"

if [[ -z "$adb_command" ]]; then
  echo "ADB was not found" >&2
  exit 1
fi

runner="$executorch_source/build-android/examples/qualcomm/executor_runner/qnn_executor_runner"
backend="$executorch_source/build-android/backends/qualcomm/libqnn_executorch_backend.so"
model="$project_root/results/p4_3/spatial_sr_qnn_htp_fp16.pte"
result_dir="$project_root/results/p4_4"

cd "$project_root"
"$qnn_python" -m tools.prepare_qnn_device_inputs --output-dir "$result_dir"

required_files=(
  "$runner"
  "$backend"
  "$model"
  "$result_dir/input.raw"
  "$result_dir/input_list.txt"
  "$project_root/tools/qnn_memory_probe.sh"
  "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"
  "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so"
  "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so"
  "$QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so"
)
for source_file in "${required_files[@]}"; do
  if [[ ! -f "$source_file" ]]; then
    echo "Required device file is missing: $source_file" >&2
    exit 1
  fi
done

"$adb_command" get-state >/dev/null
"$adb_command" shell "mkdir -p '$device_dir/outputs'"
"$adb_command" shell "rm -f '$device_dir/outputs/'*.raw '$device_dir/inference_speed.txt' '$device_dir/etdump.etdp'"
for source_file in "${required_files[@]}"; do
  "$adb_command" push "$(wslpath -w "$source_file")" "$device_dir/$(basename "$source_file")" >/dev/null
done
"$adb_command" shell "chmod 700 '$device_dir/qnn_executor_runner'"
"$adb_command" shell "chmod 700 '$device_dir/qnn_memory_probe.sh'"

"$adb_command" shell "cd '$device_dir' && export LD_LIBRARY_PATH='$device_dir' && export ADSP_LIBRARY_PATH='$device_dir' && ./qnn_executor_runner --model_path=spatial_sr_qnn_htp_fp16.pte --input_list_path=input_list.txt --output_folder_path=outputs --warm_up=5 --iteration=20 --performance_output_path=inference_speed.txt --etdump_path=etdump.etdp" 2>&1 \
  | tee "$result_dir/device_runner.log"

"$adb_command" pull "$device_dir/outputs/output_0_0.raw" "$(wslpath -w "$result_dir/device_output_0.raw")" >/dev/null
"$adb_command" pull "$device_dir/outputs/output_19_0.raw" "$(wslpath -w "$result_dir/device_output_1.raw")" >/dev/null
"$adb_command" pull "$device_dir/inference_speed.txt" "$(wslpath -w "$result_dir/inference_speed.txt")" >/dev/null
"$adb_command" pull "$device_dir/etdump.etdp" "$(wslpath -w "$result_dir/etdump.etdp")" >/dev/null
"$adb_command" shell "cd '$device_dir' && export LD_LIBRARY_PATH='$device_dir' && export ADSP_LIBRARY_PATH='$device_dir' && ./qnn_memory_probe.sh"
"$adb_command" pull "$device_dir/memory_peak_rss_kb.txt" "$(wslpath -w "$result_dir/memory_peak_rss_kb.txt")" >/dev/null
"$adb_command" pull "$device_dir/memory_runner.log" "$(wslpath -w "$result_dir/memory_runner.log")" >/dev/null
"$qnn_python" -m tools.analyze_qnn_device_output --result-dir "$result_dir"
