#!/usr/bin/env bash
# Source this file inside WSL before running ExecuTorch QNN commands.

set -o errexit
set -o nounset

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-$HOME/.cache/executorch/qnn/sdk-2.37.0.250724}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-$HOME/Android/Sdk/ndk/android-ndk-r26c}"

if [[ ! -f "$QNN_SDK_ROOT/QNN_README.txt" || ! -f "$QNN_SDK_ROOT/sdk.yaml" ]]; then
  echo "Invalid QNN_SDK_ROOT: required SDK markers are missing" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$ANDROID_NDK_ROOT/source.properties" ]]; then
  echo "Invalid ANDROID_NDK_ROOT: source.properties is missing" >&2
  return 1 2>/dev/null || exit 1
fi

qnn_host_lib="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
qnn_libcxx="${QNN_LIBCXX_ROOT:-$HOME/.cache/executorch/qnn/libcxx-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/x86_64-unknown-linux-gnu}"
for qnn_library_dir in "$qnn_host_lib" "$qnn_libcxx"; do
  if [[ ! -d "$qnn_library_dir" ]]; then
    echo "Required QNN host library directory is missing: $qnn_library_dir" >&2
    return 1 2>/dev/null || exit 1
  fi
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$qnn_library_dir:"*) ;;
    *) export LD_LIBRARY_PATH="$qnn_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
done

unset qnn_host_lib qnn_libcxx qnn_library_dir
