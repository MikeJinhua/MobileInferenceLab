"""Audit ExecuTorch QNN/HTP prerequisites without installing proprietary software."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

from tools.check_android_environment import find_sdk_root


SUPPORTED_SOCS = {
    "SM8350": "v68",
    "SM8450": "v69",
    "SM8475": "v69",
    "SM8550": "v73",
    "SM8650": "v75",
    "SM8750": "v79",
}
REQUIRED_QNN_MARKERS = ("QNN_README.txt", "sdk.yaml")
REQUIRED_QNN_LIBRARIES = (
    "lib/x86_64-linux-clang/libQnnHtp.so",
    "lib/x86_64-linux-clang/libQnnSystem.so",
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so",
)


def decode_process_output(value: bytes) -> str:
    if not value:
        return ""
    if b"\x00" in value:
        return value.decode("utf-16le", errors="replace").strip("\x00\r\n ")
    return value.decode("utf-8", errors="replace").strip()


def valid_qnn_sdk(path: Optional[Path]) -> bool:
    return bool(path and path.is_dir() and all((path / name).is_file() for name in REQUIRED_QNN_MARKERS))


def find_ndk_versions(sdk_root: Optional[Path], environment: Mapping[str, str]) -> Sequence[str]:
    explicit = environment.get("ANDROID_NDK_ROOT") or environment.get("ANDROID_NDK_HOME")
    if explicit and Path(explicit).is_dir():
        return (Path(explicit).name,)
    ndk_root = sdk_root / "ndk" if sdk_root else None
    if not ndk_root or not ndk_root.is_dir():
        return ()
    return tuple(sorted(path.name for path in ndk_root.iterdir() if path.is_dir()))


def parse_device_properties(output: str) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {"abi": None, "soc_model": None, "board": None, "hardware": None}
    for line in output.splitlines():
        match = re.fullmatch(r"([^=]+)=(.*)", line.strip())
        if match and match.group(1) in values:
            values[match.group(1)] = match.group(2) or None
    return values


def parse_os_release(output: str) -> Dict[str, str]:
    values = {}
    for line in output.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_key_values(output: str) -> Dict[str, str]:
    values = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def inspect_wsl_qnn(wsl: str, distro: str, run: Callable[..., subprocess.CompletedProcess]) -> Dict[str, str]:
    script = r'''
qnn=~/.cache/executorch/qnn/sdk-2.37.0.250724
ndk=~/Android/Sdk/ndk/android-ndk-r26c
venv=~/.venvs/tnu-qnn
echo "qnn_root=$qnn"
echo "ndk_root=$ndk"
echo "venv_python=$venv/bin/python"
echo "qnn_markers=$([[ -f "$qnn/QNN_README.txt" && -f "$qnn/sdk.yaml" ]] && echo true || echo false)"
echo "qnn_libraries=$([[ -f "$qnn/lib/x86_64-linux-clang/libQnnHtp.so" && -f "$qnn/lib/x86_64-linux-clang/libQnnSystem.so" && -f "$qnn/lib/aarch64-android/libQnnHtp.so" && -f "$qnn/lib/aarch64-android/libQnnSystem.so" && -f "$qnn/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so" ]] && echo true || echo false)"
echo "qnn_version=$(sed -n 's/^version:[[:space:]]*//p' "$qnn/sdk.yaml" 2>/dev/null | head -n1)"
echo "ndk_valid=$([[ -f "$ndk/source.properties" ]] && echo true || echo false)"
echo "tools_ready=$(for tool in python3 gcc g++ cmake ninja git curl unzip zip; do command -v "$tool" >/dev/null || exit 1; done && echo true || echo false)"
echo "venv_valid=$([[ -x "$venv/bin/python" ]] && echo true || echo false)"
echo "venv_pip_check=$([[ -x "$venv/bin/python" ]] && "$venv/bin/python" -m pip check >/dev/null 2>&1 && echo true || echo false)"
echo "venv_qnn_import=$([[ -x "$venv/bin/python" ]] && EXECUTORCH_BUILDING_WHEEL=1 "$venv/bin/python" -c 'import executorch.backends.qualcomm' >/dev/null 2>&1 && echo true || echo false)"
libcxx=~/.cache/executorch/qnn/libcxx-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/x86_64-unknown-linux-gnu
echo "qnn_host_load=$([[ -f "$libcxx/libc++.so.1" ]] && LD_LIBRARY_PATH="$qnn/lib/x86_64-linux-clang:$libcxx" "$venv/bin/python" -c 'import ctypes; ctypes.CDLL("libQnnHtp.so")' >/dev/null 2>&1 && echo true || echo false)"
'''
    result = run(
        [wsl, "-d", distro, "--", "bash", "-s"],
        input=script.encode("utf-8"), capture_output=True, check=False, timeout=30,
    )
    return parse_key_values(decode_process_output(result.stdout)) if result.returncode == 0 else {}


def inspect_environment(
    environment: Mapping[str, str] = os.environ,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Dict[str, object]:
    sdk_root = find_sdk_root(environment)
    qnn_value = environment.get("QNN_SDK_ROOT")
    qnn_root = Path(qnn_value).expanduser().resolve() if qnn_value else None
    ndk_versions = find_ndk_versions(sdk_root, environment)

    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    distros = []
    wsl_os = {}
    wsl_qnn = {}
    if wsl:
        try:
            result = run([wsl, "-l", "-q"], capture_output=True, check=False, timeout=10)
            output = decode_process_output(result.stdout)
            if result.returncode == 0 and "--install" not in output:
                distros = [line.strip() for line in output.splitlines() if line.strip()]
            if distros:
                result = run(
                    [wsl, "-d", distros[0], "--", "cat", "/etc/os-release"],
                    capture_output=True, check=False, timeout=10,
                )
                if result.returncode == 0:
                    wsl_os = parse_os_release(decode_process_output(result.stdout))
                wsl_qnn = inspect_wsl_qnn(wsl, distros[0], run)
        except (OSError, subprocess.SubprocessError):
            pass

    adb = shutil.which("adb")
    if not adb and sdk_root:
        candidate = sdk_root / "platform-tools" / "adb.exe"
        if candidate.is_file():
            adb = str(candidate)
    device = {"abi": None, "soc_model": None, "board": None, "hardware": None}
    if adb:
        command = (
            "echo abi=$(getprop ro.product.cpu.abi); "
            "echo soc_model=$(getprop ro.soc.model); "
            "echo board=$(getprop ro.board.platform); "
            "echo hardware=$(getprop ro.hardware)"
        )
        try:
            result = run([adb, "shell", command], capture_output=True, check=False, timeout=10)
            device = parse_device_properties(decode_process_output(result.stdout))
        except (OSError, subprocess.SubprocessError):
            pass

    soc = device["soc_model"]
    try:
        qualcomm_backend = importlib.util.find_spec("executorch.backends.qualcomm") is not None
    except ModuleNotFoundError:
        qualcomm_backend = False
    checks = {
        "wsl_ubuntu_22_04": (
            wsl_os.get("ID", "").lower() == "ubuntu"
            and wsl_os.get("VERSION_ID") == "22.04"
        ),
        "wsl_host_tools": wsl_qnn.get("tools_ready") == "true",
        "android_ndk": bool(ndk_versions) or wsl_qnn.get("ndk_valid") == "true",
        "qnn_sdk_root_set": qnn_root is not None or bool(wsl_qnn.get("qnn_root")),
        "qnn_sdk_layout": valid_qnn_sdk(qnn_root) or (
            wsl_qnn.get("qnn_markers") == "true"
            and wsl_qnn.get("qnn_libraries") == "true"
            and wsl_qnn.get("qnn_version") == "2.37.0"
        ),
        "wsl_qnn_venv": wsl_qnn.get("venv_valid") == "true",
        "wsl_python_dependencies": wsl_qnn.get("venv_pip_check") == "true",
        "wsl_qualcomm_backend": wsl_qnn.get("venv_qnn_import") == "true",
        "wsl_qnn_host_load": wsl_qnn.get("qnn_host_load") == "true",
        "executorch_qualcomm_backend": qualcomm_backend,
        "py_cpuinfo": importlib.util.find_spec("cpuinfo") is not None,
        "device_connected": device["abi"] is not None,
        "device_arm64": device["abi"] == "arm64-v8a",
        "device_soc_supported": soc in SUPPORTED_SOCS,
    }
    required = (
        "wsl_ubuntu_22_04", "wsl_host_tools", "android_ndk", "qnn_sdk_layout", "wsl_qnn_venv",
        "wsl_python_dependencies", "wsl_qualcomm_backend",
        "wsl_qnn_host_load",
        "executorch_qualcomm_backend", "py_cpuinfo", "device_arm64", "device_soc_supported",
    )
    return {
        "ready_for_qnn_export_and_device_validation": all(checks[name] for name in required),
        "checks": checks,
        "wsl_distributions": distros,
        "wsl_os_release": wsl_os,
        "wsl_qnn": wsl_qnn,
        "ndk_versions": list(ndk_versions),
        "qnn_sdk_root": str(qnn_root) if qnn_root else None,
        "qnn_sdk_markers": list(REQUIRED_QNN_MARKERS),
        "device": {**device, "htp_arch": SUPPORTED_SOCS.get(soc)},
        "scope": "read-only readiness audit; no QNN model was exported or executed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_environment()
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not report["ready_for_qnn_export_and_device_validation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
