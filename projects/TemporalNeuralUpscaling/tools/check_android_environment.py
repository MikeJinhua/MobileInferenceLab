"""Inspect the local Android CPU development prerequisites without installing them."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Iterable, Mapping, Optional


DEFAULT_SDK_CANDIDATES = (
    Path.home() / "AppData" / "Local" / "Android" / "Sdk",
    Path("C:/Android/Sdk"),
)


def first_existing_path(values: Iterable[Optional[str]]) -> Optional[Path]:
    for value in values:
        if value:
            path = Path(value).expanduser()
            if path.is_dir():
                return path.resolve()
    return None


def find_sdk_root(
    environment: Mapping[str, str], candidates: Iterable[Path] = DEFAULT_SDK_CANDIDATES
) -> Optional[Path]:
    return first_existing_path(
        [
            environment.get("ANDROID_HOME"),
            environment.get("ANDROID_SDK_ROOT"),
            *(str(path) for path in candidates),
        ]
    )


def command_path(name: str) -> Optional[str]:
    value = shutil.which(name)
    return str(Path(value).resolve()) if value else None


def adb_details(adb: Optional[str]) -> Dict[str, object]:
    if not adb:
        return {"version": None, "devices": [], "error": "adb not found"}
    try:
        version = subprocess.run(
            [adb, "version"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip().splitlines()
        device_output = subprocess.run(
            [adb, "devices", "-l"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        return {"version": None, "devices": [], "error": str(error)}
    devices = [
        line.strip()
        for line in device_output[1:]
        if line.strip() and " device" in line
    ]
    return {"version": version, "devices": devices, "error": None}


def inspect_environment(environment: Mapping[str, str] = os.environ) -> Dict[str, object]:
    sdk_root = find_sdk_root(environment)
    sdkmanager = command_path("sdkmanager")
    adb = command_path("adb")
    if sdk_root:
        sdkmanager_candidate = sdk_root / "cmdline-tools" / "latest" / "bin" / "sdkmanager.bat"
        adb_candidate = sdk_root / "platform-tools" / "adb.exe"
        if not sdkmanager and sdkmanager_candidate.is_file():
            sdkmanager = str(sdkmanager_candidate.resolve())
        if not adb and adb_candidate.is_file():
            adb = str(adb_candidate.resolve())

    java = command_path("java")
    javac = command_path("javac")
    java_home = first_existing_path([environment.get("JAVA_HOME")])
    if java_home:
        java_candidate = java_home / "bin" / "java.exe"
        javac_candidate = java_home / "bin" / "javac.exe"
        if not java and java_candidate.is_file():
            java = str(java_candidate.resolve())
        if not javac and javac_candidate.is_file():
            javac = str(javac_candidate.resolve())

    ndk_versions = []
    if sdk_root and (sdk_root / "ndk").is_dir():
        ndk_versions = sorted(path.name for path in (sdk_root / "ndk").iterdir() if path.is_dir())

    checks = {
        "jdk": bool(java and javac),
        "android_sdk": sdk_root is not None,
        "sdkmanager": sdkmanager is not None,
        "platform_tools": adb is not None,
        "android_ndk": bool(ndk_versions),
        "cmake": command_path("cmake") is not None,
        "ninja": command_path("ninja") is not None,
        "connected_device": False,
    }
    adb_report = adb_details(adb)
    checks["connected_device"] = bool(adb_report["devices"])
    required_for_p3_2 = (
        "jdk",
        "android_sdk",
        "sdkmanager",
        "platform_tools",
        "android_ndk",
        "cmake",
        "ninja",
    )
    return {
        "ready_for_android_build": all(checks[name] for name in required_for_p3_2),
        "ready_for_device_validation": all(checks.values()),
        "checks": checks,
        "paths": {
            "java": java,
            "javac": javac,
            "android_sdk": str(sdk_root) if sdk_root else None,
            "sdkmanager": sdkmanager,
            "adb": adb,
            "cmake": command_path("cmake"),
            "ninja": command_path("ninja"),
        },
        "ndk_versions": ndk_versions,
        "adb": adb_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict", action="store_true", help="return non-zero unless build prerequisites exist"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_environment()
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not report["ready_for_android_build"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
