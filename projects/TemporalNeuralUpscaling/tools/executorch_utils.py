"""Shared helpers for ExecuTorch export tools."""

import importlib.resources
import os
from pathlib import Path


def configure_bundled_flatc() -> Path:
    """Point ExecuTorch 1.3's Windows serializer at its bundled flatc.exe."""
    configured = os.getenv("FLATC_EXECUTABLE")
    if configured:
        return Path(configured)

    resource = importlib.resources.files("executorch.data.bin").joinpath("flatc.exe")
    if not resource.is_file():
        raise FileNotFoundError("ExecuTorch wheel does not contain executorch/data/bin/flatc.exe")
    flatc = Path(str(resource))
    os.environ["FLATC_EXECUTABLE"] = str(flatc)
    return flatc
