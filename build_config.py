from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional

from torch.utils.cpp_extension import CUDA_HOME


class ExtensionBuildConfig(NamedTuple):
    build_with_cuda: bool
    cuda_home: Optional[str]
    warning: Optional[str]


_AUTO = object()


def discover_cuda_home() -> Optional[str]:
    configured_cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or CUDA_HOME
    if configured_cuda_home is not None:
        return configured_cuda_home
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return None
    return str(Path(nvcc_path).resolve().parent.parent)


def discover_torch_cuda_arch_list() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    archs = sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and all(part.isdigit() for part in line.strip().split(".", maxsplit=1))
        },
        key=lambda arch: tuple(int(part) for part in arch.split(".")),
    )
    if not archs:
        return None
    return ";".join(archs)


def resolve_extension_build_config(
    *,
    force_cuda: bool,
    disable_cuda: bool,
    cuda_home: Optional[str] | object = _AUTO,
    cuda_available: Optional[bool] = None,
    has_nvcc: Optional[bool] = None,
) -> ExtensionBuildConfig:
    resolved_cuda_home = discover_cuda_home() if cuda_home is _AUTO else cuda_home
    resolved_cuda_available = _cuda_available() if cuda_available is None else cuda_available
    resolved_has_nvcc = _has_nvcc(resolved_cuda_home) if has_nvcc is None else has_nvcc

    if force_cuda and disable_cuda:
        raise RuntimeError("PUFFERLIB_BUILD_CUDA=1 and PUFFERLIB_DISABLE_CUDA=1 are mutually exclusive")

    if force_cuda and (resolved_cuda_home is None or not resolved_has_nvcc):
        raise RuntimeError(
            "PUFFERLIB_BUILD_CUDA=1 requires a CUDA toolkit with nvcc. "
            "Set CUDA_HOME/CUDA_PATH to a full toolkit install or add nvcc to PATH."
        )

    build_with_cuda = not disable_cuda and resolved_cuda_home is not None and resolved_has_nvcc
    warning = None
    if not build_with_cuda and not disable_cuda and resolved_cuda_available:
        detail = (
            "CUDA_HOME/CUDA_PATH is set, but no nvcc compiler was found there or on PATH. "
            "Building pufferlib-core CPU-only; CUDA training will fail until you install a full toolkit or fix "
            "CUDA_HOME."
            if resolved_cuda_home is not None and not resolved_has_nvcc
            else "no CUDA toolkit was found via CUDA_HOME/CUDA_PATH/nvcc. Building pufferlib-core CPU-only; "
            "CUDA training will fail until you reinstall after setting CUDA_HOME."
        )
        warning = f"CUDA-capable PyTorch detected a visible NVIDIA GPU, but {detail}"

    return ExtensionBuildConfig(
        build_with_cuda=build_with_cuda,
        cuda_home=resolved_cuda_home,
        warning=warning,
    )


def _cuda_available() -> bool:
    import torch  # noqa: PLC0415

    try:
        return bool(torch.version.cuda) and torch.cuda.is_available()
    except RuntimeError:
        return False


def _has_nvcc(cuda_home: Optional[str]) -> bool:
    if cuda_home is not None:
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        if Path(cuda_home, "bin", nvcc_name).is_file():
            return True
    return shutil.which("nvcc") is not None
