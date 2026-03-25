import importlib.util
import os
import platform
import sys
from pathlib import Path

from setuptools import setup

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

BUILD_CONFIG_PATH = Path(__file__).with_name("build_config.py")
BUILD_CONFIG_SPEC = importlib.util.spec_from_file_location("pufferlib_core_build_config", BUILD_CONFIG_PATH)
assert BUILD_CONFIG_SPEC is not None and BUILD_CONFIG_SPEC.loader is not None
build_config_module = importlib.util.module_from_spec(BUILD_CONFIG_SPEC)
BUILD_CONFIG_SPEC.loader.exec_module(build_config_module)
resolve_extension_build_config = build_config_module.resolve_extension_build_config

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"

# Compile args
cxx_args = ["-fdiagnostics-color=always"]
nvcc_args = []

if DEBUG:
    cxx_args += ["-O0", "-g"]
    nvcc_args += ["-O0", "-g"]
else:
    cxx_args += ["-O3"]
    nvcc_args += ["-O3"]

# Extensions setup
torch_sources = ["src/pufferlib/extensions/pufferlib.cpp"]

# Get torch library path for rpath
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

force_cuda = os.getenv("PUFFERLIB_BUILD_CUDA", "0") == "1"
disable_cuda = os.getenv("PUFFERLIB_DISABLE_CUDA", "0") == "1"
build_config = resolve_extension_build_config(force_cuda=force_cuda, disable_cuda=disable_cuda)
build_with_cuda = build_config.build_with_cuda

if build_with_cuda:
    extension_class = CUDAExtension
    # PufferLib 4.0 kernels: CUDA advantage + fused kernels live in separate CU files.
    torch_sources.append("src/pufferlib/extensions/cuda/advantage.cu")
    torch_sources.append("src/pufferlib/extensions/modules.cu")
    torch_sources.append("src/pufferlib/extensions/modules_bindings.cpp")
    print(f"Building with CUDA support (CUDA_HOME={build_config.cuda_home})")
else:
    extension_class = CppExtension
    message = (
        "Building with CPU-only support (PUFFERLIB_DISABLE_CUDA=1)"
        if disable_cuda
        else "Building with CPU-only support"
    )
    print(message)
    if build_config.warning is not None:
        print(f"WARNING: {build_config.warning}")

# Add rpath for torch libraries
extra_link_args = []
if platform.system() == "Darwin":  # macOS
    extra_link_args.extend([f"-Wl,-rpath,{torch_lib_path}", "-Wl,-headerpad_max_install_names"])
elif platform.system() == "Linux":  # Linux
    extra_link_args.extend([f"-Wl,-rpath,{torch_lib_path}", "-Wl,-rpath,$ORIGIN"])

ext_modules = [
    extension_class(
        "pufferlib._C",
        torch_sources,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
        extra_link_args=extra_link_args,
    ),
]
cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
