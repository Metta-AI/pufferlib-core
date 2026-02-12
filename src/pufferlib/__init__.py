"""
PufferLib Core - Minimal vectorized environment functionality
"""

import importlib
import sys


# Import individual modules with delayed loading to avoid circular imports
def _import_modules():
    from . import pufferlib, spaces

    # Temporarily add pufferlib to the current module namespace to resolve imports
    current_module = sys.modules[__name__]
    # Re-export core API and exceptions to match upstream import patterns.
    current_module.PufferEnv = pufferlib.PufferEnv
    current_module.set_buffers = pufferlib.set_buffers
    current_module.unroll_nested_dict = pufferlib.unroll_nested_dict
    current_module.APIUsageError = pufferlib.APIUsageError
    current_module.InvalidAgentError = pufferlib.InvalidAgentError
    current_module.EnvironmentSetupError = pufferlib.EnvironmentSetupError

    from . import emulation, vector

    # Try to import C extensions if available
    try:
        from . import _C

        current_module._C = _C
    except ImportError:
        # C extensions not available, continue without them
        pass

    # Import PyTorch modules (now required dependencies)
    from . import models, pytorch

    current_module.pytorch = pytorch
    current_module.models = models
    pytorch_modules = [pytorch, models]

    return spaces, pufferlib, emulation, vector, pytorch_modules


# Perform the imports
spaces, pufferlib, emulation, vector, pytorch_modules = _import_modules()


def __getattr__(name: str):
    # Avoid importing heavy training dependencies (torch.distributed) on simple `import pufferlib`.
    if name == "pufferl":
        mod = importlib.import_module(".pufferl", __name__)
        globals()["pufferl"] = mod
        return mod
    raise AttributeError(name)

# Keep this in sync with `packages/pufferlib-core/pyproject.toml`.
__version__ = "3.0.17"
__all__ = [
    "spaces",
    "emulation",
    "vector",
    "pufferlib",
    "pytorch",
    "models",
    "pufferl",
    "PufferEnv",
    "set_buffers",
    "unroll_nested_dict",
    "APIUsageError",
    "InvalidAgentError",
    "EnvironmentSetupError",
]
