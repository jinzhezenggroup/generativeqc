"""Generic CUDA shell lowering interface shared by production and tuning."""

from .lowering.common import (
    _emitted_component_names,
    _generic_task_component_setup,
    emit_uncached_primitive_geometry_cuda,
)
from .lowering.dispatch import emit_shell_class_fused_cuda

__all__ = [
    "_emitted_component_names",
    "_generic_task_component_setup",
    "emit_shell_class_fused_cuda",
    "emit_uncached_primitive_geometry_cuda",
]
