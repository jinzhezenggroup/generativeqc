"""Geometry/pair compiler contracts lowered through the shared TensorIR.

Load public numerical exports on demand so scalar build-time generators remain
usable without NumPy or the native runtime.
"""

import typing
from importlib import import_module

_EXPORTS = {
    "D3_COMPILER_VERSION": ".d3",
    "D3_RADII_SHA256": ".d3",
    "D3_TABLE_SHA256": ".d3",
    "GFN1_CUTOFF_BOHR": ".gfn1",
    "GFN1_HALOGEN_CUTOFF_BOHR": ".gfn1",
    "GFN1_HALOGEN_PARAMETER_IDENTITY": ".gfn1",
    "GFN1_HALOGEN_VERSION": ".gfn1",
    "GFN1_SHORT_RANGE_PARAMETER_IDENTITY": ".gfn1",
    "GFN1_SHORT_RANGE_VERSION": ".gfn1",
    "GFN2_CUTOFF_BOHR": ".gfn2",
    "GFN2_SHORT_RANGE_PARAMETER_IDENTITY": ".gfn2",
    "GFN2_SHORT_RANGE_VERSION": ".gfn2",
    "LOWERING_VERSION": ".ir",
    "PAIR_OWNERSHIP": ".ir",
    "D3CompilerSpec": ".d3",
    "D3GeneratedCudaDiagnostic": ".d3_cuda",
    "D3GeneratedCudaExecution": ".d3_cuda",
    "D3GeometryBatchProgram": ".d3",
    "D3GeometryProgram": ".d3",
    "D3PairTopology": ".d3",
    "GeometryIR": ".ir",
    "Gfn1GeometryElement": ".gfn1",
    "Gfn1HalogenGeometryProgram": ".gfn1",
    "Gfn1ShortRangeProgram": ".gfn1",
    "Gfn2ShortRangeBatchProgram": ".gfn2",
    "Gfn2ShortRangeElement": ".gfn2",
    "Gfn2ShortRangeProgram": ".gfn2",
    "PairCutoff": ".ir",
    "PairProgram": ".ir",
    "PairTensorContext": ".ir",
    "PairTopology": ".ir",
    "PreparedD3CudaBatch": ".d3_cuda",
    "build_d3_batch_geometry_program": ".d3",
    "build_d3_batch_pair_topology": ".d3",
    "build_d3_geometry_program": ".d3",
    "build_d3_pair_topology": ".d3",
    "build_gfn1_halogen_geometry_program": ".gfn1",
    "build_gfn1_halogen_topology": ".gfn1",
    "build_gfn1_pair_topology": ".gfn1",
    "build_gfn1_short_range_program": ".gfn1",
    "build_gfn2_batch_pair_topology": ".gfn2",
    "build_gfn2_pair_topology": ".gfn2",
    "build_gfn2_short_range_batch_program": ".gfn2",
    "build_gfn2_short_range_program": ".gfn2",
    "build_pair_program": ".ir",
    "compile_d3_bj": ".d3",
    "compile_d3_bj_batch": ".d3",
    "d3_geometry": ".d3",
    "execute_d3_bj": ".d3_execution",
    "execute_d3_bj_batch": ".d3_execution",
    "gfn1_element_parameters": ".gfn1",
    "gfn1_geometry": ".gfn1",
    "gfn2_element_parameters": ".gfn2",
    "gfn2_geometry": ".gfn2",
    "inverse_power_program": ".ir",
    "lower_geometry": ".ir",
    "pair_to_atom": ".ir",
    "pair_to_system": ".ir",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> typing.Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
