"""Geometry/pair contracts with dependency-light scalar source generation."""

from importlib import import_module

_LAZY_EXPORTS = {
    "D3_COMPILER_VERSION": "d3",
    "D3_RADII_SHA256": "d3",
    "D3_TABLE_SHA256": "d3",
    "GFN1_CUTOFF_BOHR": "gfn1",
    "GFN1_HALOGEN_CUTOFF_BOHR": "gfn1",
    "GFN1_HALOGEN_PARAMETER_IDENTITY": "gfn1",
    "GFN1_HALOGEN_VERSION": "gfn1",
    "GFN1_SHORT_RANGE_PARAMETER_IDENTITY": "gfn1",
    "GFN1_SHORT_RANGE_VERSION": "gfn1",
    "GFN2_CUTOFF_BOHR": "gfn2",
    "GFN2_SHORT_RANGE_PARAMETER_IDENTITY": "gfn2",
    "GFN2_SHORT_RANGE_VERSION": "gfn2",
    "LOWERING_VERSION": "ir",
    "PAIR_OWNERSHIP": "ir",
    "D3CompilerSpec": "d3",
    "D3GeneratedCudaDiagnostic": "d3_cuda",
    "D3GeneratedCudaExecution": "d3_cuda",
    "D3GeometryBatchProgram": "d3",
    "D3GeometryProgram": "d3",
    "D3PairTopology": "d3",
    "GeometryIR": "ir",
    "Gfn1GeometryElement": "gfn1",
    "Gfn1HalogenGeometryProgram": "gfn1",
    "Gfn1ShortRangeProgram": "gfn1",
    "Gfn2ShortRangeBatchProgram": "gfn2",
    "Gfn2ShortRangeElement": "gfn2",
    "Gfn2ShortRangeProgram": "gfn2",
    "PairCutoff": "ir",
    "PairProgram": "ir",
    "PairTensorContext": "ir",
    "PairTopology": "ir",
    "PreparedD3CudaBatch": "d3_cuda",
    "build_d3_batch_geometry_program": "d3",
    "build_d3_batch_pair_topology": "d3",
    "build_d3_geometry_program": "d3",
    "build_d3_pair_topology": "d3",
    "build_gfn1_halogen_geometry_program": "gfn1",
    "build_gfn1_halogen_topology": "gfn1",
    "build_gfn1_pair_topology": "gfn1",
    "build_gfn1_short_range_program": "gfn1",
    "build_gfn2_batch_pair_topology": "gfn2",
    "build_gfn2_pair_topology": "gfn2",
    "build_gfn2_short_range_batch_program": "gfn2",
    "build_gfn2_short_range_program": "gfn2",
    "build_pair_program": "ir",
    "compile_d3_bj": "d3",
    "compile_d3_bj_batch": "d3",
    "d3_geometry": "d3",
    "execute_d3_bj": "d3_execution",
    "execute_d3_bj_batch": "d3_execution",
    "gfn1_element_parameters": "gfn1",
    "gfn1_geometry": "gfn1",
    "gfn2_element_parameters": "gfn2",
    "gfn2_geometry": "gfn2",
    "inverse_power_program": "ir",
    "lower_geometry": "ir",
    "pair_to_atom": "ir",
    "pair_to_system": "ir",
}


def __getattr__(name: str) -> object:
    """Resolve numerical capabilities only when their public export is used."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep public discovery independent of optional numerical imports."""
    return sorted(set(globals()) | set(__all__))


__all__ = list(_LAZY_EXPORTS)
