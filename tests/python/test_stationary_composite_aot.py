"""Composite packaging must preserve point-model identity and primitive dispatch."""

from types import SimpleNamespace

import pytest
from generativeqc._stationary_composite_cuda import composite_stationary_aot_profile
from generativeqc._stationary_cuda import _artifact_derivative_requests
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_POLICY
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_cuda import qualified_sp_requests


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize(
    "changed", (None, "point-model", "partition", "nonlocal-partition")
)
def test_composite_aot_exact_default_domain(spin: str, changed: str | None) -> None:
    """Default composite profiles are exact; custom grid/point contracts retain JIT."""
    source = SimpleNamespace(
        hamiltonian="all-electron",
        functional_code=4,
        method_ir=resolve_method("WB97M-V", spin=spin),
        nonlocal_density_policy=MOLECULAR_VV10_DENSITY_POLICY,
        grid_spec=SimpleNamespace(
            partition_iterations=2 if changed == "partition" else 3
        ),
        nonlocal_grid_spec=SimpleNamespace(
            partition_iterations=2 if changed == "nonlocal-partition" else 3
        ),
        _batch=SimpleNamespace(
            _calculator=SimpleNamespace(
                _ks_options=SimpleNamespace(
                    scf_domain=(
                        "semilocal-scaled-v1/pbe-spin-c2-1e-18"
                        if changed == "point-model"
                        else "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"
                    )
                )
            )
        ),
    )
    profile = composite_stationary_aot_profile(
        SimpleNamespace(_source=source, identity=SimpleNamespace(spin=spin))
    )
    assert (profile is not None) == (changed is None)
    if profile is not None:
        assert profile.spin == spin
        assert profile.plan.method.identity == source.method_ir.identity


def test_geometry_only_packaged_nuclear_dispatch() -> None:
    """Packaged nuclear dispatch uses its full inventory; JIT keeps its sole kind."""
    request = (("nuclear", ()),)
    package = SimpleNamespace(metadata={"artifact_kind": "packaged-aot"})
    inventory = _artifact_derivative_requests(request, package)
    assert inventory == qualified_sp_requests()
    assert inventory.index(request[0]) > 0
    assert (
        _artifact_derivative_requests(request, SimpleNamespace(metadata={})) == request
    )
