"""Automatic Libxc representation uses default generic lowering without evidence admission."""

import pytest
from generativeqc_compiler.xc import functional
from generativeqc_compiler.xc.semilocal_codegen import build_roots
from generativeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_entire_automatic_inventory_reaches_default_energy_lowering(spin: str) -> None:
    """Every structurally supported automatic component has an energy Graph."""
    for name in AUTO_BULK_COMPONENTS:
        spec = functional(name, spin=spin)
        # This legacy payload bit describes the old qualification lane and must
        # not silently become a positive user-admission whitelist.
        assert spec.to_payload()["production_admitted"] is False
        graph, roots, identity = build_roots(spec, ((),))
        assert graph is not None
        assert len(roots) == 1
        assert len(identity) == 64


@pytest.mark.parametrize("production", (False, True))
@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"))
def test_native_lowering_does_not_restore_positive_admission_gate(
    name: str, production: bool
) -> None:
    """The legacy production flag cannot turn default-allow into a whitelist."""
    graph, roots, identity = build_roots(functional(name), ((),), production=production)
    assert graph is not None
    assert len(roots) == 1
    assert len(identity) == 64
