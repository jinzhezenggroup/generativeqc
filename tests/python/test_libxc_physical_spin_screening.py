"""Empty-spin Maple screening uses physical coordinates before differentiation."""

import pytest
from generativeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from generativeqc_compiler.xc.libxc_work import automatic_work_policy


@pytest.mark.parametrize(
    "name",
    ("GGA_X_APBE", "GGA_X_B86", "MGGA_X_LTA", "MGGA_X_GVT4"),
)
def test_screening_predicate_is_exact_physical_spin_density(name: str) -> None:
    program = build_bulk_runtime_program(name, spin="polarized", order=1)
    for spin_name in ("rho_a", "rho_b"):
        assert any(
            node.operation == "select_le"
            and program.graph.nodes[node.arguments[0]].operation == "variable"
            and program.graph.nodes[node.arguments[0]].payload == spin_name
            for node in program.graph.nodes
        )


@pytest.mark.parametrize(
    "name",
    ("GGA_X_APBE", "GGA_X_B86", "MGGA_X_LTA", "MGGA_X_GVT4"),
)
@pytest.mark.parametrize("minority_fraction", (0.25, 0.5, 1.0))
def test_below_or_equal_threshold_has_no_spurious_minor_spin_vxc(
    name: str,
    minority_fraction: float,
) -> None:
    program = build_bulk_runtime_program(name, spin="polarized", order=1)
    threshold = automatic_work_policy(name).density_threshold
    features = {
        "rho_a": 0.26285804,
        "rho_b": threshold * minority_fraction,
        "sigma_aa": 1e-3,
        "sigma_ab": 0.0,
        "sigma_bb": threshold ** (8 / 3),
        "tau_a": 0.1,
        "tau_b": 1e-20,
    }
    for feature in ("rho_b", "sigma_bb", "tau_b"):
        if feature not in program.spec.features:
            continue
        output = program.roots[1 + program.spec.features.index(feature)]
        assert program.graph.evaluate(output, features) == pytest.approx(0, abs=2e-12)
