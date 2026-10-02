"""Qualification and retirement gates for GFN2 external point-charge forces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.gfn2_external_point_charge_force import (
    build_gfn2_external_point_charge_force_projection_program,
    build_gfn2_external_point_charge_force_weight_program,
    build_gfn2_external_point_charge_pair_primal,
    build_gfn2_external_point_charge_pair_vjp,
)
from generativeqc_compiler.tensor import Program, execute

ROOT = Path(__file__).resolve().parents[2]


def _scalar_outputs(program: Program, **feeds: float) -> dict[str, float]:
    result = execute(
        program,
        {name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()},
    ).outputs
    return {name: float(np.asarray(value)) for name, value in result.items()}


@pytest.mark.parametrize(
    ("dx", "dy", "dz", "inverse_hardness", "shell_charge", "point_charge"),
    (
        (1.2, -0.7, 0.4, 1.8, 0.8, -0.3),
        (-2.4, 0.2, 1.1, 0.9, -1.2, 0.6),
        (0.03, -0.04, 0.05, 2.1, 0.2, 0.9),
    ),
)
def test_force_lowering_matches_negative_generated_pair_vjp(
    dx: float,
    dy: float,
    dz: float,
    inverse_hardness: float,
    shell_charge: float,
    point_charge: float,
) -> None:
    feeds = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "inverse_average_hardness": inverse_hardness,
        "shell_charge": shell_charge,
        "point_charge": point_charge,
    }
    primal = _scalar_outputs(build_gfn2_external_point_charge_pair_primal(), **feeds)
    reverse = _scalar_outputs(
        build_gfn2_external_point_charge_pair_vjp().program,
        **feeds,
        bar_pair_energy=1.0,
    )
    weight = _scalar_outputs(
        build_gfn2_external_point_charge_force_weight_program(),
        kernel=primal["kernel"],
        shell_charge=shell_charge,
        point_charge=point_charge,
    )["weight"]
    force = _scalar_outputs(
        build_gfn2_external_point_charge_force_projection_program(),
        weight=weight,
        dx=dx,
        dy=dy,
        dz=dz,
    )
    np.testing.assert_allclose(
        [force["fx"], force["fy"], force["fz"]],
        [-reverse["bar_dx"], -reverse["bar_dy"], -reverse["bar_dz"]],
        rtol=2e-15,
        atol=2e-16,
    )


def test_native_codegen_needs_no_site_packages(tmp_path: Path) -> None:
    output = tmp_path / "generated_gfn2_external_point_charge_force.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_external_point_charge_force.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = output.read_text(encoding="utf-8")
    assert "gfn2_external_point_charge_pair_vjp_hash" in source
    assert "evaluate_gfn2_external_point_charge_force" in source
    assert "__host__ __device__" in source


def test_production_force_consumers_do_not_restore_handwritten_q_over_r3() -> None:
    cpu = (ROOT / "src/xtb/native/src/model/gfn2/external_point_charges.cpp").read_text(
        encoding="utf-8"
    )
    cuda = (
        ROOT / "src/xtb/native/src/backends/cuda/gfn2_external_point_charges.cu"
    ).read_text(encoding="utf-8")
    for source in (cpu, cuda):
        assert "generated_gfn2_external_point_charge_force.hpp" in source
        assert "evaluate_gfn2_external_point_charge_force" in source
        assert "force_scale = shell_charge" not in source
        assert "force_scale = shell_charges" not in source
        assert "inverse_distance * inverse_distance * inverse_distance" not in source
