"""Keep shared force output policy independent of scheduling and mathematics."""

from pathlib import Path

import pytest

from tools.check_scf_structure import audit_scf_structure


def test_force_contraction_can_consume_output_policy(tmp_path: Path) -> None:
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_force_sources.hpp").write_text("// Output layout only\n")
    (source / "direct_force_low_order_sources.cuh").write_text(
        '#include "direct_force_sources.hpp"\n'
    )
    assert not audit_scf_structure(tmp_path)["errors"]


@pytest.mark.parametrize(
    "target",
    [
        "direct_force_low_order_sources.cuh",
        "direct_angular_force.hpp",
        "rhf_policy.hpp",
    ],
)
def test_force_output_policy_is_a_leaf(tmp_path: Path, target: str) -> None:
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / target).write_text("// Consumer, scheduler or method policy\n")
    (source / "direct_force_sources.hpp").write_text(f'#include "{target}"\n')
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden cuda_direct_force_sources dependency" in errors[0]
