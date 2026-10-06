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


@pytest.mark.parametrize(
    "consumer",
    [
        "direct_angular_force.cu",
        "direct_bounded_exact_force.cu",
        "direct_bounded_fallback.cu",
    ],
)
def test_force_consumers_can_use_shared_execution(
    tmp_path: Path, consumer: str
) -> None:
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_force_execution.cuh").write_text(
        '#include "direct_force_low_order_sources.cuh"\n'
    )
    (source / "direct_force_low_order_sources.cuh").write_text("// Scientific task\n")
    (source / consumer).write_text('#include "direct_force_execution.cuh"\n')
    assert not audit_scf_structure(tmp_path)["errors"]


def test_force_interface_admits_layout_but_not_execution(tmp_path: Path) -> None:
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    for header in ("direct_force_sources.hpp", "direct_force_execution.cuh"):
        (source / header).write_text("// Layout or device execution\n")
    interface = source / "direct_angular_force.hpp"
    interface.write_text('#include "direct_force_sources.hpp"\n')
    assert not audit_scf_structure(tmp_path)["errors"]
    interface.write_text('#include "direct_force_execution.cuh"\n')
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden cuda_direct_kernel_interfaces dependency" in errors[0]


def test_force_execution_cannot_acquire_method_policy(tmp_path: Path) -> None:
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "rhf_policy.hpp").write_text("// Method policy\n")
    (source / "direct_force_execution.cuh").write_text('#include "rhf_policy.hpp"\n')
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert "forbidden cuda_direct_contractions dependency" in errors[0]
