"""Prepared J/K selection is a host adapter over the compiler inventory."""

from pathlib import Path

import pytest

from tools.check_scf_structure import audit_scf_structure


@pytest.mark.parametrize("owner", ["scf/cuda/direct_coulomb.cpp", "scf/cuda_rhf.cpp"])
def test_host_owners_can_borrow_lowering_and_trace_interfaces(
    tmp_path: Path, owner: str
) -> None:
    source = tmp_path / "src"
    for header in (
        "scf/cuda/direct_fock_lowering.hpp",
        "scf/aot_shell_registry.hpp",
        "runtime/cuda_component_trace.hpp",
    ):
        path = source / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// Host interface\n")
    (source / "scf/cuda/direct_fock_lowering.hpp").write_text(
        '#include "scf/aot_shell_registry.hpp"\n'
    )
    (source / owner).write_text(
        '#include "scf/cuda/direct_fock_lowering.hpp"\n'
        '#include "runtime/cuda_component_trace.hpp"\n'
    )
    report = audit_scf_structure(tmp_path)
    assert not report["errors"]
    assert any(
        module["path"] == "scf/cuda/direct_fock_lowering.hpp"
        for module in report["modules"]
    )


@pytest.mark.parametrize(
    "target",
    [
        "scf/cuda/direct_coulomb.hpp",
        "scf/cuda/direct_fock_quartet.cuh",
        "runtime/resource_cuda.cuh",
        "scf/aot_shell_registry_stub.cpp",
    ],
)
def test_lowering_adapter_cannot_acquire_lifetime_or_device_state(
    tmp_path: Path, target: str
) -> None:
    source = tmp_path / "src"
    (source / "scf/cuda").mkdir(parents=True)
    dependency = source / target
    dependency.parent.mkdir(parents=True, exist_ok=True)
    dependency.write_text("// Forbidden dependency\n")
    (source / "scf/cuda/direct_fock_lowering.hpp").write_text(f'#include "{target}"\n')
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert f"forbidden cuda_direct_fock_lowering dependency on {target}" in errors[0]
