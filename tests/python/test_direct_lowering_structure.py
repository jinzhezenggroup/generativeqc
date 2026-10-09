"""Prepared J/K selection is a host adapter over the compiler inventory."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools.check_scf_structure import audit_scf_structure

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_aot_disabled_registry_links_prepared_launchers(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """The real no-AOT registry supplies every prepared streaming selection."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\n"
        "using cudaStream_t = void*;\n"
        "enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, "
        "cudaErrorNotSupported = 801 };\n"
    )
    driver = tmp_path / "registry.cpp"
    driver.write_text(
        '#include "scf/cuda/direct_fock_lowering.hpp"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "  using namespace generativeqc::scf::cuda_execution;\n"
        "  for (unsigned choice = 0; choice < 5; ++choice) {\n"
        "    DirectExchangeSelection selection{choice == 1, choice == 2, choice == 3};\n"
        "    if (choice == 4) selection.task_schedule =\n"
        "        generativeqc::scf::detail::GeneratedExchangeTaskSchedule::Work;\n"
        "    auto launch = direct_fock_streaming_launcher(selection, 0, false);\n"
        "    assert(launch(0, nullptr, false, 0, nullptr, nullptr, nullptr,\n"
        "                  nullptr, nullptr, 0, false, 0, nullptr, nullptr,\n"
        "                  nullptr, nullptr, nullptr, nullptr) == cudaErrorNotSupported);\n"
        "  }\n"
        "}\n"
    )
    executable = tmp_path / "registry"
    native_cxx.build_executable(
        (driver, root / "src/scf/aot_shell_registry_stub.cpp"),
        executable,
        compile_args=("-std=c++17", f"-I{tmp_path}", f"-I{root / 'src'}"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_prepared_k_owner_uses_one_selection_for_topology_and_launch() -> None:
    """Keep the frozen queue schedule and class/spin dispatch in the same policy."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/direct_coulomb.cpp").read_text()
    header = (root / "src/scf/cuda/direct_coulomb.hpp").read_text()
    assert "DirectExchangeSelection selection{};" in header
    assert "prepare_direct_exchange_selection(shared->value_class_mask)" in source
    assert "plan->selection.task_schedule}" in source
    assert "direct_fock_streaming_launcher(p.selection, cls, unrestricted)" in source
    assert "prepare_direct_fock_rys_mask(true)" not in source
    assert "prepare_direct_fock_k_block_mask()" not in source
    assert "prepare_direct_fock_rys_task_mask()" not in source


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
