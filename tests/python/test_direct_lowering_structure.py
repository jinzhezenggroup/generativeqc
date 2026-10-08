"""Prepared J/K selection is a host adapter over the compiler inventory."""

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.check_scf_structure import audit_scf_structure


def test_aot_disabled_registry_links_prepared_launchers(tmp_path: Path) -> None:
    """The real no-AOT registry supplies every prepared streaming selection."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
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
        "  for (unsigned choice = 0; choice < 4; ++choice) {\n"
        "    auto launch = direct_fock_streaming_launcher(\n"
        "        choice == 1, choice == 2, 0, choice == 3);\n"
        "    assert(launch(0, nullptr, false, 0, nullptr, nullptr, nullptr,\n"
        "                  nullptr, nullptr, 0, false, 0, nullptr, nullptr,\n"
        "                  nullptr, nullptr, nullptr, nullptr) == cudaErrorNotSupported);\n"
        "  }\n"
        "}\n"
    )
    executable = tmp_path / "registry"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{tmp_path}",
            f"-I{root / 'src'}",
            str(driver),
            str(root / "src/scf/aot_shell_registry_stub.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


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
