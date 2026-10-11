"""Protect the generated Fock sink and compensated reference routing contract."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.production_emission import (
    _streaming_fock_internal_signature,
    emit_profile_shard,
)
from generativeqc_compiler.integral.production_k_block import direct_k_block_candidates
from generativeqc_compiler.integral.production_profile import (
    ResolvedProductionProfile,
    resolve_production_profile,
)
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_registry_header,
    emit_registry_source,
    shell_class_index,
)
from generativeqc_compiler.integral.production_rys_tasks import (
    direct_rys_task_candidates,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def profile() -> ResolvedProductionProfile:
    """Use the actual production schedules rather than a synthetic allowlist."""
    return resolve_production_profile(
        ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )


@pytest.mark.parametrize("name", ("ssss", "ppps", "dpps", "dddd", "ddds"))
def test_generated_schedules_keep_both_fock_planes(
    profile: ResolvedProductionProfile, name: str
) -> None:
    """Packed, component, subgroup and high-order workers share the sink ABI."""
    selection = next(item for item in profile.selections if item.spec.name == name)
    source = emit_profile_shard(profile, (selection,))
    assert '#include "runtime/compensated_atomic.cuh"' in source
    assert source.count('#include "runtime/compensated_atomic.cuh"') == 1
    assert source.index('#include "runtime/compensated_atomic.cuh"') < source.index(
        "namespace generativeqc::scf::generated::profile_"
    )
    assert "generativeqc::runtime::CompensatedOutput fock" in source
    assert "double* fock" not in source
    assert "typename Output = double*" in source
    signature = _streaming_fock_internal_signature(selection)
    assert "generativeqc::runtime::CompensatedOutput fock" in signature.parameter_list()


def test_generated_registry_distinguishes_force_and_fock_sinks(
    profile: ResolvedProductionProfile,
) -> None:
    """Force output remains a plain pointer; Fock forwards its two-plane value."""
    source = emit_multi_registry_source((profile,))
    assert "using FockLaunchFunction" in source
    assert "FockLaunchFunction launch_fock" in source
    assert "FockLaunchFunction launch_mixed_fock" in source
    assert "LaunchFunction launch_force" in source
    assert "runtime::CompensatedOutput output" in source


def test_work_bucket_launchers_keep_the_compensated_abi(
    profile: ResolvedProductionProfile,
) -> None:
    """Optional upstream work schedules must not narrow the output to one plane."""
    source = emit_multi_registry_source((profile,))
    declarations = re.findall(
        r'extern "C" cudaError_t [^(]*_work_streaming_fock\([^;]*\);', source
    )
    assert declarations
    assert all("runtime::CompensatedOutput" in item for item in declarations)
    header = (ROOT / "src/scf/aot_shell_registry.hpp").read_text()
    declaration = re.search(
        r"cudaError_t launch_shell_class_work_streaming_fock\((.*?)\) noexcept;",
        header,
        re.DOTALL,
    )
    assert declaration is not None
    assert "runtime::CompensatedOutput fock" in declaration.group(1)


def test_compensated_reference_does_not_skip_generated_classes() -> None:
    """Pages, retries and streams share one plane; only uncovered classes recur."""
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    assert source.count("{quartet_fock, resources.reference_fock_correction_}") == 3
    assert "resources.reference_fock_correction_ ? 0U" not in source
    assert (
        "if (resources.reference_fock_correction_)\n"
        "      return launch_bounded_generic_fock"
    ) not in source
    assert "cudaMemsetAsync(resources.reference_fock_correction_, 0" in source
    assert "launch_jk_compensation_fold(resources.stream_, quartet_fock" in source


@pytest.mark.parametrize("name", ("dsss", "dsds"))
def test_rys_task_workers_keep_both_fock_planes(
    profile: ResolvedProductionProfile, name: str
) -> None:
    """Lane-local values and both bounded queue wrappers retain the same sink."""
    selection = next(
        item for item in direct_rys_task_candidates(profile) if item.spec.name == name
    )
    source = emit_profile_shard(profile, (selection,), variant="_rys_task")
    assert "double* fock" not in source
    for suffix in ("_streaming_fock", "_work_streaming_fock"):
        declaration = re.search(
            rf'extern "C" cudaError_t \w+{suffix}\((.*?)\) \{{',
            source,
            re.DOTALL,
        )
        assert declaration is not None
        assert "runtime::CompensatedOutput fock" in declaration.group(1)


@pytest.mark.parametrize("generated", (False, True))
def test_native_streaming_dispatch_preserves_fock_planes(
    tmp_path: Path,
    native_cxx: NativeCxx,
    profile: ResolvedProductionProfile,
    generated: bool,
) -> None:
    """Execute the real host selector, registry and emitted wrapper signatures.

    Only device launch bodies and CUDA device discovery are replaced. This
    exercises the native disabled stubs or generated registry in separate
    translation units, including raw-pointer callers and UHF block rollback.
    """
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\n"
        "using cudaStream_t = void*;\n"
        "enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, "
        "cudaErrorNotSupported = 801 };\n"
        "enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor, "
        "cudaDevAttrComputeCapabilityMinor };\n"
        "inline cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }\n"
        "inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int) {\n"
        "  *value = attr == cudaDevAttrComputeCapabilityMajor ? 12 : 0;\n"
        "  return cudaSuccess;\n}\n"
    )
    selection = next(item for item in profile.selections if item.spec.name == "dsss")
    # Retain dsss qualification without claiming classes omitted by this fixture.
    assert profile.preferred_rys_task_fock_shell_classes is not None
    reduced = replace(
        profile,
        selections=(selection,),
        preferred_rys_task_fock_shell_classes=tuple(
            name
            for name in profile.preferred_rys_task_fock_shell_classes
            if name == selection.spec.name
        ),
    )
    assert reduced.preferred_rys_task_fock_shell_classes == ("dsss",)
    sources = [ROOT / "src/scf/aot_shell_registry_stub.cpp"]
    if generated:
        (tmp_path / "generativeqc_generated_shell_registry.hpp").write_text(
            emit_multi_registry_header((reduced,))
        )
        registry = tmp_path / "registry.cpp"
        registry.write_text(emit_multi_registry_source((reduced,)))
        emitted = emit_profile_shard(reduced, reduced.selections)
        for candidates, variant in (
            (direct_rys_value_candidates, "_rys_value"),
            (direct_k_block_candidates, "_k_block"),
            (direct_rys_task_candidates, "_rys_task"),
        ):
            emitted += emit_profile_shard(reduced, candidates(reduced), variant=variant)
        definitions = []
        for signature, symbol in re.findall(
            r'(extern "C" cudaError_t (\w+)\(.*?\)) \{', emitted, re.DOTALL
        ):
            body = (
                f'return observe_fock(fock, "{symbol}", unrestricted);'
                if " fock," in signature
                else "return cudaSuccess;"
            )
            definitions.append(f"{signature} {{ {body} }}")
        assert definitions
        wrappers = tmp_path / "wrappers.cpp"
        wrappers.write_text(
            '#include "scf/aot_shell_registry.hpp"\n'
            "cudaError_t observe_fock(generativeqc::runtime::CompensatedOutput, "
            "const char*, bool);\n" + "\n".join(definitions)
        )
        sources = [registry, wrappers]
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        f"""
#include "scf/cuda/direct_fock_lowering.hpp"
#include <cassert>
#include <cstring>
#include <type_traits>
using generativeqc::runtime::CompensatedOutput;
using namespace generativeqc::scf::generated;
using namespace generativeqc::scf::cuda_execution;
using Queue = generativeqc::scf::detail::GeneratedExchangeTaskSchedule;
using Launch = decltype(&launch_shell_class_streaming_fock);
static_assert(std::is_same_v<Launch, decltype(&launch_shell_class_rys_task_streaming_fock)>);
static_assert(std::is_same_v<Launch, decltype(&launch_shell_class_rys_task_work_streaming_fock)>);
double* expected_sum;
double* expected_correction;
const char* expected_symbol;
bool expected_unrestricted;
unsigned observed_calls;
cudaError_t observe_fock(CompensatedOutput fock, const char* symbol, bool unrestricted) {{
  assert(fock.sum == expected_sum && fock.correction == expected_correction);
  assert(std::strcmp(symbol, expected_symbol) == 0);
  assert(unrestricted == expected_unrestricted);
  ++observed_calls;
  *fock.sum += 2.0;
  if (fock.correction) *fock.correction += 0.25;
  return cudaSuccess;
}}
int main() {{
  constexpr unsigned cls = {shell_class_index(selection.spec)};
  constexpr auto bit = std::uint64_t{{1}} << cls;
  assert(preferred_rys_task_fock_shell_class_mask() == {"bit" if generated else "0"});
  const DirectExchangeSelection selections[]{{
      {{}}, {{0, 0, 0, Queue::Work}}, {{bit}}, {{0, bit}},
      {{0, 0, bit}}, {{0, 0, bit, Queue::Work}}}};
  const Launch launches[]{{
      launch_shell_class_streaming_fock, launch_shell_class_work_streaming_fock,
      launch_shell_class_rys_streaming_fock, launch_shell_class_k_block_streaming_fock,
      launch_shell_class_rys_task_streaming_fock,
      launch_shell_class_rys_task_work_streaming_fock}};
  const char* symbols[]{{
      "generativeqc_launch_sm120_generated_dsss_streaming_fock",
      "generativeqc_launch_sm120_generated_dsss_work_streaming_fock",
      "generativeqc_launch_sm120_rys_value_generated_dsss_streaming_fock",
      "generativeqc_launch_sm120_k_block_generated_dsss_streaming_fock",
      "generativeqc_launch_sm120_rys_task_generated_dsss_streaming_fock",
      "generativeqc_launch_sm120_rys_task_generated_dsss_work_streaming_fock"}};
  for (unsigned choice = 0; choice < 6; ++choice) {{
    for (bool unrestricted : {{false, true}}) {{
      const unsigned selected = unrestricted && choice == 3 ? 0 : choice;
      auto launch = direct_fock_streaming_launcher(selections[choice], cls, unrestricted);
      assert(launch == launches[selected]);
      expected_symbol = symbols[selected];
      expected_unrestricted = unrestricted;
      for (bool compensated : {{false, true}}) {{
        double sum = 3.0, correction = 0.5;
        expected_sum = &sum;
        expected_correction = compensated ? &correction : nullptr;
        auto invoke = [&](auto output) {{
          return launch(cls, nullptr, unrestricted, 1, nullptr, nullptr, nullptr,
                        nullptr, nullptr, 0, false, 0, nullptr, nullptr,
                        output, nullptr, nullptr, nullptr);
        }};
        auto error = compensated ? invoke(CompensatedOutput{{&sum, &correction}})
                                 : invoke(&sum);
        assert(error == {"cudaSuccess" if generated else "cudaErrorNotSupported"});
        assert(sum == {"5.0" if generated else "3.0"});
        assert(correction == ({str(generated).lower()} && compensated ? 0.75 : 0.5));
      }}
    }}
  }}
  assert(observed_calls == {24 if generated else 0});
}}
"""
    )
    executable = tmp_path / "dispatch"
    native_cxx.build_executable(
        (driver, *sources),
        executable,
        compile_args=("-std=c++17", f"-I{tmp_path}", f"-I{ROOT / 'src'}"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_legacy_registry_task_declarations_match_definitions(
    profile: ResolvedProductionProfile,
) -> None:
    """The legacy single-profile ABI must also expose both output planes."""
    for source in (
        emit_registry_header(profile.selections),
        emit_registry_source(profile.selections),
    ):
        for suffix in ("rys_task_streaming_fock", "rys_task_work_streaming_fock"):
            declaration = re.search(
                rf"cudaError_t launch_shell_class_{suffix}\((.*?)\) noexcept",
                source,
                re.DOTALL,
            )
            assert declaration is not None
            assert "runtime::CompensatedOutput fock" in declaration.group(1)
