"""Public uint64 byte diagnostics need not share the platform size_t typedef."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_df_source_device_capacity_accepts_distinct_unsigned_types(
    tmp_path: Path, diagnostic_type: str, native_cxx: object
) -> None:
    """Compile the DF owner assignment with both LP64 uint64_t conventions."""
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    statement = re.search(
        r"metrics\.owned_device_bytes\s*=\s*std::max.*?;", source, re.DOTALL
    )
    assert statement is not None
    path, executable = tmp_path / "df_capacity.cpp", tmp_path / "df_capacity"
    path.write_text(f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
int main() {{
  struct {{ {diagnostic_type} owned_device_bytes; }} metrics{{}};
  struct {{ std::size_t device_capacity_bytes; }} fitted{{}};
  for (std::uint64_t old : {{0ULL, 0x100000030ULL}})
    for (std::size_t next : {{std::size_t(0), std::size_t(0x100000050ULL)}}) {{
      metrics.owned_device_bytes=old;
      fitted.device_capacity_bytes=next;
      {statement.group(0)}
      if (metrics.owned_device_bytes != (old > next ? old : next)) return 1;
    }}
}}
""")
    native_cxx.build_executable(
        [path],
        executable,
        compile_args=("-std=c++20", "-Wall", "-Werror"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_triples_capacity_maximum_accepts_distinct_unsigned_types(
    tmp_path: Path, diagnostic_type: str, native_cxx: object
) -> None:
    source = (ROOT / "src/methods/rccsdt_method.cpp").read_text()
    statement = re.search(
        r"diagnostic\.numeric_capacity_bytes\s*=\s*std::max.*?;", source, re.DOTALL
    )
    assert statement is not None
    harness = f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
std::size_t checked_add(std::size_t a, std::size_t b) {{
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::length_error("overflow");
  return a+b;
}}
int main() {{
  struct {{ {diagnostic_type} numeric_capacity_bytes; }} diagnostic{{}};
  auto update_capacity = [&](std::size_t retained, std::size_t triples_workspace_bytes,
                             std::size_t warm_reservation,
                             std::size_t plan_reservation = 0) {{
    struct {{ std::size_t external_reservation_bytes, reference_execution_plan_bytes; }}
        state{{warm_reservation, plan_reservation}};
    {statement.group(0)}
  }};
  for (std::uint64_t old : {{0ULL, 0x100000010ULL}})
    for (std::size_t retained : {{std::size_t(0), std::size_t(0x100000020ULL)}})
      for (std::size_t warm : {{std::size_t(0), std::size_t(0x100000030ULL)}})
        for (std::size_t plan : {{std::size_t(0), std::size_t(0x100000040ULL)}}) {{
        diagnostic.numeric_capacity_bytes=old;
        update_capacity(retained, 31, warm, plan);
        const auto phase_capacity=retained+31+warm+plan;
        const auto expected=old > phase_capacity ? old : phase_capacity;
        if (diagnostic.numeric_capacity_bytes!=expected) return 1;
      }}
  const auto limit=std::numeric_limits<std::size_t>::max();
  for (bool overflow_in_warm_add : {{false, true}}) {{
    diagnostic.numeric_capacity_bytes=17;
    try {{
      update_capacity(overflow_in_warm_add ? limit-31 : limit, 31,
                      overflow_in_warm_add ? 1 : 0);
      return 2;
    }} catch (const std::length_error&) {{
      if (diagnostic.numeric_capacity_bytes!=17) return 3;
    }}
  }}
}}
"""
    path, executable = tmp_path / "capacity.cpp", tmp_path / "capacity"
    path.write_text(harness)
    native_cxx.build_executable(
        [path],
        executable,
        compile_args=("-std=c++20", "-Wall", "-Werror"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_mp2_endpoint_maximum_accepts_distinct_unsigned_types(
    tmp_path: Path, diagnostic_type: str, native_cxx: object
) -> None:
    source = (ROOT / "src/methods/mp2_method.cpp").read_text()
    statement = re.search(
        r"last_->planned_endpoint_peak_bytes\s*=\s*std::max.*?;", source, re.DOTALL
    )
    assert statement is not None
    path, executable = tmp_path / "mp2_capacity.cpp", tmp_path / "mp2_capacity"
    path.write_text(f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "posthf/capacity.hpp"
namespace posthf = generativeqc::posthf;
int main() {{
  struct {{ {diagnostic_type} numeric_capacity_bytes, planned_endpoint_peak_bytes; }} diagnostic{{}};
  auto* last_ = &diagnostic;
  struct Force {{ std::size_t planned_endpoint_peak_bytes; }} force{{}};
  auto* force_diagnostic = &force;
  struct Plan {{
    std::size_t bytes{{}};
    std::size_t retained_numeric_bytes() const {{ return bytes; }}
  }} cuda_reference_plan_;
  const auto update = [&](std::size_t next, std::size_t warm_capacity, std::size_t retained,
                          std::size_t component_capacity = 0) {{
    force.planned_endpoint_peak_bytes = next;
    cuda_reference_plan_.bytes = retained;
    {statement.group(0)}
  }};
  for (std::uint64_t old : {{0ULL, 0x100000010ULL}})
    for (std::size_t next : {{std::size_t(0), std::size_t(0x100000020ULL)}})
      for (std::size_t retained : {{std::size_t(0), std::size_t(0x100000030ULL)}})
        for (std::size_t component : {{std::size_t(0), std::size_t(0x100000040ULL)}}) {{
          diagnostic.numeric_capacity_bytes = old;
          update(next, 31, retained, component);
          const auto combined = next + 31 + retained + component;
          if (diagnostic.planned_endpoint_peak_bytes != (old > combined ? old : combined)) return 1;
        }}
  diagnostic.planned_endpoint_peak_bytes = 17;
  try {{
    update(std::numeric_limits<std::int64_t>::max(), 1, 0);
    return 2;
  }} catch (const std::overflow_error&) {{
    if (diagnostic.planned_endpoint_peak_bytes != 17) return 3;
  }}
}}
""")
    native_cxx.build_executable(
        [path],
        executable,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_df_mp2_companion_diagnostic_preserves_capacity_and_method_fields(
    tmp_path: Path, diagnostic_type: str, native_cxx: object
) -> None:
    """Exercise the actual companion publisher under both LP64 ABI spellings."""
    source = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    begin = source.index("generativeqc_correlation_diagnostic mp2_diagnostic(")
    end = source.index("\n}\n", begin) + 3
    function = source[begin:end]
    path, executable = (
        tmp_path / "df_mp2_diagnostic.cpp",
        tmp_path / "df_mp2_diagnostic",
    )
    path.write_text(f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
constexpr int GENERATIVEQC_ABI_VERSION=1;
struct generativeqc_correlation_diagnostic {{
  std::size_t struct_size{{}};
  int abi_version{{}};
  double reference_energy{{}}, reference_residual{{}}, opposite_spin_energy{{}}, same_spin_energy{{}};
  double minimum_absolute_denominator{{}};
  {diagnostic_type} numeric_capacity_bytes{{}};
  int mo_host_staging{{}}, ccsd_iterations{{}};
  double ccsd_t_triples_energy{{}};
  char equation_hash[65]{{}};
}};
namespace cc {{
struct DFMP2Result {{
  double opposite_spin{{}}, same_spin{{}}, minimum_absolute_denominator{{}};
  const char* equation_hash{{}};
  std::size_t numeric_capacity_bytes{{}};
}};
}}
{function}
int main() {{
  generativeqc_correlation_diagnostic reference;
  reference.reference_energy=-75;
  reference.reference_residual=1e-12;
  reference.ccsd_iterations=17;
  reference.ccsd_t_triples_energy=-0.1;
  cc::DFMP2Result mp2{{-0.2,-0.05,0.6,"independent-mp2-hash",0}};
  for (std::uint64_t old : {{0ULL,0x100000030ULL,0xffffffffffffffffULL}})
    for (std::size_t next : {{std::size_t(0),std::size_t(0x100000050ULL),
                            std::numeric_limits<std::size_t>::max()}}) {{
      reference.numeric_capacity_bytes=old;
      mp2.numeric_capacity_bytes=next;
      const auto result=mp2_diagnostic(reference,mp2);
      if (result.numeric_capacity_bytes!=(old>next ? old : next)) return 1;
      if (result.reference_energy!=-75 || result.reference_residual!=1e-12 ||
          result.opposite_spin_energy!=-0.2 || result.same_spin_energy!=-0.05 ||
          result.minimum_absolute_denominator!=0.6) return 2;
      if (result.ccsd_iterations || result.ccsd_t_triples_energy || !result.mo_host_staging ||
          result.struct_size!=sizeof(result) || result.abi_version!=GENERATIVEQC_ABI_VERSION) return 3;
      if (std::string_view(result.equation_hash)!="independent-mp2-hash") return 4;
    }}
}}
""")
    native_cxx.build_executable(
        [path],
        executable,
        compile_args=("-std=c++20", "-Wall", "-Wextra", "-Werror"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
