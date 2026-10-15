"""Retire benchmark identities without promoting an unqualified default."""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_packed_response_auto_has_no_benchmark_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    assert "NVIDIA GeForce RTX 5090" not in source
    assert "n == 768 && a == 768" not in source
    assert "borrowed->occupied_factors[0].rank == 160" not in source
    assert "df_packed_response_preferred(" not in source
    assert "packed_default" not in source


def test_packed_response_requires_explicit_opt_in_and_ownership(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Execute actual selection across rank, ownership and fallback boundaries."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    selection = []
    for name, kind in (("packed_pairs", "bool"), ("derivative_pairs", "auto")):
        match = re.search(rf"const {kind} {name}\s*=[^;]+;", source)
        assert match is not None
        selection.append(match.group())
    unit = (
        r"""
#include <iostream>
#include <string_view>
struct Metric { bool full_rank; };
struct Borrowed { bool occupied_response; };
enum class DfDerivativePairs { full, symmetric, packed };
int main() {
  unsigned cases=0;
  for (unsigned metric=0;metric<3;++metric)
    for (const std::string_view pair_policy : {"auto","full","symmetric","packed"})
      for (bool shell_execution : {false,true})
        for (bool full_shell_domain : {false,true})
          for (unsigned owner=0;owner<3;++owner)
            for (bool promoted_default : {false,true}) {
              Metric metric_value{metric==1};
              const Metric* device_metric=metric ? &metric_value : nullptr;
              Borrowed borrowed_value{owner==2};
              const Borrowed* borrowed=owner ? &borrowed_value : nullptr;
"""
        + "\n".join(selection)
        + r"""
              const bool expected_packed=pair_policy=="packed" && metric!=2 &&
                  shell_execution && full_shell_domain && owner==2;
              const auto expected_pairs=expected_packed ? DfDerivativePairs::packed
                  : (pair_policy=="symmetric" || pair_policy=="packed" ||
                     (pair_policy=="auto" && promoted_default))
                        ? DfDerivativePairs::symmetric : DfDerivativePairs::full;
              if (packed_pairs!=expected_packed || derivative_pairs!=expected_pairs) {
                std::cerr << "selection mismatch: " << metric << ' ' << pair_policy
                    << ' ' << shell_execution << ' ' << full_shell_domain << ' '
                    << owner << ' ' << promoted_default << '\n';
                return 1;
              }
              ++cases;
            }
  std::cout << cases << " packed selection cases passed\n";
}
"""
    )
    probe, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    probe.write_text(unit)
    native_cxx.build_executable([probe], executable, compile_args=("-std=c++17",))
    subprocess.run([str(executable)], check=True, timeout=10)
