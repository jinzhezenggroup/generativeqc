"""Retire benchmark identities without promoting an unqualified default."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner


def test_packed_response_auto_has_no_benchmark_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    assert "NVIDIA GeForce RTX 5090" not in source
    assert "n == 768 && a == 768" not in source
    assert "borrowed->occupied_factors[0].rank == 160" not in source
    assert "df_packed_response_preferred(" not in source
    assert "packed_default" not in source


def test_packed_response_requires_explicit_opt_in_and_ownership(tmp_path: Path) -> None:
    """Execute the production selectors across ownership and fallback states."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    declarations = []
    for declaration in ("const bool packed_pairs", "const auto derivative_pairs"):
        match = re.search(re.escape(declaration) + r"\s*=([^;]+);", source)
        assert match is not None
        declarations.append(match.group(0))
    probe = tmp_path / "packed_policy.cpp"
    probe.write_text(
        r"""
#include <array>
#include <string_view>
enum class DfDerivativePairs { full, symmetric, packed };
struct Borrowed { bool occupied_response; };
int main() {
  for (std::string_view pair_policy : {"auto", "full", "symmetric", "packed"})
    for (unsigned bits = 0; bits < 64; ++bits) {
      const bool shell_execution = bits & 1, full_shell_domain = bits & 2;
      const bool auxiliary_g = bits & 4, promoted_default = bits & 8;
      Borrowed value{bool(bits & 16)};
      const auto* borrowed = bits & 32 ? &value : nullptr;
"""
        + "\n".join(declarations)
        + r"""
      const bool eligible = shell_execution && full_shell_domain && !auxiliary_g &&
                            borrowed && borrowed->occupied_response;
      if (packed_pairs != (pair_policy == "packed" && eligible)) return 1;
      auto expected = DfDerivativePairs::full;
      if (pair_policy == "packed")
        expected = eligible ? DfDerivativePairs::packed : DfDerivativePairs::symmetric;
      else if (pair_policy == "symmetric" || (pair_policy == "auto" && promoted_default))
        expected = DfDerivativePairs::symmetric;
      if (derivative_pairs != expected) return 2;
    }
}
"""
    )
    executable = tmp_path / "packed_policy"
    compile_owner(compiler, tmp_path, [probe], executable)
    subprocess.run([str(executable)], check=True, timeout=30)
