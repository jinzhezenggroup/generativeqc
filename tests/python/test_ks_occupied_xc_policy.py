"""Execute the native occupied-XC selector without requiring CUDA work."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def selector_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile the actual owner initializer, not a second policy implementation."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    beginning = source.index("const bool occupied_xc_requested =")
    ending = source.index("}();", beginning) + len("}();")
    folder = tmp_path_factory.mktemp("ks-occupied-selector")
    driver, binary = folder / "selector.cpp", folder / "selector"
    driver.write_text(
        "#include <cstdlib>\n#include <string_view>\nstruct Selector {\n"
        + source[beginning:ending]
        + "\n};\n"
        + "int main(int argument_count, char** arguments) {\n"
        + '  const bool expected = argument_count == 2 && std::string_view(arguments[1]) == "1";\n'
        + "  return Selector{}.occupied_xc_requested == expected ? 0 : 1;\n}\n"
    )
    native_cxx.build_executable([driver], binary, compile_args=["-std=c++20"])
    return binary


@pytest.mark.parametrize("selector", (None, "1", "0", "", "01", "10", "1 ", "true"))
def test_unset_is_automatic_and_explicit_controls_fail_closed(
    selector_probe: Path, selector: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset/1 enable admission, but cannot bypass its existing runtime gates."""
    if selector is None:
        monkeypatch.delenv("GENERATIVEQC_CUDA_KS_OCCUPIED_XC", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_CUDA_KS_OCCUPIED_XC", selector)
    subprocess.run(
        [str(selector_probe), "1" if selector in (None, "1") else "0"],
        check=True,
        timeout=10,
    )


def test_fitted_canonical_probe_excludes_provisional_xc(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """A physical final-state probe must wait for the stored-D target stage."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    beginning = source.index("const bool fitted_final_probe =")
    ending = source.index("// On a soft orbital manifold", beginning)
    driver, binary = tmp_path / "final_probe.cpp", tmp_path / "final_probe"
    driver.write_text(
        "#include <algorithm>\n#include <vector>\n"
        "struct Atom { int ecp_core{}; };\n"
        "struct System { std::vector<int> ecp_terms; std::vector<Atom> atoms{{}}; };\n"
        "struct Provider { System value; const System& system() const { return value; } };\n"
        "struct Schedule { bool lower{}; bool any_lower_precision() const { return lower; } };\n"
        "struct Probe {\n"
        "  bool occupied_stage{}, fitted_exchange{true}, budget_fock{}, has_range_correction{}, nonlocal_correlation{};\n"
        "  unsigned chunk_width{1}; Provider provider; Schedule precision_schedule;\n"
        "  unsigned configured_chunk_width() const { return chunk_width; }\n"
        "  bool eligible() const {\n"
        + source[beginning:ending]
        + "return fitted_final_probe; }\n};\n"
        "int main() {\n"
        "  Probe target; if (!target.eligible()) return 1;\n"
        "  Probe provisional; provisional.occupied_stage = true;\n"
        "  if (provisional.eligible()) return 2;\n"
        "  provisional.occupied_stage = false; if (!provisional.eligible()) return 3;\n"
        "  for (int gate = 0; gate < 8; ++gate) {\n"
        "    Probe excluded;\n"
        "    switch (gate) {\n"
        "      case 0: excluded.fitted_exchange = false; break;\n"
        "      case 1: excluded.budget_fock = true; break;\n"
        "      case 2: excluded.has_range_correction = true; break;\n"
        "      case 3: excluded.nonlocal_correlation = true; break;\n"
        "      case 4: excluded.precision_schedule.lower = true; break;\n"
        "      case 5: excluded.chunk_width = 2; break;\n"
        "      case 6: excluded.provider.value.ecp_terms.push_back(1); break;\n"
        "      case 7: excluded.provider.value.atoms[0].ecp_core = 2; break;\n"
        "    }\n"
        "    if (excluded.eligible()) return 4;\n"
        "  }\nreturn 0;\n}\n"
    )
    native_cxx.build_executable([driver], binary, compile_args=["-std=c++20"])
    subprocess.run([str(binary)], check=True, timeout=10)
