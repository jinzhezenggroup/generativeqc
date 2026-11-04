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
