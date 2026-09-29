"""The standalone DFT target must own every generated header it consumes."""

import re
from pathlib import Path


def test_standalone_dft_tracks_density_factor_codegen() -> None:
    root = Path(__file__).resolve().parents[2]
    declarations = (root / "cmake/GenerativeQCTests.cmake").read_text()
    groups = re.findall(
        r"add_dependencies\(generativeqc_dft_tests\s+([^)]*)\)", declarations
    )
    dependencies = {name for group in groups for name in group.split()}
    assert {
        "generativeqc_xc_cpu_codegen",
        "generativeqc_scf_array_cpu_codegen",
    } <= dependencies
