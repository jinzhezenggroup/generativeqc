"""Execute actual native source-control parsing without loading a CUDA runtime."""

import os
import subprocess
from pathlib import Path

import pytest
from test_coulomb_optional_allocation import ROOT, compile_cached_probe


@pytest.fixture(scope="module")
def policy_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Retain legacy joint aliases and explicit independent consumer selections."""
    source = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    begin = source.index("unsigned direct_coulomb_reachable_mode()")
    end = source.index("bool resident_ppps_bra_requested()", begin)
    folder = tmp_path_factory.mktemp("cartesian-source-policy")
    cpp, binary = folder / "probe.cpp", folder / "probe"
    cpp.write_text(
        "#include <cstdlib>\n#include <cstring>\n#include <iostream>\n"
        + source[begin:end]
        + "int main() { std::cout << direct_coulomb_reachable_mode() << ' ' "
        "<< direct_hermite_convolution_mode(); }\n"
    )
    compile_cached_probe(cpp, binary)
    return binary


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (0, 0)),
        ("", (0, 0)),
        ("0", (0, 0)),
        ("none", (0, 0)),
        ("invalid", (0, 0)),
        ("values", (1, 1)),
        ("forces", (2, 2)),
        ("all", (3, 3)),
        ("1", (3, 3)),
        ("reachable", (3, 0)),
    ],
)
def test_actual_source_policy_parser(
    policy_probe: Path, value: str | None, expected: tuple[int, int]
) -> None:
    env = dict(os.environ)
    names = (
        "GENERATIVEQC_DIRECT_COULOMB_REACHABLE",
        "GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION",
    )
    for name in names:
        env.pop(name, None)
        if value is not None:
            env[name] = value
    result = subprocess.run(
        [str(policy_probe)], env=env, check=True, capture_output=True, text=True
    )
    assert tuple(map(int, result.stdout.split())) == expected
