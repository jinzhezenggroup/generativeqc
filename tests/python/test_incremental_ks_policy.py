"""Exercise the shared HF/KS selector itself, without a runtime or GPU."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def selector(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the production header rather than reproducing its parser."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("incremental-selector")
    source = directory / "selector.cpp"
    source.write_text(
        '#include <iostream>\n#include "methods/incremental_direct_jk.hpp"\n'
        "int main() {\n"
        "  using namespace generativeqc::methods::detail;\n"
        "  try {\n"
        "    const bool enabled = incremental_direct_jk_benchmark_requested();\n"
        "    const auto interval = incremental_direct_jk_benchmark_rebuild_interval();\n"
        '    std::cout << enabled << ":" << (interval ? std::to_string(*interval) : "unset");\n'
        "    return 0;\n"
        "  } catch (const generativeqc::methods::MethodError&) { return 2; }\n"
        "}\n"
    )
    executable = directory / "selector"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize(
    ("enabled", "interval", "expected"),
    [
        (None, None, "0:unset"),
        ("0", None, "0:unset"),
        ("off", None, "0:unset"),
        ("1", None, "1:unset"),
        ("on", "8", "1:8"),
        ("1", "0", "1:0"),
        ("1", "4294967295", "1:4294967295"),
        ("true", None, None),
        ("", None, None),
        ("1", "-1", None),
        ("1", "1.0", None),
        ("1", "4294967296", None),
        ("1", "8junk", None),
        ("1", "", None),
    ],
)
def test_shared_selector(
    selector: Path, enabled: str | None, interval: str | None, expected: str | None
) -> None:
    environment = dict(os.environ)
    for key, value in (
        ("GENERATIVEQC_INCREMENTAL_DIRECT_JK", enabled),
        ("GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL", interval),
    ):
        environment.pop(key, None)
        if value is not None:
            environment[key] = value
    result = subprocess.run(
        [str(selector)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (0 if expected is not None else 2)
    if expected is not None:
        assert result.stdout == expected
