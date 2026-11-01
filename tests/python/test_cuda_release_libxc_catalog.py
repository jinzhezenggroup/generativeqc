"""Keep complete automatic Libxc CUDA packaging independent of the release GPU target."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRESETS = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
BY_NAME = {item["name"]: item for item in PRESETS["configurePresets"]}


def _cache_variables(name: str) -> dict[str, str]:
    """Resolve actual inherited CMake preset values, not just local overrides."""
    preset = BY_NAME[name]
    parents = preset.get("inherits", ())
    if isinstance(parents, str):
        parents = (parents,)
    result: dict[str, str] = {}
    for parent in reversed(parents):
        result.update(_cache_variables(parent))
    result.update(preset.get("cacheVariables", {}))
    return result


def test_all_release_cuda_presets_package_the_complete_libxc_catalog() -> None:
    for name in (
        "cuda-release",
        "cuda-release-sm120",
        "cuda-release-cli",
        "cuda-release-cli-sm120",
    ):
        options = _cache_variables(name)
        assert options["GENERATIVEQC_ENABLE_CUDA"] == "ON", name
        assert options["GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT"] == "ON", name
        # Empty means the generated full structural inventory, not a static list.
        assert options["GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS"] == "", name


def test_portable_release_cli_does_not_inherit_sm120() -> None:
    portable = _cache_variables("cuda-release-cli")
    assert "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES" not in portable
    assert portable["GENERATIVEQC_AOT_PROFILE"] == "portable"
    assert (
        _cache_variables("cuda-release-cli-sm120")[
            "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES"
        ]
        == "120"
    )
    # Ordinary development builds should not incur release AOT costs.
    assert "GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT" not in _cache_variables(
        "cuda-dev-fast"
    )
    builds = {item["name"] for item in PRESETS["buildPresets"]}
    assert {"cuda-release-cli", "cuda-release-cli-sm120"} <= builds


def test_wheel_build_enables_full_catalog_for_any_cuda_architecture() -> None:
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    wheel_step = workflow.split("      - name: Build and test wheel\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    # scikit-build-core turns this environment setting into a CMake -D flag.
    assert re.search(
        r"(?m)^\s+SKBUILD_CMAKE_DEFINE="
        r"GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON\s*$",
        wheel_step,
    )
    assert "GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS=" not in wheel_step
    assert "GENERATIVEQC_CUDA_ARCHITECTURES=" in wheel_step
