"""Keep complete automatic Libxc CUDA packaging independent of the release GPU target."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from cibuildwheel.environment import parse_environment

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
        "cuda-release-cli",
    ):
        options = _cache_variables(name)
        assert options["GENERATIVEQC_ENABLE_CUDA"] == "ON", name
        assert options["GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT"] == "ON", name
        # Empty means the generated full structural inventory, not a static list.
        assert options["GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS"] == "", name


def test_release_presets_remain_architecture_neutral() -> None:
    assert set(BY_NAME) == {"cuda-dev-fast", "cuda-release", "cuda-release-cli"}
    for name in ("cuda-release", "cuda-release-cli"):
        options = _cache_variables(name)
        assert "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES" not in options
        assert "CMAKE_CUDA_ARCHITECTURES" not in options
        assert options["GENERATIVEQC_AOT_PROFILE"] == "auto"
    # Ordinary development builds should not incur release AOT costs.
    assert "GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT" not in _cache_variables(
        "cuda-dev-fast"
    )
    builds = {item["name"] for item in PRESETS["buildPresets"]}
    assert builds == set(BY_NAME)


def test_wheel_build_enables_full_catalog_for_any_cuda_architecture() -> None:
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    wheel_step = workflow.split("      - name: Build and test wheel\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    environment_block = wheel_step.split("          CIBW_ENVIRONMENT: >-\n", 1)[
        1
    ].split("\n        with:", 1)[0]
    # Actions resolves these before cibuildwheel sees the folded YAML scalar.
    environment = re.sub(r"\$\{\{.*?\}\}", "123", environment_block)
    parsed = parse_environment(" ".join(environment.splitlines())).as_dictionary({})
    assert parsed["SKBUILD_CMAKE_DEFINE"] == (
        "GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON"
    )
    assert "GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS" not in parsed
    assert parsed["GENERATIVEQC_CUDA_ARCHITECTURES"]


def test_cibuildwheel_rejects_unquoted_nested_assignment() -> None:
    # shlex accepts this, but the actual wheel parser rejects it before building.
    with pytest.raises(ValueError, match="Unsupported bash construct"):
        parse_environment(
            "SKBUILD_CMAKE_DEFINE=GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON"
        ).as_dictionary({})
