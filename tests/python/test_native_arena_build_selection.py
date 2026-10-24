"""Paired generation must select the same storage plan in every native output."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tools.generate_rccsd_native import cross_size_arena_selected

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "df_ccsd_native",
    "df_ccsd_core",
    "df_ccsd_hoisted",
    "df_ccsd_spectator_pairs",
    "df_cc_source",
    "df_occupied_triples",
    "rhf_frame_response",
    "df_lambda",
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("enabled", (False, True))
def test_cli_keeps_all_capacity_and_pointer_outputs_in_one_scope(
    family: str, enabled: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = importlib.import_module("tools.generate_" + family)
    selections = []

    def emit() -> str:
        selections.append(cross_size_arena_selected())
        return "paired-storage-selection"

    for name in ("header", "cpu_header", "cuda_header", "cuda_source"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, emit)
    arguments = [module.__file__, "--output-dir", str(tmp_path)]
    if enabled:
        arguments.append("--cross-size-arena")
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    assert selections == [enabled] * 3
    assert len(list(tmp_path.iterdir())) == 3
    assert not cross_size_arena_selected()


def test_cmake_selects_every_shared_native_emitter() -> None:
    source = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    declarations = source.split("generativeqc_register_generated_sources(")[1:]
    families = {"generate_" + family + ".py" for family in FAMILIES}
    families.add("generate_rccsd_native.py")
    matched = 0
    for declaration in declarations:
        if any(name in declaration for name in families) or (
            "generate_df_ccsd_${_df_family}.py" in declaration
        ):
            assert "ARGS ${_generativeqc_native_arena_args}" in declaration
            matched += 1
    assert matched == 10


@pytest.mark.parametrize("enabled", (False, True))
def test_rccsd_cli_keeps_multi_output_selection_paired(
    enabled: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mixed CPU/CUDA and triples outputs cannot leak different arena selections."""
    module = importlib.import_module("tools.generate_rccsd_native")
    selections = []

    def emit() -> str:
        selections.append(cross_size_arena_selected())
        return "paired-storage-selection"

    arguments = [module.__file__]
    for selector, emitter in (
        ("cpu-header", "cpu_header"),
        ("cuda-source", "cuda_source"),
        ("triples-cuda-source", "triples_response_cuda_source"),
        ("triples-fock-cpu-header", "triples_fock_cpu_header"),
        ("triples-fock-cuda-source", "triples_fock_cuda_source"),
    ):
        monkeypatch.setattr(module, emitter, emit)
        arguments.extend(("--" + selector, str(tmp_path / selector)))
    if enabled:
        arguments.append("--cross-size-arena")
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    assert selections == [enabled] * 5
    assert len(list(tmp_path.iterdir())) == 5
    assert not cross_size_arena_selected()
