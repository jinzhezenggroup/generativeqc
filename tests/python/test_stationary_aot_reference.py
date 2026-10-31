"""AOT manifest references are independently generated once, not per profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generativeqc_compiler.common.provenance import canonical_hash

from tools import write_stationary_aot_manifest as manifest
from tools import write_stationary_aot_reference as reference

ROOT = Path(__file__).resolve().parents[2]


def _sample_reference(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(reference, "qualified_sp_requests", lambda: ("unused",))
    monkeypatch.setattr(
        reference, "emit_first_derivative_cuda", lambda requests: "independent sp"
    )
    monkeypatch.setattr(
        reference,
        "derivative_cuda_sources",
        lambda domain: tuple(
            (index, f"independent spd {index}")
            for index in range(reference.QUALIFIED_SPD_AOT_SHARDS)
        ),
    )
    return reference.build_expected_primitives()


def test_reference_is_independent_and_all_shards_are_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _sample_reference(monkeypatch)
    assert payload["schema"] == reference.REFERENCE_SCHEMA
    assert payload["sp"] == [canonical_hash("independent sp")]
    assert payload["spd"] == [
        canonical_hash(f"independent spd {index}")
        for index in range(reference.QUALIFIED_SPD_AOT_SHARDS)
    ]
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(payload))
    assert manifest._expected_primitive_hashes(path, "sp") == tuple(payload["sp"])
    assert manifest._expected_primitive_hashes(path, "spd") == tuple(payload["spd"])


@pytest.mark.parametrize("damage", ["schema", "shard", "digest", "extra"])
def test_reference_fails_closed_on_missing_or_tampered_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    payload = _sample_reference(monkeypatch)
    if damage == "schema":
        payload["schema"] = "obsolete"
    elif damage == "shard":
        payload["spd"] = payload["spd"][:-1]
    elif damage == "digest":
        payload["sp"] = ["not-a-hash"]
    else:
        payload["unrecognized"] = "untrusted"
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference"):
        manifest._expected_primitive_hashes(path, "spd")


def test_manifest_uses_independent_reference_without_relowering_per_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _sample_reference(monkeypatch)
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(payload))

    def forbid_regeneration(domain: object) -> None:
        pytest.fail("primitive lowering must run once in the reference target")

    monkeypatch.setattr(manifest, "derivative_cuda_sources", forbid_regeneration)
    monkeypatch.setattr(
        manifest,
        "emit_stationary_profile_component_aot_wrapper_cuda",
        lambda profile, iterations: "independent wrapper",
    )
    wrapper, source_hashes = manifest._expected_component_sources(
        None, spin=None, profile="pbe_rks", iterations=3, reference=path
    )
    assert wrapper == "independent wrapper"
    assert source_hashes == tuple(payload["spd"])
    altered_hashes = tuple("0" * 64 for _ in source_hashes)
    assert canonical_hash(
        {"wrapper": canonical_hash(wrapper), "primitives": source_hashes}
    ) != canonical_hash(
        {"wrapper": canonical_hash(wrapper), "primitives": altered_hashes}
    )


def test_cmake_recomputes_reference_once_and_validates_every_manifest() -> None:
    cmake = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    assert cmake.count("NAME generativeqc_stationary_expected_primitives_codegen") == 1
    assert cmake.count(
        '--expected-primitives "${_generativeqc_stationary_expected_primitives}"'
    ) == 2
    assert cmake.count(
        '          "${_generativeqc_stationary_expected_primitives}"'
    ) == 2
    assert "DEPENDS ${_generativeqc_stationary_contract_inputs}" in cmake
    identity = (ROOT / "cmake/GenerativeQCSourceIdentity.json").read_text()
    assert "tools/write_stationary_aot_reference.py" in identity


def test_wheel_reuses_semantic_sources_separately_from_compiler_objects() -> None:
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    assert workflow.count("name: Restore stationary AOT semantic sources") == 1
    assert "key: stationary-aot-source-v1-${{ hashFiles(" in workflow
    assert (
        "GENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE="
        "/host${{ runner.temp }}/stationary-aot-source-cache"
    ) in workflow
    assert "GENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE = { env =" in project
    assert "CCACHE_DIR=/host${{ runner.temp }}/ccache" in workflow
