"""Ownership guards for Direct-Fock runtime adapters."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_fock_order2_uses_generated_science_and_shared_screening() -> None:
    source = (ROOT / "src/scf/cuda/direct_fock_order2.cuh").read_text(encoding="utf-8")
    assert "generated_weighted_eri::canonicalize_direct_shell_slots" in source
    assert "contracted_eri_cartesian_source_order2_shell" in source
    assert "accumulate_direct_fock_integral" in source
    assert "direct_ao_quartet_survives_schwarz" in source
    assert "first_pair_class < second_pair_class" not in source
    assert "schwarz_bounds[physical_offset + matrix_index" not in source


def test_fock_quartet_uses_generated_science_and_shared_screening() -> None:
    source = (ROOT / "src/scf/cuda/direct_fock_quartet.cuh").read_text(encoding="utf-8")
    assert "dispatch_contracted_eri_cartesian_source_shell_class" in source
    assert "accumulate_direct_fock_integral" in source
    assert "direct_ao_quartet_survives_schwarz" in source
    assert "schwarz_bounds[physical_offset + matrix_index" not in source


def test_fock_adapters_are_runtime_owned() -> None:
    for name in ("direct_fock_order2.cuh", "direct_fock_quartet.cuh"):
        ownership = json.loads(
            (
                ROOT / "docs/cuda_ownership/files/src/scf/cuda" / f"{name}.json"
            ).read_text(encoding="utf-8")
        )
        assert ownership["role"] == "runtime"

    retirement = json.loads(
        (ROOT / "docs/cuda_ownership/direct_hf_retirement.json").read_text(
            encoding="utf-8"
        )
    )
    family = next(
        item for item in retirement["families"] if item["id"] == "native-fock-and-jk"
    )
    assert "src/scf/cuda/direct_fock_order2.cuh" not in family["files"]
    assert "src/scf/cuda/direct_fock_quartet.cuh" not in family["files"]
