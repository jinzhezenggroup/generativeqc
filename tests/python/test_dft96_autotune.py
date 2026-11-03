"""Exercise resident qualification controls without loading a CUDA library."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.dft96_autotune import compare, reference_gate, run

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self


class Samples:
    """Return explicit synthetic endpoint observations for protocol tests only."""

    def __init__(
        self,
        side: str,
        order: list[tuple[str, int]],
        mutate: Callable[[dict], None] | None = None,
    ) -> None:
        self.side = side
        self.order = order
        self.mutate = mutate
        self.identity = {"device": "test-only"}

    def sample(self, pair_id: int) -> dict:
        self.order.append((self.side, pair_id))
        sample = {
            "seconds": 2.0 if self.side == "baseline" else 1.0,
            "iterations": [1],
            "energies": [-1.0],
            "forces": [[[0.0, 0.0, 0.0]]],
            "converged": True,
            "backend": ["cuda"],
            "scientific_identity": "test-scientific-identity",
            "endpoint_kind": "energy_force",
            "synchronized": True,
            "pair_id": pair_id,
            "interleaved": True,
            "schedule_identity": self.side,
            "source_hash": "test-source",
            "native_xc_schedule": {"tile": 256},
            "dft_work": [
                {
                    "grid_points": 100,
                    "fock_builds": 1,
                    "coulomb_exchange_work": {
                        "complete": True,
                        "coulomb_applications": 1,
                        "exchange_applications": 1,
                    },
                }
            ],
            "resources": {
                "peak_bytes": 1,
                "admission": {"limit_bytes": 10, "combined_peak_bytes": 2},
                "rejected_allocations": 0,
            },
        }
        if self.mutate:
            self.mutate(sample)
        return sample


REFERENCE = {
    "independent_reference": "synthetic unit-test reference, not GPU evidence",
    "energies": [-1.0],
    "forces": [[[0.0, 0.0, 0.0]]],
}


@pytest.mark.parametrize(
    "residual", [None, True, -1.0, float("nan"), float("inf"), 2e-10]
)
def test_reference_gate_uses_a_finite_measured_strict_residual(
    residual: object,
) -> None:
    assert not reference_gate({"converged": True, "orbital_gradient_norm": residual})[
        "passed"
    ]


def test_reference_gate_does_not_trust_the_solver_tolerance_alone() -> None:
    reference = {
        "converged": True,
        "conv_tol_grad": 1e-9,
        "orbital_gradient_norm": 2e-11,
    }
    assert reference_gate(reference)["passed"]
    reference["converged"] = False
    assert not reference_gate(reference)["passed"]


def qualify(
    *,
    mutate_baseline: Callable[[dict], None] | None = None,
    mutate_candidate: Callable[[dict], None] | None = None,
    same_xc: bool = True,
) -> tuple[dict, list[tuple[str, int]]]:
    order = []
    baseline = Samples("baseline", order, mutate_baseline)
    candidate = Samples("candidate", order, mutate_candidate)
    evidence = compare(baseline, candidate, REFERENCE, repeats=5, same_xc=same_xc)
    return evidence, order


def test_qualification_alternates_complete_pairs_and_preserves_samples() -> None:
    evidence, order = qualify()
    assert evidence["passed"]
    for pair_id in range(5):
        expected = (
            ["baseline", "candidate"] if pair_id % 2 == 0 else ["candidate", "baseline"]
        )
        assert order[2 * pair_id : 2 * pair_id + 2] == [
            (side, pair_id) for side in expected
        ]
    assert [sample["pair_id"] for sample in evidence["endpoint"]["baseline"]] == list(
        range(5)
    )
    assert [sample["pair_id"] for sample in evidence["endpoint"]["candidate"]] == list(
        range(5)
    )


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_both_resource_owners_must_pass(side: str) -> None:
    def reject(sample: dict) -> None:
        sample["resources"]["rejected_allocations"] = 1

    evidence, _ = qualify(**{f"mutate_{side}": reject})
    assert not evidence["passed"]
    assert not evidence[f"{side}_resources"]["passed"]


@pytest.mark.parametrize("field,value", [("grid_points", 101), ("fock_builds", 2)])
def test_changed_semantic_work_rejects_a_faster_candidate(
    field: str, value: int
) -> None:
    def change_work(sample: dict) -> None:
        sample["dft_work"][0][field] = value

    evidence, _ = qualify(mutate_candidate=change_work)
    assert evidence["endpoint"]["passed"]
    assert not evidence["work"]["passed"]
    assert not evidence["passed"]


def test_missing_complete_census_rejects_candidate() -> None:
    def incomplete(sample: dict) -> None:
        sample["dft_work"][0]["coulomb_exchange_work"]["complete"] = False

    evidence, _ = qualify(mutate_candidate=incomplete)
    assert not evidence["work"]["passed"]


def test_only_xc_search_can_change_xc_schedule() -> None:
    def change_tile(sample: dict) -> None:
        sample["native_xc_schedule"] = {"tile": 512}

    rejected, _ = qualify(mutate_candidate=change_tile)
    accepted, _ = qualify(mutate_candidate=change_tile, same_xc=False)
    assert not rejected["passed"]
    assert accepted["passed"]


def test_independent_oracle_cannot_be_replaced_by_mutual_parity() -> None:
    def same_wrong_result(sample: dict) -> None:
        sample["energies"] = [-2.0]

    evidence, _ = qualify(
        mutate_baseline=same_wrong_result, mutate_candidate=same_wrong_result
    )
    assert evidence["endpoint"]["passed"]
    assert not evidence["numerical"]["passed"]
    assert not evidence["passed"]


@pytest.mark.parametrize("atoms,batch", [(95, 1), (96, 2)])
def test_other_workloads_fail_before_loading_native_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atoms: int, batch: int
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-only")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"workload": {"atoms": [None] * atoms, "batch": batch}}))
    with pytest.raises(ValueError, match="96-atom singleton"):
        run(plan, tmp_path / "result", repeats=5)


def test_scheduler_is_required_before_native_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm allocation"):
        run(tmp_path / "missing.json", tmp_path / "result", repeats=5)


def test_source_mismatch_rejects_before_native_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import autotune

    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-only")
    monkeypatch.setattr(autotune, "source_identity", lambda _: "different-source")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "workload": {
                    "atoms": [None] * 96,
                    "batch": 1,
                    "source_identity": "requested-source",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="source checkout differs"):
        run(plan, tmp_path / "result", repeats=5)


def test_explicit_configuration_only_runs_fresh_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import autotune
    from generativeqc.profiles import file_hash

    from benchmarks import dft96_autotune

    class Owner(Samples):
        def __init__(
            self, library: Path, plan: Path, configuration: dict, directory: Path
        ) -> None:
            self.configuration = configuration
            super().__init__(
                "baseline" if configuration["tile_points"] is None else "candidate", []
            )

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exception: object) -> None:
            pass

        def sample(self, pair_id: int) -> dict:
            sample = super().sample(pair_id)
            sample["forces"] = [[[0.0, 0.0, 0.0] for _ in range(96)]]
            sample["native_xc_schedule"] = {
                "tile": self.configuration["tile_points"] or 256
            }
            return sample

    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-only")
    monkeypatch.setattr(autotune, "source_identity", lambda _: "unit-test-source")
    monkeypatch.setattr(dft96_autotune, "Resident", Owner)
    library = tmp_path / "unit-test-library"
    library.write_bytes(b"never loaded; synthetic resident owner")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                **REFERENCE,
                "forces": [[[0.0, 0.0, 0.0] for _ in range(96)]],
                "converged": True,
                "orbital_gradient_norm": 2e-11,
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "workload": {
                    "atoms": [None] * 96,
                    "batch": 1,
                    "source_identity": "unit-test-source",
                },
                "library": str(library),
                "library_hash": file_hash(library),
                "reference": str(reference),
            }
        )
    )
    configuration = {"tile_points": 1024, "environment": {}}
    report = run(
        plan, tmp_path / "result", repeats=5, requested_configuration=configuration
    )
    assert report["trials"] == []
    assert report["canonical_source_identity"] == "unit-test-source"
    assert report["identity"] == {"device": "test-only"}
    assert report["holdout"]["passed"]
    assert report["selected"] == configuration
