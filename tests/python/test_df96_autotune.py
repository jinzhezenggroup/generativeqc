"""DF96 benchmark protocol tests use synthetic evidence, never CUDA hardware."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc.profiles import canonical_hash

from benchmarks.df96_autotune import compare, run

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE = {
    "independent_reference": "synthetic fitted-HF reference for protocol tests only",
    "reference_family": "df-hf",
    "energies": [-1.0],
    "forces": [[[0.0, 0.0, 0.0]]],
    "metric": {"naux": 2},
}


@pytest.mark.parametrize("entrypoint", ["cli", "api"])
@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_retained_output_rejected_before_plan_or_native_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    destination_kind: str,
) -> None:
    """Neither entry point may overwrite reviewed reports or worker logs."""
    import sys

    from benchmarks import _retention, df96_autotune

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    originals = [retained / "report.json", retained / "worker.log"]
    for original in originals:
        original.write_text("preserved reviewed evidence")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"
    missing_plan = tmp_path / "must-not-read-plan.json"
    if entrypoint == "api":
        with pytest.raises(ValueError, match="raw benchmark output"):
            df96_autotune.run(missing_plan, destination, repeats=5, policies=[{}])
    else:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "df96_autotune",
                str(missing_plan),
                "--output",
                str(destination),
                "--policy",
                "{}",
            ],
        )
        with pytest.raises(SystemExit) as error:
            df96_autotune.main()
        assert error.value.code == 2
    assert all(
        original.read_text() == "preserved reviewed evidence" for original in originals
    )
    assert not (tmp_path / ".artifacts").exists()


class Samples:
    """Return detached, explicitly nonphysical observations for one test owner."""

    def __init__(self, side: str, order: list, mutation: Any = None) -> None:
        self.side, self.order, self.mutation = side, order, mutation
        self.identity = {"device": "synthetic-test-only"}

    def sample(self, pair_id: int) -> dict:
        self.order.append((self.side, pair_id))
        schedule = {"component": "df", "environment": {"test_only": self.side}}
        sample = {
            "seconds": 2.0 if self.side == "baseline" else 1.0,
            "iterations": [1],
            "energies": [-1.0],
            "forces": [[[0.0, 0.0, 0.0]]],
            "converged": True,
            "backend": ["cuda"],
            "scientific_identity": "synthetic-scientific-identity",
            "source_hash": "synthetic-source",
            "endpoint_kind": "energy_force",
            "synchronized": True,
            "pair_id": pair_id,
            "interleaved": True,
            "schedule": schedule,
            "schedule_identity": canonical_hash(schedule),
            "df_work": {
                "fock_builds": [1],
                "metrics": [
                    {"system_index": 0, "effective_rank": 2, "absolute_threshold": 1e-7}
                ],
            },
            "resources": {
                "peak_bytes": 1,
                "admission": {"limit_bytes": 10, "combined_peak_bytes": 2},
                "rejected_allocations": 0,
            },
        }
        if self.mutation:
            self.mutation(sample)
        return sample


def test_complete_df_comparison_retains_alternating_pairs() -> None:
    order: list = []
    evidence = compare(
        Samples("baseline", order),
        Samples("candidate", order),
        copy.deepcopy(REFERENCE),
        repeats=5,
    )
    assert evidence["passed"]
    assert evidence["endpoint"]["speedup"] == 2.0
    assert evidence["work"]["passed"]
    expected = []
    for pair_id in range(5):
        sides = (
            ("baseline", "candidate") if pair_id % 2 == 0 else ("candidate", "baseline")
        )
        expected.extend((side, pair_id) for side in sides)
    assert order == expected


@pytest.mark.parametrize(
    "defect", ["rank", "focks", "energy", "force", "capacity", "backend"]
)
def test_fast_df_candidate_cannot_override_failed_gates(defect: str) -> None:
    def mutate(sample: dict) -> None:
        if defect == "rank":
            sample["df_work"]["metrics"][0]["effective_rank"] = 1
        elif defect == "focks":
            sample["df_work"]["fock_builds"] = [2]
        elif defect == "energy":
            sample["energies"] = [-1.0 + 1e-7]
        elif defect == "force":
            sample["forces"][0][0][0] = 1e-6
        elif defect == "capacity":
            sample["resources"]["rejected_allocations"] = 1
        elif defect == "backend":
            sample["backend"] = ["cpu"]

    order: list = []
    evidence = compare(
        Samples("baseline", order),
        Samples("candidate", order, mutate),
        copy.deepcopy(REFERENCE),
        repeats=5,
    )
    assert not evidence["passed"]


def test_df_baseline_resources_and_owner_identity_are_mandatory() -> None:
    def mutate(sample: dict) -> None:
        sample["resources"]["admission"]["limit_bytes"] = 0

    order: list = []
    baseline, candidate = (
        Samples("baseline", order, mutate),
        Samples("candidate", order),
    )
    candidate.identity = {"device": "different-test-device"}
    evidence = compare(baseline, candidate, copy.deepcopy(REFERENCE), repeats=5)
    assert not evidence["passed"]
    assert not evidence["baseline_resources"]["passed"]
    assert not evidence["identity"]["passed"]


@pytest.mark.parametrize("scheduled,repeats", [(False, 5), (True, 4)])
def test_invalid_campaign_refuses_before_loading_native_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduled: bool,
    repeats: int,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if scheduled:
        monkeypatch.setenv("SLURM_JOB_ID", "synthetic-unit-test-only")
    with pytest.raises(ValueError, match="Slurm allocation"):
        run(
            tmp_path / "nonexistent.json",
            tmp_path / "result",
            repeats=repeats,
            policies=[],
        )


@pytest.mark.parametrize("fail_holdout", [False, True])
def test_only_fresh_accepted_holdout_selects_a_policy_without_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_holdout: bool,
) -> None:
    """Exercise controller ownership/publication, not a simulated GPU result."""
    from generativeqc import autotune, profiles

    from benchmarks import df96_autotune

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-unit-test-only")
    monkeypatch.setattr(autotune, "source_identity", lambda root: "synthetic-source")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("resident benchmark attempted to activate a production profile")

    monkeypatch.setattr(profiles, "install_bundle", forbidden)
    binary = tmp_path / "nonexecutable.so"
    binary.write_bytes(b"synthetic test artifact, not a native library")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({**REFERENCE, "converged": True, "orbital_gradient_norm": 0.0})
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "library": str(binary),
                "library_hash": profiles.file_hash(binary),
                "reference": str(reference),
                "workload": {
                    "kind": "df-hf",
                    "method": "rhf",
                    "batch": 1,
                    "atoms": [["H", [0.0, 0.0, 0.0]]] * 96,
                    "source_identity": "synthetic-source",
                },
            }
        )
    )
    owners: list = []
    order: list = []

    class Owner(Samples):
        def __init__(
            self,
            library: Path,
            plan: Path,
            configuration: dict,
            directory: Path,
            **options: Any,
        ) -> None:
            assert options["worker_component"] == "df"
            assert options["worker_module"] == "benchmarks.df96_autotune"
            side = "baseline" if directory.name == "baseline" else "candidate"

            def mutate(sample: dict) -> None:
                if (
                    fail_holdout
                    and directory.parent.name == "holdout"
                    and side == "candidate"
                ):
                    sample["backend"] = ["cpu"]

            super().__init__(side, order, mutate)
            self.ready = {"cold": {"test_only": True}}
            owners.append(self)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exception: object) -> None:
            pass

    monkeypatch.setattr(df96_autotune, "Resident", Owner)
    policy = {"GENERATIVEQC_DF_EXCHANGE": "dense"}
    report = run(plan, tmp_path / "result", repeats=5, policies=[policy])
    assert len(owners) == 4
    assert report["trials"][0]["evidence"]["passed"]
    assert report["holdout"]["passed"] is not fail_holdout
    assert report["selected"] == (None if fail_holdout else policy)
    assert json.loads((tmp_path / "result/report.json").read_text()) == report


def test_rejected_worker_cleanup_preserves_admission_error() -> None:
    """A dead control pipe must not mask rejection or leak remaining handles."""
    from types import SimpleNamespace

    from benchmarks.dft96_autotune import Resident

    closed: list = []

    class DeadPipe:
        def close(self) -> None:
            closed.append("stream")
            raise BrokenPipeError("worker already rejected the request")

    owner = Resident.__new__(Resident)
    owner.stream = DeadPipe()
    owner.socket = SimpleNamespace(close=lambda: closed.append("socket"))
    owner.log = SimpleNamespace(close=lambda: closed.append("log"))
    owner.process = SimpleNamespace(poll=lambda: 1)
    with pytest.raises(RuntimeError, match="original resource refusal"):
        try:
            raise RuntimeError("original resource refusal")
        finally:
            owner.close()
    assert closed == ["stream", "socket", "log"]
