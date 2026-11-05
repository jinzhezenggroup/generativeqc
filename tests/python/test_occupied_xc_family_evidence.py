"""Keep current-source family measurements replayable without a GPU stack."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.occupied_xc_proposals import load

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/results/occupied-xc-families-20261105/replay.py"


@pytest.fixture(scope="module")
def retained() -> tuple[ModuleType, dict]:
    """Load only the standard-library reader and immutable retained capsule."""
    spec = importlib.util.spec_from_file_location("occupied_family_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, load(SCRIPT.parent / "endpoints.json.xz")


def test_retained_family_population_and_admission(
    retained: tuple[ModuleType, dict],
) -> None:
    module, capsule = retained
    summary = module.replay(capsule)
    assert len(summary["current"]) == 26
    for name in ("large-m06-2x-rks", "confirmation-large-m06-2x-rks"):
        assert summary["current"][name]["phases"]["cold"]["profitable"]
    assert summary["original"]["cold"]["profitable"]
    assert not summary["original"]["warm"]["regresses"]
    assert summary["current"]["confirmation-small-pbe0-rks"]["phases"]["warm"][
        "regresses"
    ]
    assert set(capsule["diagnostics"]) == {
        f"{method}-{spin}"
        for method in ("lda", "pbe", "pbe0", "b3lyp", "r2scan", "m06-2x", "mn15")
        for spin in ("rks", "uks")
    }
    for population in capsule["populations"].values():
        assert len(population) == 6
    for diagnostic in capsule["diagnostics"].values():
        assert any(diagnostic["frames"][0])


@pytest.mark.parametrize(
    "mutation",
    [
        "image",
        "host",
        "force",
        "inventory",
        "precision",
        "reference",
        "tail",
        "work",
        "original",
        "source_work",
        "jk",
        "history",
    ],
)
def test_retained_family_corruption_is_rejected(
    retained: tuple[ModuleType, dict], mutation: str
) -> None:
    module, original = retained
    capsule = copy.deepcopy(original)
    record = next(iter(capsule["populations"].values()))[-1]
    if mutation == "image":
        record["library_sha256"] = "0" * 64
    elif mutation == "host":
        record["host"] = "another-host"
    elif mutation == "force":
        record["observations"][0]["forces"][0][0] += 1
    elif mutation == "inventory":
        record["observations"][0]["precision"]["operator_inventory_complete"] = False
    elif mutation == "precision":
        record["observations"][0]["precision"]["operators"][0]["compute"] = "fp32"
    elif mutation == "reference":
        capsule["second_references"].clear()
    elif mutation == "tail":
        next(iter(capsule["diagnostics"].values()))["frames"][0][-1] = 1
    elif mutation == "work":
        capsule["source_handoff"][1]["receipt"]["observations"][0]["fock_builds"] += 1
    elif mutation == "original":
        capsule["original_r2scan3c"]["r0-unset"]["source_revision"] = "0" * 40
    elif mutation == "source_work":
        capsule["source_handoff"][1]["receipt"]["observations"][0]["precision"][
            "operators"
        ][0]["count"] += 1
    elif mutation == "jk":
        capsule["incremental_handoff"]["after"]["unset"]["trace_work"][0][
            "structured_basis"
        ] += 1
    else:
        capsule["history_diagnosis"]["records"][0]["source_identity"] = "0" * 64
    with pytest.raises(ValueError):
        module.replay(capsule)


@pytest.mark.parametrize("optimize", [False, True])
def test_retained_family_gates_survive_optimized_python(optimize: bool) -> None:
    command = [sys.executable, *(["-O"] if optimize else []), str(SCRIPT)]
    process = subprocess.run(
        command,
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "python"))),
            GENERATIVEQC_LIBRARY=str(SCRIPT.parent / "unavailable-offline-library.so"),
        ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
