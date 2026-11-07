"""Replay a stopped CBS source audit using Git and the Python standard library."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = "6707989cce434b0ca4ef43f9920d5f91525f30c1"
AUDITED = "7109b3d208d23352f266e2bf7440fdfcc71df974"
INPUT = "4370d6c4b1bf4033818f31a9fcc622d3045a33bf"
BLOCKER = "c630ceaeb18846fca5a98f3b621f67fc34938b2a"
FROZEN = {
    (INPUT, "benchmarks/results/cbs-routine-standard-20261025/records.json.xz"),
    *(
        (AUDITED, "benchmarks/results/h100-cbs-5090-cohort-20261105/" + name)
        for name in ("README.md", "replay.py", "records-01.json.xz", "summary.json.xz")
    ),
}
PATCHES = {
    BLOCKER: ("src/scf/rhf.cpp", "src/hf/df_reference.cpp", "src/scf/fock_symmetry.hpp"),
    "94e1eaf529d91cbb2aa7a163367caeb30beae1d8": (
        "src/scf/rhf.cpp",
        "src/scf/solver/validation_products.hpp",
    ),
    "1943a24b7c851a9c2bf4f68a1026c9f637c98f12": (
        "src/scf/default_controls.hpp",
        "src/scf/types.hpp",
        "python/generativeqc/calculator.py",
    ),
}


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def git(repository: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


def main() -> None:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]
    records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    require(records["source"] == SOURCE, "requested source changed")
    require(records["audited_master"] == AUDITED, "audit endpoint changed")
    require(records["scientific_input_revision"] == INPUT, "input revision changed")
    require(
        git(repository, "rev-parse", SOURCE + "^{tree}").decode().strip()
        == "f9cb8c86cf88eda9ce7f7e858d0d7108c3baffdf",
        "measured source tree differs",
    )
    git(repository, "merge-base", "--is-ancestor", SOURCE, BLOCKER)
    git(repository, "merge-base", "--is-ancestor", BLOCKER, AUDITED)
    history = git(
        repository, "log", "--first-parent", "--format=%H %s", f"{SOURCE}..{AUDITED}"
    ).decode().splitlines()
    require(history == records["intervening_commits"], "audit history differs")
    require(
        len(records["frozen_files"]) == len(FROZEN)
        and {(item["revision"], item["path"]) for item in records["frozen_files"]} == FROZEN,
        "frozen input population differs",
    )
    for item in records["frozen_files"]:
        raw = git(repository, "show", item["revision"] + ":" + item["path"])
        require(
            hashlib.sha256(raw).hexdigest() == item["sha256"],
            "frozen input differs: " + item["path"],
        )
    require(
        len(records["patches"]) == len(PATCHES)
        and {item["commit"]: tuple(item["paths"]) for item in records["patches"]} == PATCHES,
        "source patch population differs",
    )
    for item in records["patches"]:
        raw = git(
            repository,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            item["commit"] + "^",
            item["commit"],
            "--",
            *item["paths"],
        )
        require(raw.decode() == item["diff"], "retained source patch differs")
    require(
        git(repository, "show", SOURCE + ":python/generativeqc/cbs.py")
        == git(repository, "show", AUDITED + ":python/generativeqc/cbs.py"),
        "CBS-owned convergence controls changed within the pinned audit range",
    )
    require(summary["source_gate"] == "STOP", "source stop was lost")
    require(summary["reason"] == "closed_shell_df_default_fallback_changed", "reason differs")
    require(summary["scientific_status"] == "NOT_RUN", "unmeasured science was promoted")
    require(summary["build_status"] == "NOT_RUN", "unperformed build was promoted")
    require(summary["overall_status"] == "INCOMPLETE", "incomplete task was promoted")
    require(summary["observed_calls"] == dict.fromkeys(summary["planned_calls"], 0), "calls differ")
    require(
        summary["planned_calls"] == {"primary_ef": 73, "startup_ef": 20, "energy_only": 13},
        "planned protocol counts differ",
    )
    require(records["measurements"] == [], "stopped campaign contains measurements")
    require(
        set(summary["execution_identity"])
        == {
            "core_identity", "binary_sha256", "gpu_model", "gpu_uuid", "driver",
            "image_or_os", "slurm_job_id", "slurm_step_id", "allocation_node",
        },
        "execution identity fields differ",
    )
    require(
        all(value is None for value in summary["execution_identity"].values()),
        "unallocated execution identity was invented",
    )
    print(
        json.dumps(
            {"audit_integrity": "PASS", "source_gate": "STOP", "scientific_status": "NOT_RUN"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
