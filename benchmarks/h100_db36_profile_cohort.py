"""Replay DB36's actual SQLite profile and separately labelled 7013 baseline."""

import ast
import hashlib
import json
import lzma
import subprocess
from pathlib import Path

from benchmarks import h100_db36_safety_cohort as safety

DIRECTORY = safety.ROOT / "benchmarks/results/h100-sm90-db36-profile-baseline-20261103"
PRIOR = "7013148d8d74e714faf20ce87c533db916264f37"
PREDECESSOR = "cd2b4514be0a38c4920934481b1f02d157a694b4"
READERS = {
    "unpack-nsys-affected-v1.py",
    "replay-nsys-affected-v5.py",
    "observe-nsys-cpu-tools-v1.py",
    "unpack-terminal-v2.py",
    "replay-df-molecular-v4.py",
    "replay-screening-integration-v4.py",
}
ARCHIVES = {
    "nsys-v1": "nsys-v1-analysis.tar.xz",
    "df-molecular-v1": "df-molecular-v1-completed.tar.xz",
    "screening-v1": "screening-v1-completed.tar.xz",
}
MANIFESTS = {
    "nsys-v1": "nsys-v1-analysis-download.json",
    "df-molecular-v1": "df-molecular-v1-download-manifest.json",
    "screening-v1": "screening-v1-download-manifest.json",
}
LEDGERS = {
    "source-applicability-db36-ledger.json.xz",
    "source-applicability-7013-ledger.json.xz",
    "source-applicability-review.json.xz",
}
PROOFS = {
    "source7013-df-molecular-v1-local-replay.json": "PASS_CURRENT_SOURCE44_DF_TESTS_FULL_PUBLIC_EF_AND_COMPARISON_REPLAY",
    "source7013-screening-v1-local-replay-v4.json": "PASS_RECEIVER_REPLAY_ALL66_FIXED_POLICY_ATTEMPTS_WITH_FAILURES_RETAINED",
    "source-applicability-admission-v1.json": "PASS_SOURCE_APPLICABILITY_ONLY_NOT_NEW_DEVICE_ACCEPTANCE",
    "nsys-v1-local-replay-affected-v1.json": "PASS_CURRENT_SOURCE_ONE_AFFECTED_DIRECT_CAPTURE4_EF_ACTUAL_SQLITE_REPLAY",
}


def check(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decoded(binding: dict) -> bytes:
    raw = lzma.decompress((DIRECTORY / binding["path"]).read_bytes())
    check(
        len(raw) == binding["decoded_bytes"] and sha(raw) == binding["decoded_sha256"],
        "Decoded original differs",
    )
    return raw


def checked_publication() -> dict:
    p = json.loads((DIRECTORY / "cohort.json").read_text())
    check(
        p["schema"] == "generativeqc.h100-db36-profile-baseline.v1"
        and p["source_revision"] == safety.SOURCE
        and p["source_tree"] == safety.TREE
        and p["core_sha256"] == safety.CORE
        and p["prior_source_revision"] == PRIOR
        and p["majority_accepted"] is False
        and p["clean_final_timing_votes"] == 0,
        "Profile/baseline source scope differs",
    )
    required = (
        {"readers/" + n for n in READERS}
        | set(ARCHIVES.values())
        | {n + ".xz" for n in MANIFESTS.values()}
        | LEDGERS
        | {"source-applicability-admission-v1.json"}
    )
    check(
        len(p["files"]) == len(required)
        and {b["path"] for b in p["files"]} == required,
        "Complete profile/baseline inventory differs",
    )
    entries = {b["path"]: b for b in p["files"]}
    for binding in p["files"]:
        relative = Path(binding["path"])
        path = DIRECTORY / relative
        check(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.name != ".env"
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == binding["bytes"]
            and sha(path.read_bytes()) == binding["sha256"],
            "Profile/baseline original differs: " + str(relative),
        )
    check(
        [b["cohort"] for b in p["packets"]] == list(ARCHIVES), "Packet sequence differs"
    )
    for packet in p["packets"]:
        name = packet["cohort"]
        check(
            packet["archive"] == entries[ARCHIVES[name]]
            and packet["manifest"]["decoded_name"] == MANIFESTS[name]
            and packet["manifest"]["path"] == MANIFESTS[name] + ".xz",
            "Original packet binding differs",
        )
        check(
            all(
                packet["manifest"][k] == entries[MANIFESTS[name] + ".xz"][k]
                for k in ["path", "bytes", "sha256"]
            ),
            "Manifest storage differs",
        )
        decoded(packet["manifest"])
    check(
        len(p["source_ledgers"]) == len(LEDGERS)
        and {b["path"] for b in p["source_ledgers"]} == LEDGERS,
        "Source ledger inventory differs",
    )
    for binding in p["source_ledgers"]:
        check(
            all(
                binding[k] == entries[binding["path"]][k]
                for k in ["path", "bytes", "sha256"]
            ),
            "Source ledger storage differs",
        )
        decoded(binding)
    check(
        set(p["proofs"]) == set(PROOFS)
        and all(
            p["proofs"][n]["status"] == s
            and p["proofs"][n]["source_revision"]
            == (PRIOR if n.startswith("source7013-") else safety.SOURCE)
            for n, s in PROOFS.items()
        ),
        "Prior/current proof source labels differ",
    )
    return p


def install(task: Path, cohort: str) -> None:
    producers = task / (cohort + "-executed-producers")
    for source in producers.rglob("*"):
        if source.is_file():
            safety.support.copy_exact(source, task / source.relative_to(producers))


def stage(task: Path, publication: dict, cohort: str) -> None:
    packet = next(p for p in publication["packets"] if p["cohort"] == cohort)
    safety.support.copy_exact(
        DIRECTORY / packet["archive"]["path"], task / packet["archive"]["path"]
    )
    binding = packet["manifest"]
    with (task / binding["decoded_name"]).open("xb") as stream:
        stream.write(decoded(binding))


def source_applicability(publication: dict) -> dict:
    ledgers = {b["path"]: decoded(b) for b in publication["source_ledgers"]}
    older = json.loads(ledgers["source-applicability-7013-ledger.json.xz"])
    current = json.loads(ledgers["source-applicability-db36-ledger.json.xz"])
    admission = json.loads(
        (DIRECTORY / "source-applicability-admission-v1.json").read_text()
    )
    check(
        sha(ledgers["source-applicability-review.json.xz"])
        == admission["review_report_sha256"]
        and sha(ledgers["source-applicability-db36-ledger.json.xz"])
        == admission["source_applicability_ledger_sha256"]
        and sha(ledgers["source-applicability-7013-ledger.json.xz"])
        == current["prior_CDto7013_ledger"]["sha256"],
        "Completed applicability report/ledger binding differs",
    )
    check(
        admission["status"] == PROOFS["source-applicability-admission-v1.json"]
        and admission["source_revision"] == current["source_revision"] == safety.SOURCE
        and current["prior_baseline_source"] == older["source_revision"] == PRIOR
        and older["predecessor_source"] == PREDECESSOR
        and admission["unresolved_blockers"] == []
        and admission["scientific_acceptance"] is False
        and admission["safety_acceptance"] is False
        and admission["performance_acceptance"] is False,
        "Source-only applicability was promoted",
    )
    revisions = [PREDECESSOR, PRIOR, safety.SOURCE]

    def git(revision: str, path: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(safety.ROOT), "show", revision + ":" + path],
            env=safety.support.ENV,
        )

    check(
        len(older["unchanged_complete_blobs"]) == 14,
        "Historical API comparison population differs",
    )
    for path, expected in older["unchanged_complete_blobs"].items():
        values = [git(revision, path) for revision in revisions]
        check(
            values[0] == values[1] == values[2] and sha(values[0]) == expected,
            "Historical API/precision blob differs: " + path,
        )
    methods = older["unchanged_calculator_method_AST"]
    check(
        methods == ["prepare_batch", "_resource_request", "_preflight_hf_basis"],
        "Historical method comparison population differs",
    )
    methods_by_revision = []
    for revision in revisions:
        tree = ast.parse(git(revision, "python/generativeqc/calculator.py"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Calculator"
        )
        methods_by_revision.append(
            {
                n.name: ast.dump(n, include_attributes=False)
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in methods
            }
        )
    check(
        len(methods_by_revision[0]) == 3
        and methods_by_revision[0] == methods_by_revision[1] == methods_by_revision[2],
        "Historical Calculator methods differ",
    )
    changed = subprocess.check_output(
        ["git", "-C", str(safety.ROOT), "diff", "--name-only", PRIOR, safety.SOURCE],
        text=True,
        env=safety.support.ENV,
    ).splitlines()
    check(changed == current["all_changed_paths"], "Historical source delta differs")
    production = [
        p
        for p in changed
        if p.startswith(("src/", "python/", "manifests/", "cmake/"))
        or p == "CMakeLists.txt"
    ]
    check(
        production == current["changed_production_paths"],
        "Affected production paths differ",
    )
    return admission


def extend(base: Path, task: Path) -> dict:
    publication = checked_publication()
    prior = base / "tasks/cohort-7013"
    prior.mkdir()
    for name in READERS:
        destination = task if "nsys" in name else prior
        safety.support.copy_exact(DIRECTORY / "readers" / name, destination / name)
    for cohort, receiver, proof in [
        (
            "df-molecular-v1",
            "replay-df-molecular-v4.py",
            "df-molecular-v1-local-replay.json",
        ),
        (
            "screening-v1",
            "replay-screening-integration-v4.py",
            "screening-v1-local-replay-v4.json",
        ),
    ]:
        stage(prior, publication, cohort)
        safety.reader(prior, "unpack-terminal-v2.py", "--cohort", cohort)
        install(prior, cohort)
        safety.reader(prior, receiver)
        safety.support.copy_exact(prior / proof, task / ("source7013-" + proof))
    source_applicability(publication)
    safety.support.copy_exact(
        DIRECTORY / "source-applicability-admission-v1.json",
        task / "source-applicability-admission-v1.json",
    )
    stage(task, publication, "nsys-v1")
    safety.reader(
        task, "unpack-nsys-affected-v1.py", "--manifest", MANIFESTS["nsys-v1"]
    )
    install(task, "nsys-v1")
    safety.reader(task, "replay-nsys-affected-v5.py")
    hashes = {}
    values = {}
    for name, status in PROOFS.items():
        raw = (task / name).read_bytes()
        value = json.loads(raw)
        check(
            sha(raw) == publication["proofs"][name]["sha256"]
            and value["status"] == status
            and value["source_revision"]
            == publication["proofs"][name]["source_revision"],
            "Recomputed profile/prior proof differs: " + name,
        )
        hashes[name], values[name] = sha(raw), value
    app = values["source-applicability-admission-v1.json"]
    check(
        app["prior_DF44_proof_sha256"]
        == hashes["source7013-df-molecular-v1-local-replay.json"]
        and app["baseline7013_replay_sha256"]
        == hashes["source7013-screening-v1-local-replay-v4.json"],
        "Retained-source proof admission differs",
    )
    profile = values["nsys-v1-local-replay-affected-v1.json"]
    check(
        len(profile["rows"]) == 1
        and profile["rows"][0]["samples"] == 4
        and profile["rows"][0]["actual_sqlite_replayed_exactly_locally"] is True
        and profile["rows"][0]["clean_timing_vote"] is False,
        "Intrusive profile population differs",
    )
    reviewed = json.loads(
        lzma.decompress(
            (safety.DIRECTORY / "actual-integration-audit.json.xz").read_bytes()
        )
    )["reviewed_proofs"]
    combined = {n: sha((task / n).read_bytes()) for n in reviewed}
    check(
        len(combined) == 9 and combined == reviewed,
        "Public nine-gate proof closure differs",
    )
    return {
        "status": "PASS_PUBLIC_DB36_PROFILE_AND_RETAINED_7013_BASELINE_ONLY",
        "source_revision": safety.SOURCE,
        "prior_source_revision": PRIOR,
        "proofs": hashes,
        "all_nine_gate_proofs": combined,
        "profile_complete_EF": 4,
        "prior_DF_tests": 44,
        "prior_DF_complete_EF": 299,
        "prior_screening_processes": 66,
        "prior_screening_complete_EF": 264,
        "clean_final_timing_votes": 0,
        "majority_accepted": False,
    }


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(safety.replay(extension=extend), indent=2))


if __name__ == "__main__":
    main()
