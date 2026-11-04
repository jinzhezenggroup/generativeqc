"""Recompute the original DB36 H100 fixed30 result and its nine-gate closure."""

import io
import json
import lzma
import tarfile
from pathlib import Path

from benchmarks import h100_db36_profile_cohort as profile

safety = profile.safety
DIRECTORY = safety.ROOT / "benchmarks/results/h100-sm90-db36-final-20261103"
READERS = {"unpack-final-v1.py", "replay-final-pairs-v3.py"}
REQUIRED = {
    "final-pairs-v1-completed.tar.xz.part00",
    "final-pairs-v1-completed.tar.xz.part01",
    "final-pairs-v1-download-manifest.json.xz",
    "actual-result-audit.tar.xz",
} | {"readers/" + name for name in READERS}


def checked_publication() -> dict:
    """Authenticate the complete historical raw packet and actual review."""
    p = json.loads((DIRECTORY / "cohort.json").read_text())
    profile.check(
        p["schema"] == "generativeqc.h100-db36-fixed30-final.v1"
        and p["source_revision"] == safety.SOURCE
        and p["source_tree"] == safety.TREE
        and p["core_sha256"] == safety.CORE
        and p["original_final_protocol_sha256"]
        == "2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa",
        "Frozen final source or original scientific contract differs",
    )
    profile.check(
        len(p["files"]) == len(REQUIRED)
        and {b["path"] for b in p["files"]} == REQUIRED,
        "Complete final publication inventory differs",
    )
    entries = {b["path"]: b for b in p["files"]}
    for b in p["files"]:
        name = Path(b["path"])
        path = DIRECTORY / name
        profile.check(
            not name.is_absolute()
            and ".." not in name.parts
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == b["bytes"]
            and profile.sha(path.read_bytes()) == b["sha256"],
            "Final original differs: " + str(name),
        )
    profile.check(
        [b["path"] for b in p["parts"]]
        == [
            "final-pairs-v1-completed.tar.xz.part00",
            "final-pairs-v1-completed.tar.xz.part01",
        ]
        and all(b == entries[b["path"]] for b in p["parts"])
        and p["archive"]["name"] == "final-pairs-v1-completed.tar.xz",
        "Original final archive part sequence differs",
    )
    manifest = p["manifest"]
    profile.check(
        manifest["decoded_name"] == "final-pairs-v1-download-manifest.json"
        and manifest["path"] == manifest["decoded_name"] + ".xz"
        and all(
            manifest[k] == entries[manifest["path"]][k]
            for k in ["path", "bytes", "sha256"]
        ),
        "Original final manifest binding differs",
    )
    decoded = lzma.decompress((DIRECTORY / manifest["path"]).read_bytes())
    profile.check(
        len(decoded) == manifest["decoded_bytes"]
        and profile.sha(decoded) == manifest["decoded_sha256"],
        "Decoded final original manifest differs",
    )
    audit_members(p)
    return p


def audit_members(publication: dict) -> dict[str, bytes]:
    """Read sealed review metadata without executing historical audit scripts."""
    binding = publication["actual_result_audit"]
    profile.check(
        binding["archive"]
        == next(
            b for b in publication["files"] if b["path"] == "actual-result-audit.tar.xz"
        )
        and binding["review_independence"] == "same-family"
        and binding["acceptance_status"] == "provisional",
        "Actual review scope or storage differs",
    )
    expected = {b["path"]: b for b in binding["members"]}
    values = {}
    with tarfile.open(
        fileobj=io.BytesIO((DIRECTORY / binding["archive"]["path"]).read_bytes())
    ) as stream:
        members = stream.getmembers()
        profile.check(
            len(members) == len(expected) == len(binding["members"])
            and {m.name for m in members} == set(expected),
            "Actual review metadata population differs",
        )
        for member in members:
            path = Path(member.name)
            profile.check(
                member.isfile()
                and not path.is_absolute()
                and ".." not in path.parts
                and path.name != ".env",
                "Unsafe review member",
            )
            raw = stream.extractfile(member).read()
            b = expected[member.name]
            profile.check(
                len(raw) == b["bytes"] and profile.sha(raw) == b["sha256"],
                "Actual review member differs",
            )
            values[member.name] = raw
    report = json.loads(values["reviewer/audit-report.json"])
    host = json.loads(values["host-completion-trace-v1.json"])
    profile.check(
        profile.sha(values["reviewer/audit-report.json"])
        == binding["report_sha256"]
        == host["report_sha256"]
        and profile.sha(values["input-seal-v1.json"]) == host["input_seal_sha256"]
        and profile.sha(values["host-received-final-response-v1.txt"])
        == host["response_sha256"]
        and all(
            host[k] is True
            for k in ["review_completed", "review_scope_complete", "review_positive"]
        )
        and report["actual_result_verdict"]
        == "PASS_ACTUAL_FINAL_FIXED30_SCIENCE_AND_THREE_PAIR_MAJORITY"
        and report["unresolved_blockers"] == []
        and report["review_independence"] == "same-family"
        and report["acceptance_status"] == "provisional",
        "Actual result report or host completion binding differs",
    )
    return values


def extend(base: Path, task: Path) -> dict:
    """Close all source-labelled gates before replaying all 180 populations."""
    supporting = profile.extend(base, task)
    p = checked_publication()
    for name in READERS:
        safety.support.copy_exact(DIRECTORY / "readers" / name, task / name)
    with (task / p["archive"]["name"]).open("xb") as stream:
        for b in p["parts"]:
            stream.write((DIRECTORY / b["path"]).read_bytes())
    archive = task / p["archive"]["name"]
    profile.check(
        archive.stat().st_size == p["archive"]["bytes"]
        and safety.support.sha(archive) == p["archive"]["sha256"],
        "Reconstructed final archive differs",
    )
    (task / p["manifest"]["decoded_name"]).write_bytes(
        lzma.decompress((DIRECTORY / p["manifest"]["path"]).read_bytes())
    )
    safety.reader(task, "unpack-final-v1.py")
    # Existing recomputed proofs must match the exact bytes consumed at dispatch.
    # Copying refuses any differing existing file rather than replacing a proof.
    for suffix in ["executed-producers", "available-producers", "available-admissions"]:
        source = task / ("final-pairs-v1-" + suffix)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                safety.support.copy_exact(path, task / path.relative_to(source))
    safety.reader(task, "replay-final-pairs-v3.py")
    proof = task / "final-pairs-v1-local-replay.json"
    result = json.loads(proof.read_text())
    profile.check(
        safety.support.sha(proof) == p["expected_proof_sha256"]
        and result["status"] == "PASS_FIXED30_SCIENCE_AND_THREE_PAIR_MAJORITY"
        and result["source_revision"] == safety.SOURCE
        and result["core_sha256"] == safety.CORE
        and all(result[k] == v for k, v in p["acceptance"].items())
        and result["majority_accepted"] is True
        and result["all30_scientifically_qualified"] is True
        and result["physical_cases"] == 30
        and result["attempted_populations"] == result["qualified_populations"] == 180
        and result["returned_EF"] == 720
        and result["wins"] >= 20
        and result["complete_set_geomean"] > 1,
        "Original final scientific or three-pair majority result differs",
    )
    metadata = audit_members(p)
    report = json.loads(metadata["reviewer/audit-report.json"])
    profile.check(
        all(
            report[k] == result[k]
            for k in [
                "physical_cases",
                "attempted_populations",
                "qualified_populations",
                "returned_EF",
                "wins",
                "complete_set_geomean",
            ]
        ),
        "Actual result review differs from replay",
    )
    return {
        "supporting": supporting,
        "final_proof_sha256": safety.support.sha(proof),
        "final": result,
    }


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(safety.replay(extension=extend), indent=2))


if __name__ == "__main__":
    main()
