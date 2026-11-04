"""Recompute the frozen H100 fixed30 three-pair result from complete raw data."""

import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

from benchmarks import h100_public_guards as guards
from tools.restore_retained_evidence import read_evidence_bytes

support = guards.support
DIRECTORY = support.ROOT / "benchmarks/results/h100-sm90-cd2b-final-20261103"
HISTORICAL = {
    "final-pairs-v3-completed.tar.xz.part00",
    "final-pairs-v3-completed.tar.xz.part01",
    "final-pairs-v3-download-manifest.json.xz",
    "precompute-failures-v1.tar.xz",
    "actual-result-audit.json.xz",
}
REQUIRED = {
    "final-pairs-v3-completed.tar.xz.part00",
    "final-pairs-v3-completed.tar.xz.part01",
    "final-pairs-v3-download-manifest.json.xz",
    "precompute-failures-v1.tar.xz",
    "precompute-failures-v1.json",
    "actual-result-audit.json.xz",
    "readers/replay-final-pairs-public-v1.py",
}


def check(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def artifact_bytes(name: str) -> bytes:
    """Authenticate current or explicitly retained original final-cohort bytes."""
    relative = Path(name)
    path = DIRECTORY / relative
    check(
        not relative.is_absolute()
        and ".." not in relative.parts
        and not path.is_symlink(),
        "Unsafe final evidence path",
    )
    if path.is_file():
        return path.read_bytes()
    check(not path.exists() and name in HISTORICAL, "Missing original final evidence")
    original = (
        support.ROOT / "benchmarks/results/h100-sm90-cd2b-final-20261103" / relative
    )
    try:
        return read_evidence_bytes(
            original, manifest=DIRECTORY / "historical-snapshot.json"
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError("Missing historical final evidence: " + name) from error


def checked_publication() -> dict:
    publication = json.loads((DIRECTORY / "cohort.json").read_text())
    entries = publication["files"]
    check(
        publication["schema"] == "generativeqc.h100-fixed30-final.v1"
        and publication["source_revision"] == support.SOURCE
        and publication["core_sha256"]
        == "37b8bd02c1dddbf42385c494ae21ff51b50d874dae1dd0c01ab2b75fb0eb194e"
        and publication["original_final_protocol_sha256"]
        == "2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa",
        "Final source or original acceptance contract differs",
    )
    check(
        len(entries) == len(REQUIRED) and {b["path"] for b in entries} == REQUIRED,
        "Mandatory final evidence inventory differs",
    )
    for binding in entries:
        relative = Path(binding["path"])
        data = artifact_bytes(binding["path"])
        check(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(data) == binding["bytes"]
            and hashlib.sha256(data).hexdigest() == binding["sha256"],
            "Final evidence bytes differ: " + str(relative),
        )
    parts = publication["parts"]
    check(
        [b["path"] for b in parts]
        == [
            "final-pairs-v3-completed.tar.xz.part00",
            "final-pairs-v3-completed.tar.xz.part01",
        ]
        and all(b == next(e for e in entries if e["path"] == b["path"]) for b in parts),
        "Final archive part order or identity differs",
    )
    return publication


def extend(task: Path, native: Path, reference: Path) -> list[dict]:
    publication = checked_publication()
    outputs = guards.extend(task, native, reference)
    archive = task / "final-pairs-v3-completed.tar.xz"
    check(publication["archive"]["name"] == archive.name, "Final archive name differs")
    with archive.open("xb") as stream:
        for binding in publication["parts"]:
            stream.write(artifact_bytes(binding["path"]))
    check(
        archive.stat().st_size == publication["archive"]["bytes"]
        and support.sha(archive) == publication["archive"]["sha256"],
        "Final archive reconstruction differs",
    )
    binding = publication["manifest"]
    check(
        binding["name"] == "final-pairs-v3-download-manifest.json"
        and binding["stored"] == binding["name"] + ".xz",
        "Final manifest identity differs",
    )
    raw = lzma.decompress(artifact_bytes(binding["stored"]))
    check(
        len(raw) == binding["decoded_bytes"]
        and hashlib.sha256(raw).hexdigest() == binding["decoded_sha256"],
        "Decoded original manifest differs",
    )
    with (task / binding["name"]).open("xb") as stream:
        stream.write(raw)
    support.reader(
        support.artifact("unpack-cohort-v2.py"),
        task,
        native,
        reference,
        "--cohort",
        "final-pairs-v3",
    )
    support.install(task, "final-pairs-v3")
    # The frozen receiver checks every imported helper against the pinned Git
    # revision. Add local object metadata only to this temporary reference view.
    subprocess.run(["git", "init", "-q", str(reference)], check=True, env=support.ENV)
    objects = subprocess.check_output(
        [
            "git",
            "-C",
            str(support.REPO),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        ],
        text=True,
        env=support.ENV,
    ).strip()
    (reference / ".git/objects/info/alternates").write_text(objects + "\n")
    (reference / ".git/HEAD").write_text(support.REFERENCE + "\n")
    script = task / "replay-final-pairs-public-v1.py"
    support.copy_exact(DIRECTORY / "readers" / script.name, script)
    support.reader(script, task, native, reference)
    proof = task / "final-pairs-v3-local-replay.json"
    result = json.loads(proof.read_text())
    check(
        support.sha(proof) == publication["expected_proof_sha256"]
        and result["status"] == "PASS_FIXED30_SCIENCE_AND_THREE_PAIR_MAJORITY"
        and result["majority_accepted"] is True
        and result["all30_scientifically_qualified"] is True
        and result["physical_cases"] == 30
        and result["attempted_populations"] == result["qualified_populations"] == 180
        and result["returned_EF"] == 720
        and result["wins"] >= 20
        and result["complete_set_geomean"] > 1,
        "Final fixed30 raw replay is not accepted",
    )
    outputs.append(
        {
            key: result[key]
            for key in [
                "status",
                "source_revision",
                "core_sha256",
                "physical_cases",
                "wins",
                "complete_set_geomean",
                "all30_scientifically_qualified",
                "attempted_populations",
                "qualified_populations",
                "returned_EF",
                "device",
                "cases",
            ]
        }
    )
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    check(
        sys.version_info >= (3, 12) and not sys.flags.optimize,
        "Use unoptimized Python 3.12 or later",
    )
    phases = support.replay(extension=extend)
    print(json.dumps({"supporting_phases": phases[:-1], "final": phases[-1]}, indent=2))


if __name__ == "__main__":
    main()
