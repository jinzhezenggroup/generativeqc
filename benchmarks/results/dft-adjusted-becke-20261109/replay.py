"""Replay a rejected experiment, including its incomplete A100 diagnostics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from benchmarks._exact_fp64_evidence import load
from benchmarks.dft_becke_qualification import qualify_becke

FAMILY = Path(__file__).resolve().parent


def replay(directory: Path = FAMILY) -> dict:
    """Authenticate original observations and recompute why promotion was rejected."""

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise ValueError(detail)

    cohort = json.loads((directory / "cohort.json").read_text())
    bundle, files = load(directory)
    require(len(files) == cohort["member_count"], "Missing original evidence")
    storage = json.loads((directory / "input-storage-receipt.json").read_text())
    for entry in storage["files"]:
        path = FAMILY.parents[2] / entry["path"]
        payload = path.read_bytes()
        require(
            len(payload) == entry["bytes"]
            and hashlib.sha256(payload).hexdigest() == entry["sha256"],
            "Changed scientific input container",
        )
        with ZipFile(path) as archive:
            actual = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
            }
            require(actual == entry["members"], "Changed original NPY array bytes")
    image = cohort["candidate_image"]
    prefix = "normal-builds/candidate/build-receipt/"
    require(json.loads(files[prefix + "identity.json"]) == image, "Wrong built ELF")
    for name in (prefix + "exit-code.txt", "normal-builds/owner-build/exit-code.txt"):
        require(files[name].strip() == "0", "Failed normal or native-owner build")
    require(
        files[prefix + "restored-runtime.sha256"].split()[0]
        == "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a",
        "Borrowed runtime not restored",
    )
    owner_digest = hashlib.sha256(files["reproducer/owner.cu"].encode()).hexdigest()
    require(
        files["normal-builds/owner-build/identity.sha256"].split()[0] == owner_digest,
        "Native fixture source changed",
    )
    require(
        files["failed-owner-harness-16867/exit-code.txt"].strip() != "0",
        "Lost earlier test-harness failure",
    )
    transport = json.loads(files["reproducer/execute-a100.out.transport.json"])
    require(transport["remote_returncode"] == 1, "Lost actual A100 failure")
    freeze = json.loads(files["reproducer/candidate/source-receipt.json"])
    require(
        freeze["normal_release"]
        and freeze["source_identity"] == image["source_identity"]
        and freeze["control_source_identity"]
        == cohort["qualified_production_source_restored"],
        "Changed experimental versus restored production provenance",
    )
    cards = {}
    with tempfile.TemporaryDirectory(prefix="rejected-becke-") as temporary:
        destination = Path(temporary)
        for name, text in files.items():
            relative = PurePosixPath(name)
            require(
                not relative.is_absolute() and ".." not in relative.parts,
                "Unsafe receipt path",
            )
            if relative.parts[0] in ("native-a100", "native-5090"):
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        for card in ("a100", "5090"):
            report = qualify_becke(destination / ("native-" + card), card, image=image)
            require(
                not report["default_promotion_qualified"],
                "Historical rejected experiment unexpectedly qualified",
            )
            cards[card] = report
    require(
        cards["a100"]["allocation_exit"] == 1
        and len(cards["a100"]["missing_diagnostics"]) == 4
        and not cards["a100"]["profiles"],
        "Incomplete A100 profiling was hidden",
    )
    require(
        cards["5090"]["single_pairs"]["pbe0-48"]["cold_ratio"] > 1.03,
        "Lost observed Direct48 single-pair regression",
    )
    return {
        "default_promotion_qualified": False,
        "all_readme_leadership_established": False,
        "retained_members": len(files),
        "scope": bundle["experiment"]["scope"],
        "cards": cards,
        "qualified_production_source_restored": cohort[
            "qualified_production_source_restored"
        ],
    }


if __name__ == "__main__":
    result = replay()
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "cards"}
            | {
                "cards": {
                    card: {
                        key: report[key]
                        for key in (
                            "methods",
                            "single_pairs",
                            "performance_failures",
                            "allocation_exit",
                            "missing_diagnostics",
                        )
                    }
                    for card, report in result["cards"].items()
                }
            },
            indent=2,
        )
    )
