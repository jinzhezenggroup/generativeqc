"""Replay new-master native qualification without relabelling historical images."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path, PurePosixPath
from statistics import median
from tempfile import TemporaryDirectory

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.dft_streaming_qualification import qualify_native


def replay(directory: Path = Path(__file__).parent) -> dict:
    """Authenticate every observation, independent gate, normal image and default."""
    _, files = load(directory)
    cohort = json.loads((directory / "cohort.json").read_text())
    require(len(files) == cohort["member_count"], "Incomplete retained population")
    cards = {}
    with TemporaryDirectory(prefix="dft-master-replay-") as temporary:
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
            report = qualify_native(
                destination / ("native-" + card), card, images=cohort["native_images"]
            )
            require("b3lyp_cold_diagnostic" in report, "Missing B3LYP cold attribution")
            for result in report["methods"].values():
                require(result["paired_warm_gate"], "Paired warm regression")
                if card == "a100":
                    relative_mads = [
                        median(abs(value - median(values)) for value in values)
                        / median(values)
                        for values in result["cold_samples"].values()
                    ]
                    gate = max(0.02, 2 * sum(relative_mads))
                    require(result["cold_ratio"] <= 1 - gate, "Cold/noise gate failed")
                else:
                    require(result["cold_ratio"] <= 1.03, "5090 cold regression")
            cards[card] = report
    require(
        set(cards["a100"].get("df_single_pair_diagnostics", {})) == {"pbe0", "b3lyp"},
        "Missing DF single-pair diagnostics",
    )
    for arm, expected in cohort["native_images"].items():
        prefix = "native-builds/" + arm + "/build-receipt/"
        require(
            json.loads(files[prefix + "identity.json"]) == expected, "Wrong built ELF"
        )
        require(files[prefix + "exit-code.txt"].strip() == "0", "Failed normal build")
        require(
            files[prefix + "restored-runtime.sha256"].split()[0]
            == "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a",
            "Borrowed runtime not restored",
        )
    require(
        files["native-builds/control/build-receipt-timeout-16857/exit-code.txt"].strip()
        == "124",
        "Missing original compile timeout",
    )
    reuse = json.loads(files["reproducer/raw-k-owner-reuse.json"])
    require(reuse["nonarray_executable_identical"], "Changed raw-K fixture owner")
    resource_proof = json.loads(files["reproducer/direct-aot-ledger-preflight.json"])
    previous_text = files["reproducer/previous-direct-jk-resource-ledger.json"]
    require(
        hashlib.sha256(previous_text.encode()).hexdigest()
        == resource_proof["original_ledger_sha256"],
        "Changed prior resource ledger",
    )
    previous = json.loads(previous_text)
    measured = json.loads(files["ci-direct-aot/direct-jk-aot/report.json"])[
        "measurement"
    ]
    require(previous["kernels"] == measured["kernels"], "sm120 kernel resource drift")
    for key, value in measured["identity"].items():
        if key != "manifest_sha256":
            require(previous[key] == value, "sm120 qualified identity drift")
    require(
        measured["identity"]["manifest_sha256"]
        == resource_proof["new_manifest_sha256"],
        "Wrong resource manifest binding",
    )
    historical = directory.parent / "dft-shared-rys-streaming-20261108"
    prior = runpy.run_path(str(historical / "replay.py"))["replay"](historical)
    return {
        "default_promotion_qualified": True,
        "all_readme_leadership_established": False,
        "scope": cohort["scope"],
        "retained_members": len(files),
        "cards": cards,
        "historical_raw_k_matrix_predicates": prior["raw_k_matrix_predicates"],
        "raw_k_boundary": reuse["boundary"],
        "retained_sm120_resource_kernels": len(measured["kernels"]),
        "known_remaining_gaps": cohort["known_remaining_gaps"],
    }


if __name__ == "__main__":
    report = replay()
    print(
        json.dumps(
            {
                "default_promotion_qualified": report["default_promotion_qualified"],
                "all_readme_leadership_established": False,
                "retained_members": report["retained_members"],
                "cards": {
                    card: {
                        method: {
                            key: result[key]
                            for key in ("cold_means", "cold_ratio", "paired_warm_gate")
                        }
                        for method, result in measurements["methods"].items()
                    }
                    for card, measurements in report["cards"].items()
                },
                "df_single_pairs": report["cards"]["a100"][
                    "df_single_pair_diagnostics"
                ],
                "known_remaining_gaps": report["known_remaining_gaps"],
            },
            indent=2,
        )
    )
