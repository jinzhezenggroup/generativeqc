"""Replay final native DFT qualification, not the earlier interposer pilots."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.dft_streaming_qualification import qualify_native


def replay(directory: Path = Path(__file__).parent) -> dict:
    """Authenticate complete scientific populations and both default guards."""
    bundle, files = load(directory)
    cohort = json.loads((directory / "cohort.json").read_text())
    require(len(files) == cohort["member_count"], "Incomplete retained population")
    cards = {}
    with TemporaryDirectory(prefix="dft-streaming-replay-") as temporary:
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
            cards[card] = qualify_native(
                destination / ("native-" + card), card, images=cohort["native_images"]
            )
            for result in cards[card]["methods"].values():
                require(result["paired_warm_gate"], "Paired warm regression")
                limit = 0.98 if card == "a100" else 1.03
                require(result["cold_ratio"] <= limit, "Cold qualification failed")
    fixture_checks = {}
    for card, name in cohort["raw_k_logs"].items():
        text = files[name]
        count = sum(line.startswith("PASS ") for line in text.splitlines())
        require(
            count == 144 and "ALL SYNTHETIC GATES PASS" in text,
            "Incomplete raw-K oracle gate",
        )
        fixture_checks[card] = count
    for name in cohort["sanitizer_logs"]:
        text = files[name]
        marker = (
            "RACECHECK SUMMARY: 0 hazards"
            if name.endswith("racecheck.log")
            else "ERROR SUMMARY: 0 errors"
        )
        require(marker in text, "Raw-K sanitizer failed")
    transfer = json.loads(files["reproducer/fixture-source-transfer.json"])
    require(transfer["nonarray_executable_identical"], "Unqualified fixture executable")
    require(
        transfer["all_original_independent_arrays_retained"],
        "Missing independent arrays",
    )
    return {
        "default_promotion_qualified": True,
        "all_readme_leadership_established": False,
        "scope": cohort["scope"],
        "retained_members": len(files),
        "raw_k_matrix_predicates": fixture_checks,
        "cards": cards,
        "storage_boundary": bundle["experiment"]["scope_partition"],
    }


if __name__ == "__main__":
    report = replay()
    print(
        json.dumps(
            {
                "default_promotion_qualified": report["default_promotion_qualified"],
                "all_readme_leadership_established": False,
                "raw_k_matrix_predicates": report["raw_k_matrix_predicates"],
                "cards": {
                    card: {
                        method: {
                            key: record[key]
                            for key in (
                                "cold_means",
                                "cold_ratio",
                                "paired_warm_gate",
                                "reference_cold",
                            )
                        }
                        for method, record in measurements["methods"].items()
                    }
                    for card, measurements in report["cards"].items()
                },
            },
            indent=2,
        )
    )
