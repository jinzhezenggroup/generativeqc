"""Replay final native DFT qualification, not the earlier interposer pilots."""

from __future__ import annotations

import json
import math
import re
from itertools import product
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.dft_streaming_qualification import qualify_native


def raw_k_predicates(text: str) -> int:
    """Require one independent result for every retained fixture combination."""
    expected = set(
        product(
            ("dppp", "pppp", "dpds", "ddps", "dpdp", "ddpp"),
            ("cartesian", "coincident", "reversed_pairs"),
            ("private", "shared"),
            ("0", "1"),
            ("0", "1"),
        )
    )
    observed = []
    for line in text.splitlines():
        if not line.startswith("PASS "):
            continue
        match = re.fullmatch(
            r"PASS ([a-z]+)_(cartesian|coincident|reversed_pairs)_(private|shared)"
            r" graph=([01]) replay=([01]) max_abs=(\S+)",
            line,
        )
        require(match is not None, "Malformed raw-K oracle predicate")
        require(
            math.isfinite(float(match[6])) and float(match[6]) >= 0,
            "Nonfinite or negative raw-K oracle error",
        )
        observed.append(match.groups()[:5])
    require(
        len(observed) == len(expected)
        and set(observed) == expected
        and "ALL SYNTHETIC GATES PASS" in text,
        "Incomplete raw-K oracle gate",
    )
    return len(observed)


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
    require(
        cohort["raw_k_logs"]
        == {
            "a100": "weighted-pilot-a100/raw-k-numerics.log",
            "5090": "raw-k-science/fixture-5090-16846/numerics.log",
        },
        "Changed raw-K device provenance",
    )
    sanitizers = cohort["sanitizer_logs"]
    require(
        len(sanitizers) == 3
        and set(sanitizers)
        == {
            f"raw-k-science/fixture-5090-16846/{kind}.log"
            for kind in ("memcheck", "racecheck", "synccheck")
        },
        "Changed raw-K sanitizer provenance",
    )
    fixture_checks = {}
    for card, name in cohort["raw_k_logs"].items():
        fixture_checks[card] = raw_k_predicates(files[name])
    for name in cohort["sanitizer_logs"]:
        text = files[name]
        marker = (
            "RACECHECK SUMMARY: 0 hazards"
            if name.endswith("racecheck.log")
            else "ERROR SUMMARY: 0 errors"
        )
        summaries = re.findall(
            r"(?:ERROR|RACECHECK) SUMMARY: (\d+) (?:errors|hazards)", text
        )
        require(
            marker in text and summaries and all(int(n) == 0 for n in summaries),
            "Raw-K sanitizer failed",
        )
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
