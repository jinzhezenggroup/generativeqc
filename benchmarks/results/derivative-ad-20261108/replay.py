"""Recheck retained derivative-migration parity and recompute endpoint ratios."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record


def inspect(samples: dict) -> tuple[list[dict], dict]:
    """Check numerical/work parity without imposing an artificial zero-noise gate."""
    if (samples.get("schema"), samples.get("schema_version")) != (
        "generativeqc.derivative-ad-samples",
        1,
    ):
        raise ValueError("unsupported derivative sample schema")
    errors = defaultdict(lambda: [[], [], None, None])
    rows = []

    def compare(
        name: str, actual: Any, reference: Any, atol: float, rtol: float = 0.0
    ) -> None:
        observed = block_error(actual, reference, atol=atol, rtol=rtol)
        if not observed["passed"]:
            raise ValueError(f"{name}: {observed}")
        bucket = errors[name]
        bucket[0].extend(np.asarray(actual).ravel().tolist())
        bucket[1].extend(np.asarray(reference).ravel().tolist())
        bucket[2:] = atol, rtol

    def work(actual: dict, reference: dict, keys: tuple[str, ...]) -> None:
        for key in keys:
            if actual.get(key) != reference.get(key):
                raise ValueError(f"work mismatch: {key}")

    def report(
        name: str,
        ratios: list[float],
        *,
        randomized: bool = False,
        blocks: list[float] | None = None,
    ) -> None:
        row = {
            "case": name,
            "samples": len(ratios),
            "median_ratio": statistics.median(ratios),
        }
        if randomized:
            generator = np.random.default_rng(20261108)
            values = np.asarray(ratios)
            medians = np.median(generator.choice(values, (5000, len(values))), axis=1)
            row["paired_bootstrap_95"] = np.quantile(medians, [0.025, 0.975]).tolist()
        if blocks is not None:
            row["counterbalanced_block_ratios"] = blocks
        rows.append(row)

    for case in samples["gfn2_cuda"]["cases"]:
        if len(case["pairs"]) != 64:
            raise ValueError("incomplete GFN2 cohort")
        for pair in [case["prime"], *case["pairs"]]:
            for reference, actual in zip(
                pair["baseline"]["samples"], pair["candidate"]["samples"], strict=True
            ):
                work(actual, reference, ("stage", "iterations"))
                compare("gfn2_energy", actual["energy"], reference["energy"], 5e-10)
                compare("gfn2_force", actual["forces"], reference["forces"], 5e-9)
        report(
            "cuda/gfn2/" + case["case"],
            [
                pair["candidate"]["seconds"] / pair["baseline"]["seconds"]
                for pair in case["pairs"]
            ],
            randomized=True,
        )

    nonlocal_records = [*samples["nonlocal_cuda"], samples["nonlocal_cpu"]]
    for record in nonlocal_records:
        for case in record["results"]:
            for pair in [case["prime"], *case["pairs"]]:
                for reference, actual in zip(
                    pair["baseline"]["stages"], pair["candidate"]["stages"], strict=True
                ):
                    work(
                        actual,
                        reference,
                        (
                            "stage",
                            "iterations",
                            "fock_builds",
                            "warm_start_used",
                            "pair_evaluations",
                            "tiles",
                        ),
                    )
                    compare(
                        "nonlocal_energy", actual["energy"], reference["energy"], 2e-10
                    )
                    if "features" in actual:
                        compare(
                            "nonlocal_features",
                            actual["features"],
                            reference["features"],
                            2e-9,
                            1e-10,
                        )
                    else:
                        work(actual, reference, ("features_count",))
                        compare(
                            "nonlocal_compacted_feature_error",
                            actual["maximum_feature_difference"],
                            0.0,
                            2e-9,
                        )
            report(
                record["device"] + "/" + case["name"],
                [
                    pair["candidate"]["seconds"] / pair["baseline"]["seconds"]
                    for pair in case["pairs"]
                ],
                randomized=True,
            )

    for group, expected in (("ks_cpu", 16), ("ks_cuda", 32), ("force_cpu", 32)):
        records = samples[group]
        if len(records) != expected:
            raise ValueError(f"incomplete {group} cohort")
        grouped = defaultdict(list)
        for name, reference in records.items():
            prefix, selection, round_number = name.rsplit("-", 2)
            if selection not in ("baseline", "master"):
                continue
            actual = records[prefix + "-candidate-" + round_number]
            if len(reference["samples"]) != 8 or len(actual["samples"]) != 8:
                raise ValueError("incomplete complete-endpoint worker")
            for left, right in zip(
                [reference["first_use"], *reference["samples"]],
                [actual["first_use"], *actual["samples"]],
                strict=True,
            ):
                if reference["kind"] == "cpks":
                    compare(
                        group + "_density_response",
                        right["density_solutions"],
                        left["density_solutions"],
                        2e-8,
                        2e-8,
                    )
                    compare(
                        group + "_scf_energy", right["energy"], left["energy"], 2e-9
                    )
                    work(right["scf"], left["scf"], ("iterations", "fock_builds"))
                    work(right, left, ("validation_actions", "dimension"))
                    residuals = [*left["true_residuals"], *right["true_residuals"]]
                    compare(
                        group + "_true_residual",
                        residuals,
                        np.zeros(len(residuals)),
                        1.01e-12,
                    )
                    for first, second in zip(
                        left["solves"], right["solves"], strict=True
                    ):
                        work(
                            second,
                            first,
                            (
                                "iterations",
                                "operator_actions",
                                "preconditioner_actions",
                                "converged",
                            ),
                        )
                        if not second["converged"] or not np.isfinite(
                            second["residual_norm"]
                        ):
                            raise ValueError("nonconverged response")
                else:
                    field = "observations" if group == "force_cpu" else "samples"
                    for first, second in zip(left[field], right[field], strict=True):
                        work(
                            second,
                            first,
                            ("stage", "iterations", "fock_builds", "warm_start_used"),
                        )
                        compare(
                            group + "_energy", second["energy"], first["energy"], 2e-9
                        )
                        compare(
                            group + "_force", second["forces"], first["forces"], 2e-7
                        )
            ratios = [
                right["seconds"] / left["seconds"]
                for left, right in zip(
                    reference["samples"], actual["samples"], strict=True
                )
            ]
            grouped[prefix].append(
                (
                    ratios,
                    statistics.median(right["seconds"] for right in actual["samples"])
                    / statistics.median(
                        left["seconds"] for left in reference["samples"]
                    ),
                )
            )
        for name, blocks in grouped.items():
            if len(blocks) != 2:
                raise ValueError("missing reversed-order process block")
            report(
                group + "/" + name,
                [ratio for block, _ in blocks for ratio in block],
                blocks=[ratio for _, ratio in blocks],
            )

    grouped = defaultdict(list)
    for round_number in (1, 2):
        baseline = samples["classical_cuda"][f"master-{round_number}"]
        candidate = samples["classical_cuda"][f"candidate-{round_number}"]
        if len(baseline) != 36 or len(candidate) != 36:
            raise ValueError("incomplete classical endpoint inventory")
        for reference, actual in zip(baseline, candidate, strict=True):
            work(
                actual,
                reference,
                (
                    "method",
                    "mode",
                    "atoms_per_system",
                    "systems",
                    "pair_candidates",
                    "triple_candidates",
                    "eeq_dimension",
                    "samples",
                    "repeats",
                ),
            )
            if (
                len(actual["sample_seconds"]) != 7
                or len(reference["sample_seconds"]) != 7
            ):
                raise ValueError("classical warm samples were not retained")
            compare("classical_energy_parity", actual["energy_error"], 0.0, 2e-10)
            compare("classical_force_parity", actual["gradient_error"], 0.0, 2e-9)
            compare(
                "classical_checksum", actual["checksum"], reference["checksum"], 1e-8
            )
            name = (
                f"cuda/{actual['method']}/{actual['atoms_per_system']}/{actual['mode']}"
            )
            grouped[name].append(actual["median_seconds"] / reference["median_seconds"])
    for name, ratios in grouped.items():
        report(name, ratios, blocks=ratios)

    quantitative = {
        name: block_error(actual, reference, atol=atol, rtol=rtol)
        for name, (actual, reference, atol, rtol) in errors.items()
    }
    return rows, quantitative


def main() -> None:
    """Authenticate the publication before replaying its scientific/work checks."""
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "publication.json").read_bytes())
    files = {
        entry["path"]: (directory / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    source = (directory / "source.patch.xz").read_bytes()
    evidence = load_publication_record(directory)
    if (
        hashlib.sha256(lzma.decompress(source)).hexdigest()
        != evidence["hashes"]["source"]
    ):
        raise ValueError("measured-source patch identity mismatch")
    samples = load_publication_record(directory, role="samples")
    rows, errors = inspect(samples)
    print(
        json.dumps(
            {"endpoints": rows, "block_errors": errors}, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
