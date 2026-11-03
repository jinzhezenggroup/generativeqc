"""Authenticate retained receipts and recompute complete diagnostic endpoints."""
from pathlib import Path
import json
import statistics

import numpy as np

from benchmarks._exact_fp64_evidence import load
from benchmarks.readme_defaults import COMMON_GATES
from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_report import checked_sample


def compare(files: dict[str, str], prefix: str, identity: str) -> dict:
    """Check one immutable source and all cold/warm pairs against its oracle."""
    native, reference = (json.loads(files[prefix + name + ".json"]) for name in ("fitted", "reference"))
    proof = json.loads(files[prefix + "identity.json"])
    if proof["source_identity"] != identity:
        raise ValueError("diagnostic source identity differs")
    for record in (native, reference):
        if record["status"] != "complete" or len(record["samples"]) != 4:
            raise ValueError("incomplete population")
        for sample in record["samples"]:
            checked_sample(sample, record)
    if native["geometry_bohr"] != reference["geometry_bohr"]:
        raise ValueError("different input geometry")
    comparisons = []
    for first, second in zip(native["samples"], reference["samples"], strict=True):
        if (first["phase"], first["repeat"]) != (second["phase"], second["repeat"]):
            raise ValueError("different endpoint phases")
        if np.asarray(first["forces"]).shape != np.asarray(second["forces"]).shape:
            raise ValueError("different force shapes")
        result = errors(first["energy"], first["forces"], second, native["atoms"])
        if not result["passed"]:
            raise ValueError("paired independent observable gate failed")
        comparisons.append(result)
    warm = {name: statistics.median(sample["seconds"] for sample in record["samples"][1:]) for name, record in (("native", native), ("reference", reference))}
    return {"source_identity": identity, "comparisons": comparisons, "diagnostic_warm_seconds": warm, "native_over_reference": warm["native"] / warm["reference"]}


def main() -> None:
    """Report measured sources without transferring qualification to new builds."""
    bundle, files = load(Path(__file__).resolve().parent)
    populations = {
        "a100_generic_224641": ("bench-224641/", "f72dc560b8fc9330c78665dbc3242c2ea1169fd5ac713020256984b313e5484f"),
        "rtx5090_shell_15268": ("shell-lr-rank-15268-final/shell-lr-rank-15268/", "df3d9cbbe4f2fddc6b100cd0d4056fbdd36300b0cb65e19eb927418c14b92558"),
    }
    results = {name: compare(files, *values) for name, values in populations.items()}
    print(json.dumps({"scope": "diagnostic BUILD_TESTS=ON; no production/default qualification", "archives": len(bundle["input_archives"]), "gates": COMMON_GATES, "populations": results}, indent=2))


if __name__ == "__main__":
    main()
