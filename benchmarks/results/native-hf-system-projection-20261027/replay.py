"""Authenticate and replay the complete CPU progressive-HF system-projection qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "tools/generativeqc_validation/record.py").is_file():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "python"))
        break

from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record


def audit(samples: dict, reference: dict) -> dict:
    """Recompute all original gates from every warmup and measured observation."""
    protocol = samples["protocol"]
    assert (
        protocol["completed"] and protocol["samples"] == 21 and protocol["warmups"] == 3
    )
    assert protocol["orders"] == {
        "abba": ["a1", "b1", "b2", "a2"],
        "baab": ["b1", "a1", "a2", "b2"],
    }
    assert protocol["ratio_gate"] == 1.05
    assert protocol["cases"] == [r["case"] for r in reference["cases"]]
    assert protocol["geometries"] == reference["geometries"]
    assert reference["production_modules_imported"] is False
    gates = {"energy": 1e-8, "forces": 1e-6, "density": 1e-7}
    maximum = {k: 0.0 for k in gates}
    actual_values = {key: [] for key in gates}
    oracle_values = {key: [] for key in gates}
    publications = 0
    comparisons = []
    for order, slots in protocol["orders"].items():
        blocks = {slot: samples["blocks"][f"{order}-{slot}"] for slot in slots}
        for slot, block in blocks.items():
            variant = "baseline" if slot.startswith("a") else "candidate"
            assert (
                block["completed"]
                and block["variant"] == variant
                and block["slot"] == slot
            )
            assert block["affinity"] == protocol["affinity"]
            assert block["library_sha256"] == protocol["library_sha256"][variant]
            assert block["collector_sha256"] == protocol["collector_sha256"]
            assert (block["samples"], block["warmups"]) == (21, 3)
            assert [r["case"] for r in block["cases"]] == protocol["cases"]
            for case, ref in zip(block["cases"], reference["cases"], strict=True):
                assert (
                    len(case["observations"]) == 21
                    and len(case["warmup_observations"]) == 3
                )
                for item in case["warmup_observations"] + case["observations"]:
                    assert np.isfinite(item["seconds"]) and item["seconds"] > 0
                    assert len(item["phases"]) == len(ref["phases"]) == 2
                    for phase, expected in zip(
                        item["phases"], ref["phases"], strict=True
                    ):
                        assert (
                            phase["target_established"]
                            and phase["verification_status"] == "unverified"
                        )
                        assert phase["strict_cleanup"] == "not_required_fp64"
                        assert 0 <= phase["physical_residual_max"] <= 1e-10
                        owners = ["native_cpu_system" if variant == "candidate" else "native_cpu"]
                        assert phase["projection_owners"] == owners * (2 if case["case"]["method"] == "uhf" else 1)
                        for key, gate in gates.items():
                            actual = np.asarray(phase[key])
                            oracle = np.asarray(expected[key])
                            assert (
                                actual.shape == oracle.shape
                                and np.isfinite(actual).all()
                            )
                            error = float(np.max(np.abs(actual - oracle)))
                            maximum[key] = max(maximum[key], error)
                            actual_values[key].extend(actual.reshape(-1).tolist())
                            oracle_values[key].extend(oracle.reshape(-1).tolist())
                            assert error <= gate, (case["case"]["name"], key, error)
                        publications += 1
        for index, case in enumerate(protocol["cases"]):
            observations = [
                (blocks[s]["variant"], o)
                for s in slots
                for o in blocks[s]["cases"][index]["observations"]
            ]
            work = [
                [(p["iterations"], p["fock_builds"]) for p in o["phases"]]
                for _, o in observations
            ]
            work_passed = all(w == work[0] for w in work)
            populations = {
                v: [o["seconds"] for kind, o in observations if kind == v]
                for v in ("baseline", "candidate")
            }
            assert all(len(p) == 42 for p in populations.values())
            medians = {v: statistics.median(p) for v, p in populations.items()}
            ratio = medians["candidate"] / medians["baseline"]
            comparisons.append(
                {
                    "order": order,
                    "case": case["name"],
                    "median_seconds": medians,
                    "ratio": ratio,
                    "work_passed": work_passed,
                    "passed": work_passed and ratio <= 1.05,
                }
            )
    assert len(samples["blocks"]) == 8 and len(comparisons) == 8
    errors = {
        key: block_error(actual_values[key], oracle_values[key], atol=gate, rtol=0)
        for key, gate in gates.items()
    }
    return {
        "block_errors": errors,
        "publications": publications,
        "maximum_absolute_errors": maximum,
        "comparisons": comparisons,
        "all_gates_passed": all(r["passed"] for r in comparisons),
    }


def main() -> int:
    """Replay without loading a solver; optionally restore the measured runners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--extract-frozen", type=Path)
    args = parser.parse_args()
    root = args.directory
    samples = load_publication_record(root, role="samples", name="samples.json")
    reference = load_publication_record(root, role="input", name="reference.json")
    sources = load_publication_record(
        root, role="reproduction", name="frozen-sources.json"
    )
    manifest = json.loads((root / "publication.json").read_text())
    files = {
        row["path"]: (root / row["path"]).read_bytes() for row in manifest["files"]
    }
    validate_publication(manifest, files)
    restored = {}
    for name, entry in sources["files"].items():
        # The capsule stores text, never executable serialized Python objects.
        assert Path(name).name == name and name not in {".", ".."}
        raw = entry["text"].encode()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        restored[name] = raw
    protocol = samples["protocol"]
    assert (
        hashlib.sha256(restored["endpoint.py"]).hexdigest()
        == protocol["collector_sha256"]
    )
    assert (
        hashlib.sha256(lzma.decompress(files["reference.json.xz"])).hexdigest()
        == protocol["reference_sha256"]
    )
    assert (
        hashlib.sha256(lzma.decompress(files["candidate.patch.xz"])).hexdigest()
        == protocol["source"]["patch_sha256"]
    )
    result = audit(samples, reference)
    expected = load_publication_record(root, role="summary", name="summary.json")[
        "audit"
    ]
    assert result == expected, "derived summary differs from all-sample replay"
    if args.extract_frozen:
        args.extract_frozen.mkdir(parents=True, exist_ok=False)
        for name, raw in restored.items():
            (args.extract_frozen / name).write_bytes(raw)
    print(json.dumps(result, indent=2))
    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
