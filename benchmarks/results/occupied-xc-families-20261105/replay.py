"""Replay source-bound E+F populations without a GPU, native library or PySCF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import lzma
import math
from functools import lru_cache
from pathlib import Path

from benchmarks._exact_fp64_evidence import load as load_exact
from benchmarks.occupied_xc_functionals import independent_gate, replay_records
from benchmarks.occupied_xc_proposals import check_vectors

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def original_transport(text: str) -> bytes:
    """Recreate the original writer's XZ bytes; cache by the complete text."""
    return lzma.compress(
        text.encode(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6
    )


def recover_originals(folder: Path = ROOT) -> dict[str, bytes]:
    """Authenticate complete original JSON/XZ and Git blobs without Git access."""
    receipt = json.loads((folder / "storage-receipt.json").read_text())
    bundle, texts = load_exact(folder)
    names = {"endpoints", "integration"}
    if (
        receipt["schema"] != "generativeqc.lossless-storage-receipt.v1"
        or receipt["representation"] != bundle["schema"]
        or receipt["float_count"] != bundle["float_count"]
        or receipt["original_xz_recipe"]
        != {"format": "xz", "check": "crc64", "preset": 6}
        or set(texts) != {name + ".json" for name in names}
        or len(receipt["originals"]) != len(names)
        or {row["path"] for row in receipt["originals"]}
        != {name + ".json.xz" for name in names}
        or len(receipt["stored_streams"]) != 3
        or {row["path"] for row in receipt["stored_streams"]}
        != {"fp64.xz", "receipts.json.xz", "sha256.txt"}
    ):
        raise ValueError("original capsule storage contract changed")
    for row in receipt["stored_streams"]:
        data = (folder / row["path"]).read_bytes()
        if (
            len(data) != row["bytes"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ValueError("stored stream receipt changed")
    originals = {}
    for row in receipt["originals"]:
        if row["text_path"] != row["path"].removesuffix(".xz"):
            raise ValueError("original text identity changed")
        text = texts[row["text_path"]]
        raw = text.encode()
        if (
            len(raw) != row["text_bytes"]
            or hashlib.sha256(raw).hexdigest() != row["text_sha256"]
        ):
            raise ValueError("original JSON bytes changed")
        packed = original_transport(text)
        blob = b"blob " + str(len(packed)).encode() + b"\0" + packed
        if (
            len(packed) != row["bytes"]
            or hashlib.sha256(packed).hexdigest() != row["sha256"]
            or hashlib.sha1(blob, usedforsecurity=False).hexdigest()
            != row["git_blob_sha1"]
        ):
            raise ValueError("original compressed transport changed")
        originals[row["path"]] = packed
    return originals


def load_capsules(folder: Path = ROOT) -> dict[str, dict]:
    """Load both complete measured capsules after lossless storage verification."""
    return {
        name.removesuffix(".json.xz"): json.loads(lzma.decompress(data))
        for name, data in recover_originals(folder).items()
    }


def source_work(observation: dict) -> int:
    """Count actual ERI production, not just applications of a resident source."""
    precision = observation["precision"]
    if not precision["complete"] or not precision["operator_inventory_complete"]:
        raise ValueError("incomplete operator inventory")
    for operator in precision["operators"]:
        if (
            any(
                operator[key] != "fp64"
                for key in ("storage", "compute", "accumulation", "reduction")
            )
            or operator["arithmetic_mode"] != "strict"
        ):
            raise ValueError("timed population changed precision")
    return sum(
        operator["count"]
        for operator in precision["operators"]
        if operator["name"] == "coulomb_recurrence"
    )


def audit_diagnostic(
    record: dict, frames: list[list[int]], *, require_proposal: bool = True
) -> None:
    """Require independent forces and a fresh stored-D final comparison tail."""
    if not record["completed"] or not record["traced"]:
        raise ValueError("incomplete admission diagnostic")
    if len(frames) != 4 or len(record["observations"]) != 4:
        raise ValueError("incomplete diagnostic solves")
    for observation, generations in zip(record["observations"], frames, strict=True):
        independent_gate(
            observation,
            record["oracle"]["observations"][
                int(observation["phase"].startswith("moved"))
            ],
            record["atoms"],
        )
        source_work(observation)
        if (
            not generations
            or generations[-1]
            or not observation["iterations"]
            <= len(generations)
            <= observation["fock_builds"]
        ):
            raise ValueError("provisional or missing final target")
        if any(generations):
            last = max(
                index for index, generation in enumerate(generations) if generation
            )
            if len(generations) - last - 1 < 2:
                raise ValueError("missing strict energy comparison tail")
    if require_proposal and not any(
        generation for solve in frames for generation in solve
    ):
        raise ValueError("diagnostic never admitted a proposal")


def replay(capsule: dict) -> dict:
    """Keep wins, neutral cases and regressions under the same numerical gates."""
    if capsule["schema"] != "generativeqc.occupied-xc-families.v1":
        raise ValueError("unsupported family capsule")
    summaries = {}
    identity = ("source_commit", "source_identity", "library_sha256")
    for name, records in capsule["populations"].items():
        for record in records:
            if any(record[key] != capsule[key] for key in identity):
                raise ValueError("measured native image changed")
            if record["host"] != capsule["population_hosts"][name]:
                raise ValueError("timing population changed host")
            key = f"{record['method']}-{record['atoms']}-{record['spin']}"
            if record["atoms"] == 48:
                if key not in capsule["second_references"]:
                    raise ValueError("missing second independent reference")
                oracle = capsule["second_references"][key]
                if not oracle["full_fock"] or oracle["grid"]["level"] != 6:
                    raise ValueError("second independent reference scope changed")
                if oracle["geometries"] != record["oracle"]["geometries"]:
                    raise ValueError("second independent reference geometry changed")
                for low, high in zip(
                    record["oracle"]["observations"],
                    oracle["observations"],
                    strict=True,
                ):
                    energy, force = independent_gate(low, high, record["atoms"])
                    rms = math.sqrt(
                        sum(
                            (left - right) ** 2
                            for low_row, high_row in zip(
                                low["forces"], high["forces"], strict=True
                            )
                            for left, right in zip(low_row, high_row, strict=True)
                        )
                        / (3 * record["atoms"])
                    )
                    if energy > 5e-6 * record["atoms"] or force > 5e-5 or rms > 1.5e-5:
                        raise ValueError("independent grid stability gate failed")
                for row in record["observations"]:
                    independent_gate(
                        row,
                        oracle["observations"][int(row["phase"].startswith("moved"))],
                        record["atoms"],
                    )
            for row in record["observations"]:
                source_work(row)
        summaries[name] = replay_records(records)
        for index, phase in enumerate(summaries[name]["phases"].values()):
            phase["source_recurrence_work"] = {
                label: [
                    source_work(record["observations"][index])
                    for record in records
                    if record["occupied_xc_env"] == control
                ]
                for label, control in (("matrix", "0"), ("default", None))
            }
    for name, diagnostic in capsule["diagnostics"].items():
        record, frames = diagnostic["receipt"], diagnostic["frames"]
        if any(record[key] != capsule[key] for key in identity):
            raise ValueError("diagnostic native image changed")
        audit_diagnostic(record, frames)
    before, after = capsule["source_handoff"]
    if any(
        before["receipt"][key] != capsule["before_source_image"][key]
        for key in identity
    ):
        raise ValueError("pre-fix diagnostic native image changed")
    for diagnostic in (before, after):
        audit_diagnostic(diagnostic["receipt"], diagnostic["frames"])
    if any(after["receipt"][key] != capsule[key] for key in identity):
        raise ValueError("handoff diagnostic native image changed")
    for index, (old, new) in enumerate(
        zip(
            before["receipt"]["observations"],
            after["receipt"]["observations"],
            strict=True,
        )
    ):
        check_vectors(new, old)
        if (old["iterations"], old["fock_builds"]) != (
            new["iterations"],
            new["fock_builds"],
        ):
            raise ValueError("source-only diagnostic changed SCF work")
        if source_work(old) != (2 if index == 0 else 1) * source_work(new):
            raise ValueError("redundant source production was not removed")
    history = capsule["history_diagnosis"]
    for record in history["records"]:
        if any(record[key] != history[key] for key in identity):
            raise ValueError("history diagnostic native image changed")
    replay_records(history["records"])
    discontinued = {}
    for name, records in capsule["pre_incremental_populations"].items():
        for record in records:
            if any(
                record[key] != capsule["before_incremental_image"][key]
                for key in identity
            ):
                raise ValueError("pre-incremental timing native image changed")
        discontinued[name] = replay_records(records)
    for stage, pairs in capsule["incremental_handoff"].items():
        expected = capsule if stage == "after" else capsule["before_incremental_image"]
        for control, diagnostic in pairs.items():
            record = diagnostic["receipt"]
            if any(record[key] != expected[key] for key in identity):
                raise ValueError("incremental diagnostic native image changed")
            audit_diagnostic(
                record, diagnostic["frames"], require_proposal=control != "0"
            )
        matrix, default = pairs["0"], pairs["unset"]
        for old, new in zip(
            matrix["receipt"]["observations"],
            default["receipt"]["observations"],
            strict=True,
        ):
            check_vectors(new, old)
        baseline = matrix["trace_work"][0]["structured_basis"]
        proposal = default["trace_work"][0]["structured_basis"]
        if (stage == "before" and proposal <= baseline) or (
            stage == "after" and proposal != baseline
        ):
            raise ValueError("J/K handoff did not preserve basis-transform work")
    original = original_scaling(capsule, capsule["original_r2scan3c"])
    return {"current": summaries, "pre_incremental": discontinued, "original": original}


def original_scaling(capsule: dict, records: dict) -> dict:
    """Reuse checksum-bound original-workload references without importing CUDA."""
    references = ROOT.parent / "readme-final-20261022"
    for name, expected in capsule["original_reference_sha256"].items():
        if hashlib.sha256((references / name).read_bytes()).hexdigest() != expected:
            raise ValueError("original independent reference changed")
    script = ROOT.parent / "occupied-xc-default-proposals-20261105/replay.py"
    spec = importlib.util.spec_from_file_location("original_default_replay", script)
    if spec is None or spec.loader is None:
        raise ValueError("missing original independent qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scaling(dict(capsule, scaling_records=records))


def replay_integration(supplement: dict, baseline: dict) -> dict:
    """Qualify the master integration as a separate image, never pooled timings."""
    identity = ("source_commit", "source_identity", "library_sha256")
    if supplement["schema"] != "generativeqc.occupied-xc-integration.v1":
        raise ValueError("unsupported integration capsule")
    if (
        hashlib.sha256(
            original_transport(json.dumps(baseline, separators=(",", ":")) + "\n")
        ).hexdigest()
        != supplement["baseline_capsule_sha256"]
        or any(supplement["baseline_image"][key] != baseline[key] for key in identity)
        or supplement["original_reference_sha256"]
        != baseline["original_reference_sha256"]
    ):
        raise ValueError("integration reference provenance changed")
    records = supplement["m06_records"]
    if len(records) != 6:
        raise ValueError("incomplete integration family population")
    expected_oracle = baseline["populations"]["large-m06-2x-rks"][0]["oracle"]
    second = baseline["second_references"]["m06-2x-48-rks"]
    if (
        not second["full_fock"]
        or second["grid"]["level"] != 6
        or second["geometries"] != expected_oracle["geometries"]
    ):
        raise ValueError("integration second reference scope changed")
    for record in records:
        if (
            any(record[key] != supplement[key] for key in identity)
            or record["host"] != supplement["host"]
            or (record["method"], record["atoms"], record["spin"])
            != ("m06-2x", 48, "rks")
            or record["oracle"] != expected_oracle
        ):
            raise ValueError("integration family image, host or oracle changed")
        for row in record["observations"]:
            source_work(row)
            independent_gate(
                row,
                second["observations"][int(row["phase"].startswith("moved"))],
                record["atoms"],
            )
    family = replay_records(records)
    for record in supplement["original_records"].values():
        if record["environment"]["host"]["node"] != supplement["host"]:
            raise ValueError("integration original workload changed host")
    original = original_scaling(supplement, supplement["original_records"])
    return {"m06": family, "original": original}


def corruption_probes(capsule: dict) -> None:
    """Confirm acceptance and provenance gates remain active under python -O."""
    population = next(iter(capsule["populations"]))
    for kind in (
        "image",
        "host",
        "force",
        "control",
        "incomplete",
        "original",
        "source_work",
        "jk",
        "reference",
        "tail",
    ):
        damaged = copy.deepcopy(capsule)
        record = damaged["populations"][population][-1]
        if kind == "image":
            record["library_sha256"] = "0" * 64
        elif kind == "host":
            record["host"] = "another-node"
        elif kind == "force":
            record["observations"][0]["forces"][0][0] += 1.0
        elif kind == "control":
            record["occupied_xc_env"] = "1"
        elif kind == "incomplete":
            record["completed"] = False
        elif kind == "original":
            damaged["original_r2scan3c"]["r0-unset"]["samples"][0]["forces"][0][0] += 1
        elif kind == "source_work":
            damaged["source_handoff"][1]["receipt"]["observations"][0]["precision"][
                "operators"
            ][0]["count"] += 1
        elif kind == "jk":
            damaged["incremental_handoff"]["after"]["unset"]["trace_work"][0][
                "structured_basis"
            ] += 1
        elif kind == "reference":
            damaged["second_references"].clear()
        else:
            next(iter(damaged["diagnostics"].values()))["frames"][0][-1] = 1
        try:
            replay(damaged)
        except ValueError:
            continue
        raise ValueError(f"corruption probe escaped: {kind}")


def integration_corruption_probes(supplement: dict, baseline: dict) -> None:
    """Exercise the new image's independent gates even with assertions removed."""
    for kind in (
        "image",
        "host",
        "force",
        "control",
        "incomplete",
        "original",
        "original_host",
        "inventory",
        "reference",
        "baseline",
    ):
        damaged = copy.deepcopy(supplement)
        record = damaged["m06_records"][-1]
        if kind == "image":
            record["library_sha256"] = "0" * 64
        elif kind == "host":
            record["host"] = "another-node"
        elif kind == "force":
            record["observations"][0]["forces"][0][0] += 1
        elif kind == "control":
            record["occupied_xc_env"] = "1"
        elif kind == "incomplete":
            damaged["m06_records"].pop()
        elif kind == "original":
            damaged["original_records"]["r0-unset"]["samples"][0]["forces"][0][0] += 1
        elif kind == "original_host":
            damaged["original_records"]["r0-unset"]["environment"]["host"]["node"] = (
                "another-node"
            )
        elif kind == "inventory":
            record["observations"][0]["precision"]["operator_inventory_complete"] = (
                False
            )
        elif kind == "reference":
            damaged["original_reference_sha256"] = {}
        else:
            damaged["baseline_image"]["source_identity"] = "0" * 64
        try:
            replay_integration(damaged, baseline)
        except ValueError:
            continue
        raise ValueError(f"integration corruption probe escaped: {kind}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-originals", type=Path)
    arguments = parser.parse_args()
    if arguments.restore_originals is not None:
        originals = recover_originals()
        arguments.restore_originals.mkdir(parents=True, exist_ok=False)
        for name, data in originals.items():
            with (arguments.restore_originals / name).open("xb") as stream:
                stream.write(data)
        raise SystemExit(0)
    capsules = load_capsules()
    capsule = capsules["endpoints"]
    result = replay(capsule)
    corruption_probes(capsule)
    supplement = capsules["integration"]
    result["integration"] = replay_integration(supplement, capsule)
    integration_corruption_probes(supplement, capsule)
    print(json.dumps(result, indent=2))
