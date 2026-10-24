"""Read immutable evidence blobs and recompute historical errors, offline only."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAX_RAW = 16 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
XZ_MEMORY = 96 * 1024 * 1024


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_path(name: str) -> PurePosixPath:
    require(isinstance(name, str) and name != "", "Empty path")
    p = PurePosixPath(name)
    require(
        not p.is_absolute()
        and all(x not in ("", ".", "..") for x in name.split("/"))
        and "\\" not in name
        and ":" not in name
        and all(ord(c) >= 32 and ord(c) != 127 for c in name),
        "Unsafe relative path",
    )
    return p


def json_bytes(raw: bytes | str) -> dict:
    def pairs(items: list) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def unpack(raw: bytes, limit: int = MAX_EXPANDED) -> bytes:
    require(len(raw) <= MAX_RAW, "Compressed input bound")
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=XZ_MEMORY)
    value = decoder.decompress(raw, max_length=limit + 1)
    require(len(value) <= limit, "Expanded input bound")
    require(decoder.eof and not decoder.unused_data, "Truncated/trailing XZ data")
    return value


def raw_records(bundle: dict) -> dict[str, bytes]:
    """Validate without writing paths or importing any historical executable."""
    result = {}
    entries = bundle["files"]
    if isinstance(entries, dict):
        require(set(entries) == set(bundle["file_sha256"]), "Raw inventory mismatch")
        for name, metadata in bundle.get("file_metadata", {}).items():
            require(name in entries, "Metadata for unknown raw file")
            require(
                metadata["sha256"] == bundle["file_sha256"][name],
                "Conflicting raw hashes",
            )
        entries = [
            {
                **bundle.get("file_metadata", {}).get(k, {}),
                "path": k,
                "utf8": v,
                "sha256": bundle["file_sha256"][k],
            }
            for k, v in entries.items()
        ]
    for entry in entries:
        name = entry["path"]
        safe_path(name)
        require(name not in result, "Duplicate raw path")
        raw = entry["utf8"].encode("utf-8")
        require(sha(raw) == entry["sha256"], "Raw SHA256 mismatch")
        require(len(raw) == entry.get("bytes", len(raw)), "Raw byte count mismatch")
        mode = entry.get("original_mode", entry.get("mode", "UNAVAILABLE"))
        require(
            mode == "UNAVAILABLE"
            or (isinstance(mode, int) and 0 <= mode <= 0o777)
            or (isinstance(mode, str) and re.fullmatch(r"100[0-7]{3}", mode)),
            "Invalid original mode",
        )
        result[name] = raw
    return result


class Evidence:
    def __init__(self, sources: dict, root: Path = ROOT) -> None:
        self.root = root
        self.entries = {}
        self.cache = {}
        for entry in sources["blobs"]:
            safe_path(entry["path"])
            require(
                re.fullmatch(r"[0-9a-f]{40}", entry["commit"]), "Fixed OID required"
            )
            key = (entry["commit"], entry["path"])
            require(key not in self.entries, "Duplicate source locator")
            require(entry["mode"] in ("100644", "100755"), "Regular blobs only")
            self.entries[key] = entry

    def raw(self, commit: str, path: str) -> bytes:
        key = (commit, path)
        if key not in self.cache:
            entry = self.entries[key]
            if "snapshot" in entry:
                saved = entry["snapshot"]
                safe_path(saved["path"])
                target = HERE / saved["path"]
                require(
                    not target.is_symlink() and target.resolve().is_relative_to(HERE),
                    "Unsafe prepared snapshot",
                )
                require(
                    target.stat().st_size == saved["bytes"] <= MAX_RAW,
                    "Prepared snapshot size bound",
                )
                packed = target.read_bytes()
                require(
                    len(packed) == saved["bytes"] and sha(packed) == saved["sha256"],
                    "Prepared snapshot byte mismatch",
                )
                raw = unpack(packed)
                require(
                    len(raw) == entry["bytes"] and sha(raw) == entry["sha256"],
                    "Prepared contract hash mismatch",
                )
                git_blob = hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw,
                    usedforsecurity=False,
                ).hexdigest()
                require(
                    git_blob == entry["blob"] and entry["mode"] == "100644",
                    "Prepared contract Git identity mismatch",
                )
                self.cache[key] = raw
                return raw
            spec = f"{commit}:{path}"
            size = int(
                subprocess.check_output(["git", "cat-file", "-s", spec], cwd=self.root)
            )
            require(
                size == entry["bytes"] and size <= MAX_RAW, "Git blob size mismatch"
            )
            raw = subprocess.check_output(
                ["git", "cat-file", "blob", spec], cwd=self.root
            )
            row = (
                subprocess.check_output(
                    ["git", "ls-tree", commit, "--", path], cwd=self.root
                )
                .decode()
                .split()
            )
            require(
                row[:3] == [entry["mode"], "blob", entry["blob"]],
                "Git mode/blob mismatch",
            )
            require(sha(raw) == entry["sha256"], "Git byte SHA256 mismatch")
            self.cache[key] = raw
        return self.cache[key]

    def data(self, commit: str, path: str) -> dict:
        raw = self.raw(commit, path)
        return json_bytes(unpack(raw) if path.endswith(".xz") else raw)

    def verify(self) -> None:
        # Batch immutable reads to avoid hundreds of Windows process launches.
        snapshots = [key for key, entry in self.entries.items() if "snapshot" in entry]
        for commit, path in snapshots:
            self.raw(commit, path)
        keys = [key for key in self.entries if key not in snapshots]
        requests = "".join(f"{c}:{p}\n" for c, p in keys).encode()
        headers = subprocess.run(
            ["git", "cat-file", "--batch-check"],
            input=requests,
            capture_output=True,
            check=True,
            cwd=self.root,
        ).stdout.splitlines()
        require(len(headers) == len(keys), "Git batch inventory mismatch")
        sizes = []
        for key, header in zip(keys, headers, strict=True):
            row = header.decode().split()
            entry = self.entries[key]
            require(
                len(row) == 3 and row[:2] == [entry["blob"], "blob"],
                "Git batch blob mismatch",
            )
            size = int(row[2])
            require(
                size == entry["bytes"] and 0 <= size <= MAX_RAW, "Git batch size bound"
            )
            sizes.append(size)
        require(sum(sizes) <= 256 * 1024 * 1024, "Git inventory aggregate bound")
        trees = {}
        for commit in {c for c, _ in keys}:
            paths = [p for c, p in keys if c == commit]
            rows = subprocess.check_output(
                ["git", "ls-tree", "-r", commit, "--", *paths], cwd=self.root
            )
            trees[commit] = {}
            for row in rows.decode().splitlines():
                meta, path = row.split("\t")
                trees[commit][path] = meta.split()
        payload = subprocess.run(
            ["git", "cat-file", "--batch"],
            input=requests,
            capture_output=True,
            check=True,
            cwd=self.root,
        ).stdout
        offset = 0
        verified = {}
        for key, header, size in zip(keys, headers, sizes, strict=True):
            entry = self.entries[key]
            end = payload.index(b"\n", offset)
            require(payload[offset:end] == header, "Git batch header changed")
            offset = end + 1
            raw = payload[offset : offset + size]
            require(
                len(raw) == size and sha(raw) == entry["sha256"],
                "Git batch byte mismatch",
            )
            require(
                trees[key[0]][key[1]] == [entry["mode"], "blob", entry["blob"]],
                "Git batch mode mismatch",
            )
            offset += size
            require(
                payload[offset : offset + 1] == b"\n", "Git batch boundary mismatch"
            )
            offset += 1
            verified[key] = raw
        require(offset == len(payload), "Git batch trailing data")
        self.cache.update(verified)


def coefficients() -> dict[str, float]:
    low = -1 / math.expm1(1.63)
    return {
        "low/df-ccsd(t)": 1,
        "low/mp2": -8 / 19 - 1,
        "low/rhf": low + 8 / 19,
        "high/mp2": 27 / 19,
        "high/rhf": 1 - low - 27 / 19,
    }


def finite(values: Iterable[float]) -> None:
    require(
        all(isinstance(x, (float, int)) and math.isfinite(x) for x in values),
        "Nonfinite numeric input",
    )


def reference(record: dict) -> dict[str, float]:
    require(record["status"] == "completed", "Incomplete oracle")
    low, high = record["stages"]
    require([low["cardinal"], high["cardinal"]] == [2, 3], "Wrong cardinal pair")
    for stage in (low, high):
        require(
            stage["status"] == "completed" and stage["rhf"]["converged"],
            "Unconverged oracle",
        )
        require(
            stage["mp2"]["frozen"] == 0 and stage["mp2"]["reference_df_shared"],
            "Wrong MP2 model",
        )
    require(
        low["ccsd"]["converged"] and low["ccsd"]["reference_df_shared"],
        "Wrong CC model",
    )
    values = {
        "low/rhf": low["rhf"]["energy"],
        "low/mp2": low["mp2"]["energy"],
        "low/df-ccsd(t)": low["triples"]["energy"],
        "high/rhf": high["rhf"]["energy"],
        "high/mp2": high["mp2"]["energy"],
    }
    values["composite"] = math.fsum(coefficients()[k] * v for k, v in values.items())
    finite(values.values())
    require(
        all(abs(v - record["energies"][k]) < 5e-12 for k, v in values.items()),
        "Oracle algebra mismatch",
    )
    # Different summation orders can change the last bit of a large energy.
    # Finite differences must use the retained raw composite after checking it.
    values["composite"] = record["energies"]["composite"]
    return values


def native(point: dict, protocol: dict, source: dict) -> tuple[dict, dict]:
    record = point["native"]
    require(
        point["native_outcome"]["exit_code"] == 0 and record["status"] == "completed",
        "Native outcome",
    )
    require(
        record["protocol"] == protocol
        and record["library_sha256"] == source["library_sha256"],
        "Native cohort mismatch",
    )
    require(len(record["samples"]) == 1, "Historical independent sample count")
    sample = record["samples"][0]
    require(
        sample["converged"] and sample["executed_backend"] == "cuda",
        "Native convergence/backend",
    )
    energies, forces = {}, {}
    for item in sample["components"]:
        require(
            item["basis_identity"] in protocol["orbital_identities"], "Unknown basis"
        )
        key = (
            (
                "low"
                if item["basis_identity"] == protocol["orbital_identities"][0]
                else "high"
            )
            + "/"
            + item["method"]
        )
        require(
            key not in energies and item["result"]["converged"],
            "Duplicate/unconverged component",
        )
        energies[key] = item["result"]["energy"]
        forces[key] = item["result"]["forces"]
    require(set(energies) == set(coefficients()), "Missing constituent")
    require(
        abs(
            math.fsum(coefficients()[k] * v for k, v in energies.items())
            - sample["energy"]
        )
        < 5e-12,
        "Native algebra",
    )
    energies["composite"], forces["composite"] = sample["energy"], sample["forces"]
    finite(energies.values())
    for vector in forces.values():
        require(
            len(vector) == len(protocol["atoms_bohr"])
            and all(len(row) == 3 for row in vector),
            "Force shape",
        )
        finite(x for row in vector for x in row)
    return energies, forces


def replay(sources: dict, coverage: dict) -> dict:
    evidence = Evidence(sources)
    evidence.verify()
    families = sources["families"]
    prepared = sources["prepared_protocol_contract"]
    contract = evidence.data(prepared["commit"], prepared["path"])
    require(
        contract["schema"] == "generativeqc.h100-paper-protocol.v1",
        "Owner protocol schema mismatch",
    )
    cells = {cell["id"]: cell for cell in contract["cells"]}
    require(len(cells) == len(contract["cells"]) == 11, "Owner protocol cell inventory")
    require(
        {row["protocol_cell_id"] for row in coverage["h100_core"]} == set(cells),
        "Map/owner cell mismatch",
    )
    for row in coverage["h100_core"]:
        cell = cells[row["protocol_cell_id"]]
        canonical = json.dumps(
            cell["input"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        require(
            sha(canonical) == cell["input_sha256"] == row["protocol_input_sha256"],
            "Owner input digest mismatch",
        )
        require(row["protocol_arms"] == cell["arms"], "Owner arm mismatch")

    def data(family: str, name: str) -> dict:
        spec = families[family]
        return evidence.data(spec["commit"], spec["path"] + "/" + name)

    summary = data("native", "summary.json.xz")
    source = summary["qualification"]["source"]["candidate"]
    inputs = data("energy", "inputs.json")
    original_failure = data("energy", "failures/C1.json.xz")
    failure_outcome = data("energy", "failures/C1.outcome.json.xz")
    energy_rows = []
    for carbon, index in ((1, 0), (3, 8), (4, 48), (5, 183), (6, 293), (7, 5604)):
        bundle = data("energy", f"receipts/C{carbon}.json.xz")
        require(
            sha(bundle["receipt_json"].encode()) == bundle["receipt_sha256"],
            "Receipt hash",
        )
        record = json_bytes(bundle["receipt_json"])
        protocol = inputs["protocols"][str(carbon)]
        require(
            record["protocol"] == protocol
            and protocol["qm7_index_zero_based"] == index,
            "Geometry/model mismatch",
        )
        require(json_bytes(bundle["outcome_json"])["exit_code"] == 0, "Oracle outcome")
        for stage in record["stages"]:
            require(
                sha(bundle["logs"][str(stage["cardinal"])].encode())
                == stage["log_sha256"],
                "Oracle log mismatch",
            )
        ref = reference(record)
        for policy, series in (("standard", summary), ("fast", summary["fast"])):
            point = next(p for p in series["points"] if p["carbon"] == carbon)
            values, _ = native(point, protocol, source)
            for term, value in values.items():
                error = abs(value - ref[term])
                require(error <= 5e-8, "Energy gate FAIL")
                energy_rows.append(
                    {
                        "carbon": carbon,
                        "qm7_index": index,
                        "policy": policy,
                        "term": term,
                        "absolute_error_eh": error,
                    }
                )

    displaced = data("forces", "inputs.json.xz")
    collection = data("forces", "receipts.json.xz")
    require(
        set(collection["tasks"]) == set(displaced["tasks"])
        and len(collection["tasks"]) == 8,
        "Displacement inventory",
    )
    refs, raw_count, raw_bytes = {}, 0, 0
    for task, bundle in collection["tasks"].items():
        require(
            re.fullmatch(r"C[47]-h[01]-(minus|plus)", task) and bundle["task"] == task,
            "Task identity",
        )
        files = raw_records(bundle)
        require(
            set(files)
            == {
                "stdout.log",
                "outcome.json",
                "science/receipt.json",
                "science/basis-2.log",
                "science/basis-3.log",
            },
            "Incomplete raw task",
        )
        require(
            json_bytes(files["outcome.json"])["exit_code"] == 0, "Displaced outcome"
        )
        receipt = json_bytes(files["science/receipt.json"])
        require(
            receipt["task"] == displaced["tasks"][task], "Displaced geometry mismatch"
        )
        refs[task] = reference(receipt)
        raw_count += len(files)
        raw_bytes += sum(map(len, files.values()))
    projections, translations = [], []
    for carbon in (4, 7):
        protocol = inputs["protocols"][str(carbon)]
        for policy, series in (("standard", summary), ("fast", summary["fast"])):
            _, vectors = native(
                next(p for p in series["points"] if p["carbon"] == carbon),
                protocol,
                source,
            )
            for term, vector in vectors.items():
                translation = max(
                    abs(math.fsum(row[a] for row in vector)) for a in range(3)
                )
                require(translation <= 1e-6, "Translation gate FAIL")
                translations.append(
                    {
                        "carbon": carbon,
                        "policy": policy,
                        "term": term,
                        "residual_eh_per_bohr": translation,
                    }
                )
                for i, step in enumerate(displaced["steps_bohr"]):
                    minus, plus = f"C{carbon}-h{i}-minus", f"C{carbon}-h{i}-plus"
                    direction = displaced["tasks"][minus]["direction"]
                    projection = math.fsum(
                        x * y
                        for a, b in zip(vector, direction, strict=True)
                        for x, y in zip(a, b, strict=True)
                    )
                    derivative = (refs[plus][term] - refs[minus][term]) / (2 * step)
                    error = abs(projection + derivative)
                    require(error <= 5e-6, "Directional gate FAIL")
                    projections.append(
                        {
                            "carbon": carbon,
                            "policy": policy,
                            "term": term,
                            "step_bohr": step,
                            "error_eh_per_bohr": error,
                        }
                    )
    cpu = []
    for filename, case in (
        ("raw-h2.json.xz", "h2-dz-tz"),
        ("raw-water.json.xz", "water-algebra"),
    ):
        files = raw_records(data("cpu", filename))
        identity = json_bytes(files["identity.json"])
        ref = evidence.data(
            identity["source_commit"], f"tests/reference_data/cbs/{case}.json"
        )
        calls = [
            json_bytes(line) for line in files["native/endpoints.jsonl"].splitlines()
        ]
        base = next(
            c["result"]
            for c in calls
            if c["atoms"] == ref["atoms"]
            and c.get("result", {}).get("forces") is not None
        )
        require(
            base["converged"] and base["executed_backend"] == "cpu_reference",
            "CPU endpoint",
        )
        values = {
            "composite": base,
            **{c["roles"][0]: c["result"] for c in base["components"]},
        }
        for term, value in values.items():
            other = (
                ref["reference"]
                if term == "composite"
                else ref["reference"]["components"][term]
            )
            e = abs(value["energy"] - other["energy"])
            delta = [
                a - b
                for x, y in zip(value["forces"], other["forces"], strict=True)
                for a, b in zip(x, y, strict=True)
            ]
            finite([e, *delta])
            require(
                e <= 1e-9 and max(map(abs, delta)) <= 2e-6, "CPU reference gate FAIL"
            )
            cpu.append(
                {
                    "case": case,
                    "term": term,
                    "energy_error_eh": e,
                    "max_force_error_eh_per_bohr": max(map(abs, delta)),
                    "force_rms_eh_per_bohr": math.sqrt(
                        math.fsum(x * x for x in delta) / len(delta)
                    ),
                }
            )
    return {
        "schema": "generativeqc.paper-evidence-replay.v1",
        "offline_integrity_status": "PASS",
        "h100_core_status": "INCOMPLETE",
        "negative_records": [
            {
                "family": "energy",
                "input": "C1",
                "attempt": "initial",
                "status": original_failure["status"],
                "error": original_failure["error"],
                "outcome": failure_outcome,
                "locator": families["energy"]["path"]
                + "/failures/C1.json.xz@"
                + families["energy"]["commit"],
                "later_success_does_not_erase_failure": True,
            }
        ],
        "source_blob_count": len(evidence.entries),
        "native_source": source,
        "energy": {
            "gate_eh": 5e-8,
            "comparisons": energy_rows,
            "count": len(energy_rows),
            "max_error_eh": max(r["absolute_error_eh"] for r in energy_rows),
        },
        "directional": {
            "gate_eh_per_bohr": 5e-6,
            "comparisons": projections,
            "count": len(projections),
            "raw_files": raw_count,
            "raw_bytes": raw_bytes,
            "max_by_carbon": {
                str(c): max(
                    r["error_eh_per_bohr"] for r in projections if r["carbon"] == c
                )
                for c in (4, 7)
            },
        },
        "translation": {
            "gate_eh_per_bohr": 1e-6,
            "count": len(translations),
            "max_residual_eh_per_bohr": max(
                r["residual_eh_per_bohr"] for r in translations
            ),
        },
        "cpu_reference": cpu,
        "coverage": coverage,
        "performance": {
            "qualified_h100_pairs_imported": 0,
            "ratios": [],
            "reason": "No accepted H100 receipts imported; reference runtime is not a denominator",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with committed compact-report.json",
    )
    args = parser.parse_args()
    report = replay(
        json_bytes((HERE / "sources.json").read_bytes()),
        json_bytes((HERE / "coverage-status.json").read_bytes()),
    )
    if args.check:
        require(
            report == json_bytes((HERE / "compact-report.json").read_bytes()),
            "Stored report differs",
        )
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
