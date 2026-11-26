"""Package native receipts and recompute geometry statistics without a solver."""

import argparse
import hashlib
import json
import lzma
import math
import re
from pathlib import Path
from statistics import fmean

from run import (
    ANGSTROM_TO_BOHR,
    ATOMIC_NUMBER,
    PRELIMINARY_FMAX,
    TARGET_FMAX,
    JsonObject,
    validate,
)

ROOT = Path(__file__).resolve().parent
STAGES = ("rhf", "cc-pcvtz", "cc-pcvqz", "df-cc-pcvtz")
LITERATURE_GATES = {"bonds": 3e-4, "angles": 0.05}
SHARED_ROOT = re.compile(rb"/inspire/hdd/global_user/[^/]+/projects/GenerativeQC")


def read_json(path: Path) -> JsonObject:
    return json.loads(path.read_text(encoding="utf-8"))


def capture(raw: Path) -> JsonObject:
    """Record all available stage files, including interrupted/failed requests."""
    result = {"campaign": read_json(raw / "campaign.json"), "cases": {}}
    for case in sorted(raw.iterdir()):
        if not case.is_dir():
            continue
        result["cases"][case.name] = files(case)
        for name in STAGES:
            stage = case / name
            if stage.is_dir():
                result["cases"][case.name][name] = files(stage)
    marker = raw / "campaign-complete.json"
    result["complete"] = read_json(marker) if marker.is_file() else None
    return result


def files(directory: Path) -> JsonObject:
    content = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            original = path.read_bytes()
            published = SHARED_ROOT.sub(b"${GQC_SHARED}", original)
            content[path.name] = {
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "sha256": hashlib.sha256(published).hexdigest(),
                "utf8": published.decode("utf-8"),
            }
    return content


def decode(stage: JsonObject, filename: str) -> str:
    entry = stage[filename]
    data = entry["utf8"].encode("utf-8")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Receipt hash mismatch: {filename}")
    return data.decode("utf-8")


def difference(a: list[float], b: list[float]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def parameter(xyz: list[list[float]], indices: list[int]) -> float:
    if len(indices) == 2:
        return math.dist(xyz[indices[0]], xyz[indices[1]]) / ANGSTROM_TO_BOHR
    i, j, k = indices
    arm1 = difference(xyz[i], xyz[j])
    arm2 = difference(xyz[k], xyz[j])
    cosine = math.fsum(x * y for x, y in zip(arm1, arm2, strict=True))
    cosine /= math.hypot(*arm1) * math.hypot(*arm2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def stats(entries: list[JsonObject]) -> JsonObject:
    if not entries:
        return {"n": 0, "mae": None, "max_abs": None, "worst": None}
    worst = max(entries, key=lambda row: abs(row["error"]))
    return {
        "n": len(entries),
        "mae": fmean(abs(row["error"]) for row in entries),
        "max_abs": abs(worst["error"]),
        "worst": {key: worst[key] for key in ("molecule", "label", "error")},
    }


def stage_outcome(
    raw: JsonObject, case: JsonObject, basis: str
) -> tuple[JsonObject, list[list[float]] | None]:
    if not raw:
        return {"status": "not_attempted"}, None
    for name in raw:
        decode(raw, name)
    if "receipt.json" not in raw:
        return {"status": "interrupted"}, None
    receipt = json.loads(decode(raw, "receipt.json"))
    outcome = {
        key: receipt[key] for key in ("status", "wall_seconds", "returncode", "error")
    }
    if receipt["status"] != "converged":
        outcome["stderr"] = decode(raw, "stderr.txt")
        return outcome, None

    request = json.loads(decode(raw, "request.json"))["argv"]
    method, basis_arg, backend = (
        request[request.index(key) + 1] for key in ("--method", "--basis", "--backend")
    )
    expected = (
        ("rhf", backend, PRELIMINARY_FMAX)
        if basis == "rhf"
        else (
            "df-rccsd(t)" if basis.startswith("df-") else "rccsd(t)",
            "cuda",
            TARGET_FMAX,
        )
    )
    if (method, backend) != expected[:2] or backend not in ("cpu", "cuda"):
        raise ValueError(f"{case['id']}: unexpected method/backend for {basis}")
    expected_basis = (
        "cc-pvdz.json" if basis == "rhf" else f"{basis.removeprefix('df-')}.json"
    )
    if Path(basis_arg).name != expected_basis:
        raise ValueError(f"{case['id']}: unexpected basis for {basis}")
    if basis.startswith("df-"):
        auxiliary = Path(request[request.index("--auxiliary-basis") + 1]).name
        if auxiliary != "cc-pwcvtz-rifit.json":
            raise ValueError(f"{case['id']}: unexpected DF auxiliary basis")
    result = json.loads(decode(raw, "stdout.json"))
    validate(result, case, method, basis_arg, backend, expected[2])
    xyz = decode(raw, "optimized.xyz").splitlines()
    if int(xyz[0]) != len(case["symbols"]) or len(xyz[2:]) != len(case["symbols"]):
        raise ValueError(f"{case['id']}: inconsistent XYZ atom count")
    for index, line in enumerate(xyz[2:]):
        z, *position = line.split()
        expected_z = ATOMIC_NUMBER[case["symbols"][index]]
        coordinates = result["coordinates_bohr"][index]
        if (
            int(z) != expected_z
            or len(position) != 3
            or any(
                not math.isclose(
                    float(value), x / ANGSTROM_TO_BOHR, rel_tol=0, abs_tol=5e-12
                )
                for value, x in zip(position, coordinates, strict=True)
            )
        ):
            raise ValueError(f"{case['id']}: inconsistent XYZ coordinates")
    outcome.update(
        {
            key: result["stages"][0][key]
            for key in ("energy_hartree", "max_force", "steps", "evaluations")
        }
    )
    return outcome, result["coordinates_bohr"]


def comparisons(
    case: JsonObject, xyz: list[list[float]], kind: str, reference: str
) -> list[JsonObject]:
    entries = []
    for point in case[kind]:
        if reference in point:
            measured = parameter(xyz, point["atoms"])
            entries.append(
                {
                    "molecule": case["id"],
                    "label": point["label"],
                    "atoms": point["atoms"],
                    "reference": point[reference],
                    "measured": measured,
                    "error": measured - point[reference],
                }
            )
    return entries


def summarize(cohort: JsonObject, attempts: list[JsonObject]) -> JsonObject:
    cases = {case["id"]: case for case in cohort["molecules"]}
    report = {"molecules": [], "comparison": {}, "parameters": {}}
    for name, case in cases.items():
        row = {"id": name, "re_se_source": case["re_se_source"], "attempts": []}
        compared: set[str] = set()
        for attempt in attempts:
            stages = attempt["records"]["cases"].get(name, {})
            if not stages:
                continue
            stage_results = {"attempt": attempt["id"]}
            for basis in STAGES:
                if basis == "cc-pcvqz" and name not in cohort["qz_controls"]:
                    continue
                mark, xyz = stage_outcome(stages.get(basis, {}), case, basis)
                stage_results[basis] = mark
                if xyz is None or basis == "rhf":
                    continue
                if basis in compared:
                    raise ValueError(f"{name}: duplicate converged {basis} attempt")
                compared.add(basis)
                references = ("re_se", basis.removeprefix("df-"))
                for reference in references:
                    for kind in ("bonds", "angles"):
                        label = f"{basis}/{reference}/{kind}"
                        entries = comparisons(case, xyz, kind, reference)
                        mark[f"{reference}_{kind}"] = stats(entries)
                        for entry in entries:
                            entry["attempt"] = attempt["id"]
                        report["parameters"].setdefault(label, []).extend(entries)
            row["attempts"].append(stage_results)
        report["molecules"].append(row)
    for label, entries in report["parameters"].items():
        basis, reference, kind = label.split("/")
        summary = stats(entries)
        if basis == reference:
            gate = LITERATURE_GATES[kind]
            summary["gate"] = gate
            summary["pass"] = None if not entries else summary["max_abs"] <= gate
        report["comparison"][label] = summary
    report["df_minus_conventional"] = {}
    for kind in ("bonds", "angles"):
        direct = {
            (row["molecule"], row["label"]): row
            for row in report["parameters"].get(f"cc-pcvtz/re_se/{kind}", [])
        }
        paired = []
        for row in report["parameters"].get(f"df-cc-pcvtz/re_se/{kind}", []):
            control = direct.get((row["molecule"], row["label"]))
            if control is not None:
                paired.append(
                    {
                        "molecule": row["molecule"],
                        "label": row["label"],
                        "conventional": control["measured"],
                        "df": row["measured"],
                        "error": row["measured"] - control["measured"],
                    }
                )
        report["df_minus_conventional"][kind] = {
            "summary": stats(paired),
            "parameters": paired,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--records", type=Path, default=ROOT / "records.json.xz")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capture:
        records = [
            {"id": path.parent.name, "records": capture(path)} for path in args.capture
        ]
        with args.records.open("xb") as stream:
            stream.write(
                lzma.compress(json.dumps(records, allow_nan=False).encode("utf-8"))
            )
    else:
        with lzma.open(args.records, "rt", encoding="utf-8") as stream:
            records = json.load(stream)
    report = summarize(read_json(ROOT / "cohort.json"), records)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["comparison"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
