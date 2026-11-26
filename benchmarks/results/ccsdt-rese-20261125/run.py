"""Run a frozen cohort through separate native RHF and CCSD(T) optimizations."""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
ANGSTROM_TO_BOHR = 1.8897261254578281
PRELIMINARY_FMAX = 3e-4
TARGET_FMAX = 2e-5
TARGET_SECONDS = 1200


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_xyz(case: JsonObject, path: Path) -> None:
    lines = [str(len(case["symbols"])), "Rough starting geometry, angstrom"]
    for symbol, xyz in zip(case["symbols"], case["seed_angstrom"], strict=True):
        lines.append(f"{ATOMIC_NUMBER[symbol]} " + " ".join(f"{x:.10f}" for x in xyz))
    with path.open("x", encoding="ascii") as stream:
        stream.write("\n".join(lines) + "\n")


def validate(
    result: JsonObject,
    molecule: JsonObject,
    method: str,
    basis: str | Path,
    backend: str,
    fmax: float,
) -> None:
    (stage,) = result["stages"]
    if (
        result["status"] != "converged"
        or stage["status"] != "converged"
        or result["method"] != method
        or stage["method"] != method
        or stage["basis"] != str(basis)
        or stage["backend"] != backend
        or not 0 <= result["max_force_hartree_per_bohr"] <= fmax
        or stage["max_force"] != result["max_force_hartree_per_bohr"]
        or stage["energy_hartree"] != result["energy_hartree"]
        or not math.isfinite(result["energy_hartree"])
        or not isinstance(stage["steps"], int)
        or not isinstance(stage["evaluations"], int)
        or not 0 <= stage["steps"] < stage["evaluations"]
        or len(result["coordinates_bohr"]) != len(molecule["symbols"])
        or any(
            len(row) != 3 or not all(map(math.isfinite, row))
            for row in result["coordinates_bohr"]
        )
    ):
        raise ValueError("native result did not meet the declared optimizer gate")


def optimize(
    directory: Path,
    cli: Path,
    molecule: JsonObject,
    start: Path,
    method: str,
    basis: str | Path,
    backend: str,
    fmax: float,
    timeout: int,
    auxiliary_basis: Path | None = None,
) -> Path | None:
    directory.mkdir()
    output = directory / "optimized.xyz"
    args = [
        str(cli),
        "optimize",
        str(start),
        "--output",
        str(output),
        "--json",
        "--method",
        method,
        "--stages",
        method,
        "--basis",
        str(basis),
        "--backend",
        backend,
        "--representation",
        "spherical",
        "--units",
        "angstrom",
        "--charge",
        str(molecule["charge"]),
        "--multiplicity",
        str(molecule["multiplicity"]),
        "--density-fitting",
        "cuda" if auxiliary_basis else "none",
        "--precision",
        "fp64",
        "--max-iterations",
        "220",
        "--energy-tolerance",
        "1e-12",
        "--density-tolerance",
        "1e-10",
        "--opt-fmax",
        str(fmax),
        "--opt-max-steps",
        "80",
        "--opt-max-step",
        "0.25",
    ]
    if auxiliary_basis is not None:
        args.extend(["--auxiliary-basis", str(auxiliary_basis)])
    if method in ("rccsd(t)", "df-rccsd(t)"):
        args.extend(
            [
                "--correlation-memory-bytes",
                str(32 * 1024**3),
                "--ccsd-max-iterations",
                "160",
                "--ccsd-energy-tolerance",
                "1e-12",
                "--ccsd-residual-tolerance",
                "1e-9",
            ]
        )
    stamp = datetime.now(timezone.utc).isoformat()
    dump(
        directory / "request.json",
        {"argv": args, "timeout_seconds": timeout, "utc": stamp},
    )
    print(molecule["id"], directory.name, "start", stamp, flush=True)
    tick = time.monotonic()
    receipt = {"status": "launch_failed", "returncode": None, "error": None}
    try:
        with (
            (directory / "stdout.json").open("x") as out,
            (directory / "stderr.txt").open("x") as err,
        ):
            child = subprocess.Popen(
                args, stdout=out, stderr=err, cwd=directory, start_new_session=True
            )
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
                receipt["status"] = "timeout"
            else:
                receipt["status"] = (
                    "native_failure" if code else "result_pending_validation"
                )
        receipt["returncode"] = code
        if code == 0:
            result = json.loads((directory / "stdout.json").read_text())
            validate(result, molecule, method, basis, backend, fmax)
            if not output.is_file():
                raise ValueError("optimized XYZ is missing")
            receipt["status"] = "converged"
    except (OSError, ValueError, KeyError, TypeError) as error:
        if receipt["status"] == "result_pending_validation":
            receipt["status"] = "result_invalid"
        receipt["error"] = str(error)
    wall = time.monotonic() - tick
    receipt["wall_seconds"] = wall
    dump(directory / "receipt.json", receipt)
    print(molecule["id"], directory.name, receipt["status"], f"{wall:.3f}s", flush=True)
    return output if receipt["status"] == "converged" else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--molecule", action="append", help="Run only named molecule IDs"
    )
    parser.add_argument(
        "--preliminary-backend", choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument(
        "--df-auxiliary-basis",
        type=Path,
        help="Run correlation-only DF-CCSD(T) instead of conventional RCCSD(T)",
    )
    parser.add_argument("--target-timeout", type=int, default=TARGET_SECONDS)
    args = parser.parse_args()
    if args.target_timeout <= 0:
        parser.error("target timeout must be positive")
    payload, cli, output = (p.resolve() for p in (args.payload, args.cli, args.output))
    cohort = json.loads((payload / "cohort.json").read_text())
    cases = {row["id"]: row for row in cohort["molecules"]}
    selection = args.molecule or list(cases)
    if len(set(selection)) != len(selection) or set(selection) - cases.keys():
        parser.error("molecule IDs must be unique cohort members")
    output.mkdir()
    dump(
        output / "campaign.json",
        {"molecules": selection, "qz_controls": cohort["qz_controls"]},
    )
    auxiliary = args.df_auxiliary_basis.resolve() if args.df_auxiliary_basis else None
    target_method = "df-rccsd(t)" if auxiliary else "rccsd(t)"
    target_stage = "df-cc-pcvtz" if auxiliary else "cc-pcvtz"
    preoptimized = {}
    tz_converged = 0
    for name in selection:
        case_dir = output / name
        case_dir.mkdir()
        seed = case_dir / "seed.xyz"
        seed_xyz(cases[name], seed)
        preopt = optimize(
            case_dir / "rhf",
            cli,
            cases[name],
            seed,
            "rhf",
            payload / "cc-pvdz.json",
            args.preliminary_backend,
            PRELIMINARY_FMAX,
            180,
        )
        if preopt is not None:
            preoptimized[name] = preopt
            result = optimize(
                case_dir / target_stage,
                cli,
                cases[name],
                preopt,
                target_method,
                payload / "cc-pcvtz.json",
                "cuda",
                TARGET_FMAX,
                args.target_timeout,
                auxiliary,
            )
            tz_converged += result is not None
    for name in [] if auxiliary else cohort["qz_controls"]:
        if name in preoptimized:
            optimize(
                output / name / "cc-pcvqz",
                cli,
                cases[name],
                preoptimized[name],
                "rccsd(t)",
                payload / "cc-pcvqz.json",
                "cuda",
                TARGET_FMAX,
                args.target_timeout,
            )
    dump(
        output / "campaign-complete.json",
        {"status": "attempts_complete", "tz_converged": tz_converged},
    )
    if not tz_converged:
        raise SystemExit("no cc-pCVTZ optimization passed")


if __name__ == "__main__":
    main()
