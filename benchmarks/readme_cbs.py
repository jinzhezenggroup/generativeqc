"""Independent complete GPU DF-CCSD(T)/CBS E+F endpoints for QM7 C1--C7.

The explicit reference selector distinguishes correlation-only and full DF. Native
timing never depends on an oracle completing. Each call starts a new calculator;
repeats are full replays, not retained-density warm calls. Output is checkpointed
before and after each call so later failures do not erase completed endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks._support import environment_metadata

SCHEMA = "generativeqc.readme-df-cbs.v1"
QM7_SHA256 = "fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea"
QM7_INDICES = {1: 0, 2: 1, 3: 8, 4: 48, 5: 183, 6: 293, 7: 5604}
BASES = ("cc-pvdz", "cc-pvtz")
AUXILIARIES = ("cc-pvdz-ri", "cc-pvtz-ri")
HF_WEIGHTS = (-1 / math.expm1(1.63), 1 + 1 / math.expm1(1.63))
MP2_WEIGHTS = (-8 / 19, 27 / 19)
STEPS = (1e-4, 3e-5)


def digest(path: Path) -> str:
    """Hash the actual dataset, executable or receipt bytes."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def checkpoint(path: Path, payload: dict) -> None:
    """Replace a complete JSON record, retaining the prior record on interruption."""
    path = raw_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_array) + "\n"
    )
    temporary.replace(path)


def _json_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def geometry(dataset: Path, carbon: int) -> list:
    """Preserve QM7's float32 Bohr values, without recentering or optimization."""
    from scipy.io import loadmat

    if digest(dataset) != QM7_SHA256:
        raise ValueError("QM7 dataset checksum differs from the pinned input")
    data = loadmat(dataset)
    index = QM7_INDICES[carbon]
    selected = data["Z"][index] > 0
    numbers = data["Z"][index][selected].astype(int)
    coordinates = data["R"][index][selected].astype(float)
    if sorted(numbers.tolist()) != [1] * (2 * carbon + 2) + [6] * carbon:
        raise ValueError("QM7 representative is not the requested alkane formula")
    return [
        [int(number), position.tolist()]
        for number, position in zip(numbers, coordinates, strict=True)
    ]


def setup(
    atoms: list, carbon: int, budget: int, hf_reference: str = "exact"
) -> tuple[dict, tuple, tuple]:
    """Snapshot only basis metadata with PySCF; never compute an SCF seed here."""
    from pyscf import df, gto

    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot

    numbers = [atom[0] for atom in atoms]
    orbital = tuple(basis_snapshot(name, numbers) for name in BASES)
    if hf_reference not in {"exact", "df"}:
        raise ValueError("HF reference must be exact or df")
    fitting_names = (
        tuple(name + "-jkfit" for name in BASES)
        if hf_reference == "df"
        else AUXILIARIES
    )
    auxiliary = tuple(basis_snapshot(name, numbers) for name in fitting_names)
    molecules = [
        gto.M(atom=atoms, unit="Bohr", basis=name, verbose=0) for name in BASES
    ]
    protocol = {
        "carbon": carbon,
        "formula": f"C{carbon}H{2 * carbon + 2}",
        "qm7_index_zero_based": QM7_INDICES[carbon],
        "dataset_sha256": QM7_SHA256,
        "atoms_bohr": atoms,
        "orbital_bases": list(BASES),
        "auxiliary_bases": list(fitting_names),
        "orbital_identities": [basis.identity for basis in orbital],
        "auxiliary_identities": [basis.identity for basis in auxiliary],
        "nao": [molecule.nao_nr() for molecule in molecules],
        "naux": [
            df.make_auxmol(molecule, name).nao_nr()
            for molecule, name in zip(molecules, fitting_names, strict=True)
        ],
        "electrons": molecules[0].nelectron,
        "reference": (
            "RI-JK RHF; shared JK/correlation auxiliary"
            if hf_reference == "df"
            else "conventional exact RHF; correlation-only DF"
        ),
        "endpoint": "energy+analytic_forces_host",
        "precision": "fp64",
        "frozen_core": False,
        "correlation_memory_budget_bytes": budget,
        "hf_weights": list(HF_WEIGHTS),
        "mp2_weights": list(MP2_WEIGHTS),
        "correction_basis": "cc-pvdz",
        "timing": "calculator construction through returned host E+F; no retained state",
    }
    if hf_reference == "df":
        protocol["hf_reference"] = "df"
    return protocol, orbital, auxiliary


def native_call(
    atoms: list,
    orbital: tuple,
    auxiliary: tuple,
    budget: int,
    hf_reference: str = "exact",
) -> dict:
    """Measure all native basis groups and their response, not a sum of subphases."""
    from generativeqc import CBSBasis, CCSDTCBSCalculator

    options = {"hf_reference": "df"} if hf_reference == "df" else {}
    started = time.perf_counter()
    owner = CCSDTCBSCalculator(
        (CBSBasis(orbital[0], 2), CBSBasis(orbital[1], 3)),
        auxiliary_bases=auxiliary,
        device="cuda",
        calculator_options={"correlation_memory_budget_bytes": budget},
        **options,
    )
    result = owner.singlepoint(atoms, properties=("energy", "forces"))
    seconds = time.perf_counter() - started
    record = asdict(result)
    record["complete_seconds"] = seconds
    return record


def oracle_energies(atoms: list, hf_reference: str = "exact") -> dict[str, float]:
    """Independently reconverge every PySCF reference; no native outputs are read."""
    from tools.generativeqc_validation.correlation_df_oracle import energies

    fits = (
        tuple(name + "-jkfit" for name in BASES)
        if hf_reference == "df"
        else AUXILIARIES
    )
    options = {"df_reference": True} if hf_reference == "df" else {}
    lower = energies(atoms, BASES[0], fits[0], triples=True, **options)
    higher = energies(atoms, BASES[1], fits[1], **options)
    result = {f"low/{method}": energy for method, energy in lower.items()}
    result.update({f"high/{method}": energy for method, energy in higher.items()})
    result["composite"] = float(
        sum(
            weight * value["rhf"]
            for weight, value in zip(HF_WEIGHTS, (lower, higher), strict=True)
        )
        + sum(
            weight * (value["mp2"] - value["rhf"])
            for weight, value in zip(MP2_WEIGHTS, (lower, higher), strict=True)
        )
        + lower["df-ccsd(t)"]
        - lower["mp2"]
    )
    return result


def run_reference(payload: dict, output: Path) -> None:
    """Retain central energies and two-step directional checks incrementally."""
    atoms = payload["protocol"]["atoms_bohr"]
    hf_reference = payload["protocol"].get("hf_reference", "exact")
    coordinates = np.array([atom[1] for atom in atoms])
    direction = np.random.default_rng(709).normal(size=coordinates.shape)
    direction /= np.linalg.norm(direction)
    payload.update(direction=direction.tolist(), derivatives=[])
    payload["stage"] = "central energies"
    checkpoint(output, payload)
    payload["energies"] = oracle_energies(atoms, hf_reference)
    checkpoint(output, payload)
    for step in STEPS:
        payload["stage"] = f"directional derivative {step} Bohr"
        checkpoint(output, payload)
        displaced = [
            oracle_energies(
                [
                    [atom[0], position.tolist()]
                    for atom, position in zip(
                        atoms, coordinates + sign * step * direction, strict=True
                    )
                ],
                hf_reference,
            )
            for sign in (-1, 1)
        ]
        payload["derivatives"].append(
            {
                "step_bohr": step,
                "values": {
                    key: (displaced[1][key] - displaced[0][key]) / (2 * step)
                    for key in payload["energies"]
                },
            }
        )
        checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "reference"))
    parser.add_argument("--qm7", type=Path, required=True)
    parser.add_argument("--carbon", type=int, choices=QM7_INDICES, required=True)
    parser.add_argument("--budget-mib", type=int, default=65536)
    parser.add_argument("--hf-reference", choices=("exact", "df"), default="exact")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.budget_mib < 1:
        parser.error("positive sample count and budget required")
    if args.engine == "native" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU endpoints must run through srun")
    atoms = geometry(args.qm7, args.carbon)
    protocol, orbital, auxiliary = setup(
        atoms, args.carbon, args.budget_mib << 20, args.hf_reference
    )
    payload = {
        "schema": SCHEMA,
        "engine": args.engine,
        "protocol": protocol,
        "source_revision": args.source_revision,
        "driver_sha256": digest(Path(__file__)),
        "status": "running",
        "stage": "setup complete",
        "requested_samples": args.samples,
        "samples": [],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "environment": environment_metadata(
            distributions={"numpy": ("numpy",), "pyscf": ("pyscf",)}
        ),
        "controls": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(
                ("GENERATIVEQC_", "OMP_", "OPENBLAS_", "MKL_", "SCCACHE_")
            )
        },
    }
    if args.engine == "native":
        payload["library_sha256"] = digest(Path(os.environ["GENERATIVEQC_LIBRARY"]))
        payload["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0],
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    checkpoint(args.output, payload)
    try:
        if args.engine == "reference":
            run_reference(payload, args.output)
        else:
            for sample in range(args.samples):
                payload["stage"] = f"{'cold' if sample == 0 else 'replay'}/{sample}"
                checkpoint(args.output, payload)
                record = native_call(
                    atoms, orbital, auxiliary, args.budget_mib << 20, args.hf_reference
                )
                payload["samples"].append(record)
                checkpoint(args.output, payload)
                print(
                    f"C{args.carbon} sample={sample} {record['complete_seconds']:.6f} s",
                    flush=True,
                )
        payload.update(status="completed", stage="complete")
    except Exception as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoint(args.output, payload)


if __name__ == "__main__":
    main()
