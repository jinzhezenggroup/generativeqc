"""Acquire the complete fixed converged Li/UHF Lambda oracle in fresh PySCF processes.

No GenerativeQC implementation is imported. All scientific arrays are complete,
losslessly encoded C-order little-endian FP64, rather than selected entries.
"""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import generate_cpu_hf_high_l_reference as audit

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "tests/reference_data/li_uhf_lambda/inputs.json"
MAX_BYTES = 256 * 1024
BLOCK_SHAPES = {
    "t1a": (2, 3),
    "t1b": (1, 4),
    "t2aa": (2, 2, 3, 3),
    "t2ab": (2, 1, 3, 4),
    "t2bb": (1, 1, 4, 4),
}
SHAPES = {
    **{
        name: (5, 5)
        for name in (
            "overlap",
            "hcore",
            "density_alpha",
            "density_beta",
            "fock_ao_alpha",
            "fock_ao_beta",
            "coefficients_alpha",
            "coefficients_beta",
            "fock_mo_alpha",
            "fock_mo_beta",
        )
    },
    **{
        name: (5,)
        for name in (
            "orbital_energies_alpha",
            "orbital_energies_beta",
            "occupations_alpha",
            "occupations_beta",
            "eris_energies_alpha",
            "eris_energies_beta",
        )
    },
    "ao_eri": (5, 5, 5, 5),
    **{
        f"{group}.{name}": shape
        for group in (
            "amplitudes",
            "updated_amplitudes",
            "residuals",
            "denominators",
            "multipliers",
            "updated_multipliers",
        )
        for name, shape in BLOCK_SHAPES.items()
    },
    **{name: () for name in ("hf_energy", "correlation_energy", "scf_gradient_norm")},
}

REPLAY_ATOL = 1e-13


def encode_array(value: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.shape != shape:
        raise ValueError(
            "scientific output is not complete real-FP64 of the declared shape"
        )
    if not np.isfinite(array).all():
        raise ValueError("nonfinite independent output")
    raw = array.astype("<f8", copy=False).tobytes(order="C")
    return {
        "dtype": "float64",
        "shape": list(shape),
        "order": "C",
        "encoding": "base64-le-f64",
        "bytes": len(raw),
        "sha256_le_f64": audit.digest(raw),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(record: dict[str, Any], shape: tuple[int, ...]) -> tuple[float, ...]:
    if set(record) != {
        "dtype",
        "shape",
        "order",
        "encoding",
        "bytes",
        "sha256_le_f64",
        "data",
    }:
        raise ValueError("incorrect complete array envelope")
    if (
        record["dtype"] != "float64"
        or record["shape"] != list(shape)
        or record["order"] != "C"
        or record["encoding"] != "base64-le-f64"
        or type(record["bytes"]) is not int
        or record["bytes"] != math.prod(shape) * 8
    ):
        raise ValueError("array shape/type/encoding mismatch")
    raw = base64.b64decode(record["data"], validate=True)
    if (
        len(raw) != record["bytes"]
        or base64.b64encode(raw).decode("ascii") != record["data"]
        or audit.digest(raw) != record["sha256_le_f64"]
    ):
        raise ValueError("array byte identity mismatch")
    values = struct.unpack(f"<{math.prod(shape)}d", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite decoded array")
    return values


def validate_inputs(inputs: dict[str, Any]) -> None:
    if (
        inputs["schema"] != "generativeqc.li-uhf-lambda.inputs.v1"
        or inputs["atoms"] != [["Li", [0, 0, 0]]]
        or inputs["unit"] != "Angstrom"
        or inputs["basis_name"] != "sto-3g"
        or type(inputs["charge"]) is not int
        or inputs["charge"] != 0
        or type(inputs["spin"]) is not int
        or inputs["spin"] != 1
        or inputs["cart"] is not False
        or inputs["populations"] != {"oa": 2, "ob": 1, "va": 3, "vb": 4}
        or inputs["scf"]
        != {
            "method": "UHF",
            "initial_guess": "minao",
            "conv_tol": 1e-13,
            "polish": {
                "direct_scf": False,
                "conv_tol_grad": 1e-10,
                "max_cycle": 300,
                "dm0": "converged initial density",
                "newton_fallback": True,
            },
        }
        or inputs["coupled_cluster"]
        != {
            "method": "UCCSD",
            "conv_tol": 1e-13,
            "conv_tol_normt": 1e-11,
            "max_cycle": 400,
            "lambda": "solve_lambda(eris=eris)",
        }
        or inputs["block_order"] != list(BLOCK_SHAPES)
        or inputs["preserved_gates"] != {"lambda_atol": 2e-7, "lambda_rtol": 0}
        or inputs["precision"] != "float64"
        or inputs["array_encoding"] != "base64 little-endian C-order FP64"
        or inputs["python"] != "3.12.12"
        or platform.python_version() != inputs["python"]
        or inputs["packages"]
        != {
            "pyscf": "2.14.0",
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "h5py": "3.16.0",
            "threadpoolctl": "3.6.0",
        }
    ):
        raise ValueError("outside the reviewed fixed converged Li Lambda profile")
    expected_paths = {
        "tests/python/test_uccsd_equations.py",
        "tests/python/test_uccsd_lambda.py",
        "tests/python/test_uccsd_solver.py",
        "tools/generate_cpu_hf_high_l_reference.py",
    }
    if {entry["path"] for entry in inputs["sources"]} != expected_paths:
        raise ValueError("incomplete reviewed source inventory")
    for entry in inputs["sources"]:
        if audit.digest((ROOT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("reviewed source/helper changed")
    if (
        Path(audit.__file__).resolve()
        != ROOT / "tools/generate_cpu_hf_high_l_reference.py"
    ):
        raise ValueError("unexpected imported audit helper")


def installed_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("pyscf")
    wanted = {inputs["basis_data"]["path"]: inputs["basis_data"]["sha256"]}
    wanted.update(
        {entry["path"]: entry["sha256"] for entry in inputs["upstream_lambda"]["files"]}
    )
    files = {}
    for entry in distribution.files or ():
        relative = str(entry)
        if relative in wanted:
            raw = Path(distribution.locate_file(entry)).read_bytes()
            if audit.digest(raw) != wanted[relative]:
                raise ValueError("installed basis or upstream Lambda source differs")
            files[relative] = {"bytes": len(raw), "sha256": audit.digest(raw)}
    if set(files) != set(wanted):
        raise ValueError("missing pinned installed scientific source")
    return files


def calculate(
    inputs: dict[str, Any], replica: str, config: Path, inventory: dict
) -> dict:
    import numpy as np
    import pyscf
    from pyscf import cc, gto, scf
    from pyscf.cc import uccsd_lambda
    from threadpoolctl import threadpool_info

    if (
        Path(pyscf.__config__.conf_file).resolve() != config.resolve()
        or config.read_bytes() != b""
        or pyscf.__version__ != inputs["packages"]["pyscf"]
    ):
        raise ValueError("PySCF configuration/version identity mismatch")
    audit.check_imported_modules(inventory)
    pinned_files = installed_inputs(inputs)
    packages = {
        name: audit.distribution_identity(name, version)
        for name, version in inputs["packages"].items()
    }
    with pyscf.lib.with_omp_threads(1):
        basis = {"Li": gto.basis.load("sto-3g", "Li")}
        if basis != inputs["resolved_basis"]:
            raise ValueError(
                "resolved STO-3G contractions differ from the reviewed input"
            )
        molecule = gto.M(
            atom=inputs["atoms"],
            unit=inputs["unit"],
            basis=basis,
            charge=0,
            spin=1,
            cart=False,
            verbose=0,
        )
        if (
            molecule.nao_nr() != 5
            or molecule.nelec != (2, 1)
            or molecule._basis != basis
        ):
            raise ValueError("unexpected resolved Li molecule")
        reference = scf.UHF(molecule)
        reference.conv_tol = 1e-13
        reference.kernel()
        if not reference.converged or reference.init_guess != "minao":
            raise ValueError(
                "fixed UHF reference did not converge with the original guess"
            )
        # Reproduce the original helper's first AO-to-MO preparation before polish.
        initial = cc.CCSD(reference)
        initial.ao2mo()
        reference.direct_scf = False
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 300
        reference.kernel(dm0=reference.make_rdm1())
        used_newton = not reference.converged
        if used_newton:
            reference = reference.newton()
            reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = (
                1e-13,
                1e-10,
                300,
            )
            reference.kernel(dm0=reference.make_rdm1())
        if not reference.converged:
            raise ValueError("polished independent UHF did not converge")
        coupled_cluster = cc.UCCSD(reference)
        coupled_cluster.conv_tol = 1e-13
        coupled_cluster.conv_tol_normt = 1e-11
        coupled_cluster.max_cycle = 400
        eris = coupled_cluster.ao2mo()
        coupled_cluster.kernel(eris=eris)
        if not coupled_cluster.converged:
            raise ValueError("independent UCCSD did not converge")
        coupled_cluster.solve_lambda(eris=eris)
        if not coupled_cluster.converged_lambda:
            raise ValueError("independent UCCSD Lambda did not converge")
        coeff_a, coeff_b = reference.mo_coeff
        ao_eri = molecule.intor("int2e")
        amplitudes = dict(
            zip(BLOCK_SHAPES, (*coupled_cluster.t1, *coupled_cluster.t2), strict=True)
        )
        multipliers = dict(
            zip(BLOCK_SHAPES, (*coupled_cluster.l1, *coupled_cluster.l2), strict=True)
        )
        singles, doubles = coupled_cluster.update_amps(
            coupled_cluster.t1, coupled_cluster.t2, eris
        )
        updates = dict(zip(BLOCK_SHAPES, (*singles, *doubles), strict=True))
        intermediates = uccsd_lambda.make_intermediates(
            coupled_cluster, coupled_cluster.t1, coupled_cluster.t2, eris
        )
        singles, doubles = uccsd_lambda.update_lambda(
            coupled_cluster,
            coupled_cluster.t1,
            coupled_cluster.t2,
            coupled_cluster.l1,
            coupled_cluster.l2,
            eris,
            intermediates,
        )
        lambda_updates = dict(zip(BLOCK_SHAPES, (*singles, *doubles), strict=True))
        ea, eb = eris.mo_energy
        gap_a, gap_b = ea[:2, None] - ea[None, 2:], eb[:1, None] - eb[None, 1:]
        denominators = {
            "t1a": gap_a,
            "t1b": gap_b,
            "t2aa": gap_a[:, None, :, None] + gap_a[None, :, None, :],
            "t2ab": gap_a[:, None, :, None] + gap_b[None, :, None, :],
            "t2bb": gap_b[:, None, :, None] + gap_b[None, :, None, :],
        }
        residuals = {
            name: denominators[name] * (updates[name] - value)
            for name, value in amplitudes.items()
        }
        correlation = float(coupled_cluster.e_corr)
        density = reference.make_rdm1()
        fock_ao = reference.get_fock()
        arrays = {
            "overlap": reference.get_ovlp(),
            "hcore": reference.get_hcore(),
            "density_alpha": density[0],
            "density_beta": density[1],
            "fock_ao_alpha": fock_ao[0],
            "fock_ao_beta": fock_ao[1],
            "coefficients_alpha": coeff_a,
            "coefficients_beta": coeff_b,
            "orbital_energies_alpha": reference.mo_energy[0],
            "orbital_energies_beta": reference.mo_energy[1],
            "occupations_alpha": reference.mo_occ[0],
            "occupations_beta": reference.mo_occ[1],
            "fock_mo_alpha": eris.focka,
            "fock_mo_beta": eris.fockb,
            "eris_energies_alpha": eris.mo_energy[0],
            "eris_energies_beta": eris.mo_energy[1],
            "ao_eri": ao_eri,
            "hf_energy": float(reference.e_tot),
            "correlation_energy": correlation,
            "scf_gradient_norm": float(
                np.linalg.norm(
                    reference.get_grad(reference.mo_coeff, reference.mo_occ, fock_ao)
                )
            ),
            **{
                f"{group}.{name}": value
                for group, blocks in (
                    ("amplitudes", amplitudes),
                    ("updated_amplitudes", updates),
                    ("denominators", denominators),
                    ("residuals", residuals),
                    ("multipliers", multipliers),
                    ("updated_multipliers", lambda_updates),
                )
                for name, value in blocks.items()
            },
        }
        encoded = {
            name: encode_array(arrays[name], shape) for name, shape in SHAPES.items()
        }
        pools = []
        for pool in threadpool_info():
            path = Path(pool["filepath"])
            pools.append(
                {
                    key: pool.get(key)
                    for key in (
                        "user_api",
                        "internal_api",
                        "version",
                        "num_threads",
                        "architecture",
                    )
                }
                | {"library": path.name, "sha256": audit.digest(path.read_bytes())}
            )
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("native scientific thread pool is not single-threaded")
    audit.check_imported_modules(inventory)
    receipt = {
        "schema": "generativeqc.li-uhf-lambda.receipt.v1",
        "replica": replica,
        "process_id": os.getpid(),
        "inputs": inputs,
        "provenance": {
            "inputs_sha256": audit.digest(INPUTS.read_bytes()),
            "generator_sha256": audit.digest(Path(__file__).read_bytes()),
            "source_sha256": {
                entry["path"]: entry["sha256"] for entry in inputs["sources"]
            },
            "execution": audit.execution_identity(),
            "configuration": {
                "policy": "explicit-empty-regular-file",
                "sha256": audit.digest(b""),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
            "threadpools": pools,
            "installed_scientific_inputs": pinned_files,
        },
        "molecule": {
            "basis": basis,
            "atoms_bohr": [["Li", molecule.atom_coord(0).tolist()]],
            "ao_labels": molecule.ao_labels(),
            "ao_labels_structured": [
                list(label) for label in molecule.ao_labels(fmt=False)
            ],
            "ao_locations": molecule.ao_loc_nr().tolist(),
            "nao": molecule.nao_nr(),
            "nelec": list(molecule.nelec),
            "charge": molecule.charge,
            "spin": molecule.spin,
            "cart": molecule.cart,
            "scf_converged": bool(reference.converged),
            "newton_fallback_used": used_newton,
            "cc_converged": bool(coupled_cluster.converged),
            "lambda_converged": bool(coupled_cluster.converged_lambda),
            "resolved_cc": {
                name: getattr(coupled_cluster, name)
                for name in (
                    "conv_tol",
                    "conv_tol_normt",
                    "max_cycle",
                    "level_shift",
                    "diis_space",
                )
            },
            "resolved_scf": {
                name: getattr(reference, name)
                for name in (
                    "init_guess",
                    "conv_tol",
                    "conv_tol_grad",
                    "max_cycle",
                    "diis_space",
                    "direct_scf",
                    "direct_scf_tol",
                    "level_shift",
                    "damp",
                )
            },
        },
        "arrays": encoded,
    }
    validate(receipt, inputs)
    return receipt


def validate(receipt: dict[str, Any], inputs: dict[str, Any]) -> None:
    if (
        set(receipt)
        != {
            "schema",
            "replica",
            "process_id",
            "inputs",
            "provenance",
            "molecule",
            "arrays",
        }
        or receipt["schema"] != "generativeqc.li-uhf-lambda.receipt.v1"
        or receipt["replica"] not in ("first", "second")
        or type(receipt["process_id"]) is not int
        or receipt["process_id"] <= 0
        or receipt["inputs"] != inputs
        or set(receipt["arrays"]) != set(SHAPES)
    ):
        raise ValueError("invalid fixed receipt identity/inventory")
    provenance = receipt["provenance"]
    if (
        provenance["inputs_sha256"] != audit.digest(INPUTS.read_bytes())
        or provenance["generator_sha256"] != audit.digest(Path(__file__).read_bytes())
        or provenance["source_sha256"]
        != {entry["path"]: entry["sha256"] for entry in inputs["sources"]}
        or provenance["execution"] != audit.execution_identity()
        or provenance["python"] != inputs["python"]
        or provenance["configuration"]
        != {"policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")}
        or provenance["installed_scientific_inputs"] != installed_inputs(inputs)
        or {name: entry["version"] for name, entry in provenance["packages"].items()}
        != inputs["packages"]
    ):
        raise ValueError("acquisition provenance mismatch")
    molecule = receipt["molecule"]
    if (
        type(molecule["nao"]) is not int
        or molecule["nao"] != 5
        or molecule["nelec"] != [2, 1]
        or any(type(value) is not int for value in molecule["nelec"])
        or molecule["basis"] != inputs["resolved_basis"]
        or molecule["ao_locations"] != inputs["ao_identity"]["locations"]
        or molecule["ao_labels_structured"] != inputs["ao_identity"]["labels"]
        or molecule["atoms_bohr"] != [["Li", [0, 0, 0]]]
        or type(molecule["charge"]) is not int
        or molecule["charge"] != 0
        or type(molecule["spin"]) is not int
        or molecule["spin"] != 1
        or molecule["cart"] is not False
        or molecule["scf_converged"] is not True
        or len(molecule["ao_labels"]) != 5
        or len(set(molecule["ao_labels"])) != 5
        or molecule["resolved_scf"]["conv_tol"] != 1e-13
        or molecule["resolved_scf"]["init_guess"] != "minao"
    ):
        raise ValueError("resolved Li UHF identity mismatch")
    arrays = {
        name: decode_array(receipt["arrays"][name], shape)
        for name, shape in SHAPES.items()
    }
    if arrays["occupations_alpha"] != (1, 1, 0, 0, 0) or arrays["occupations_beta"] != (
        1,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("incorrect spin occupations")
    if (
        molecule["resolved_scf"]["conv_tol_grad"] != 1e-10
        or molecule["resolved_scf"]["max_cycle"] != 300
        or molecule["resolved_scf"]["direct_scf"] is not False
        or molecule["cc_converged"] is not True
        or molecule["lambda_converged"] is not True
        or molecule["resolved_cc"]["conv_tol"] != 1e-13
        or molecule["resolved_cc"]["conv_tol_normt"] != 1e-11
        or molecule["resolved_cc"]["max_cycle"] != 400
    ):
        raise ValueError("convergence protocol differs from original Lambda fixture")
    for name in (
        "overlap",
        "hcore",
        "density_alpha",
        "density_beta",
        "fock_ao_alpha",
        "fock_ao_beta",
        "fock_mo_alpha",
        "fock_mo_beta",
    ):
        if any(
            abs(arrays[name][5 * i + j] - arrays[name][5 * j + i]) > 1e-12
            for i in range(5)
            for j in range(5)
        ):
            raise ValueError("nonsymmetric one-electron matrix")
    for name in BLOCK_SHAPES:
        for old, updated, denominator, residual in zip(
            arrays[f"amplitudes.{name}"],
            arrays[f"updated_amplitudes.{name}"],
            arrays[f"denominators.{name}"],
            arrays[f"residuals.{name}"],
            strict=True,
        ):
            if (
                abs(residual - denominator * (updated - old)) > 1e-13
                or abs(residual) >= 2e-9
            ):
                raise ValueError("independent converged CC residual mismatch")
        if (
            max(
                abs(a - b)
                for a, b in zip(
                    arrays[f"multipliers.{name}"],
                    arrays[f"updated_multipliers.{name}"],
                    strict=True,
                )
            )
            >= 2e-9
        ):
            raise ValueError("independent Lambda fixed point mismatch")
    for group in (
        "amplitudes",
        "updated_amplitudes",
        "residuals",
        "multipliers",
        "updated_multipliers",
    ):
        if any(value != 0 for value in arrays[f"{group}.t2bb"]):
            raise ValueError("single-beta same-spin channel must be explicit zero")


def generate(inputs: dict[str, Any], replica: str) -> dict[str, Any]:
    if any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules):
        raise ValueError("generation requires a fresh interpreter without PySCF")
    if any(name.startswith("PYSCF_") for name in os.environ):
        raise ValueError("unset external PySCF configuration")
    inventory = audit.module_inventory(inputs)
    with tempfile.TemporaryDirectory(prefix="gqc-li-config-") as directory:
        config = Path(directory) / "empty.py"
        config.write_bytes(b"")
        os.environ["PYSCF_CONFIG_FILE"] = str(config)
        try:
            result = calculate(inputs, replica, config, inventory)
        finally:
            os.environ.pop("PYSCF_CONFIG_FILE", None)
    return result


def verify(paths: list[Path], inputs: dict[str, Any]) -> dict[str, Any]:
    first, second = [audit.read_json(path) for path in paths]
    for receipt in (first, second):
        validate(receipt, inputs)
    if (
        first["replica"] != "first"
        or second["replica"] != "second"
        or first["process_id"] == second["process_id"]
        or first["provenance"] != second["provenance"]
        or first["molecule"] != second["molecule"]
    ):
        raise ValueError("independent-process identity/reconstruction mismatch")
    maximum = 0.0
    for name, shape in SHAPES.items():
        left = decode_array(first["arrays"][name], shape)
        right = decode_array(second["arrays"][name], shape)
        maximum = max(maximum, *(abs(a - b) for a, b in zip(left, right, strict=True)))
    if maximum > REPLAY_ATOL:
        raise ValueError("independent full-array replay disagrees")
    first["replay"] = {
        "schema": "generativeqc.independent-process-replay.v1",
        "receipt_sha256": [audit.digest(path.read_bytes()) for path in paths],
        "replicas": ["first", "second"],
        "process_ids": [first["process_id"], second["process_id"]],
        "absolute_tolerance": REPLAY_ATOL,
        "max_absolute_difference": maximum,
    }
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", choices=("first", "second"))
    parser.add_argument("--verify", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    inputs = audit.read_json(INPUTS)
    validate_inputs(inputs)
    if bool(args.replica) == bool(args.verify):
        parser.error("provide exactly one generation or replay mode")
    receipt = (
        verify(args.verify, inputs) if args.verify else generate(inputs, args.replica)
    )
    raw = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("complete Li receipt exceeds its strict byte cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    identity = args.replica or "validated"
    print(
        f"Validated complete Li receipt: {identity} bytes={len(raw)} sha256={audit.digest(raw)}"
    )
    if args.emit:
        print(
            f"GQC_LI_LAMBDA_JSON_BEGIN replica={identity} sha256={audit.digest(raw)} bytes={len(raw)}"
        )
        sys.stdout.write(raw.decode())
        print("GQC_LI_LAMBDA_JSON_END")


if __name__ == "__main__":
    main()
