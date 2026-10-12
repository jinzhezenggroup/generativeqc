"""Generate four independent fixed libcint ECP matrix references.

This optional tool imports no GenerativeQC implementation. It constructs each
PySCF molecule in each fresh process, retaining full matrices and resolved inputs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import generate_cpu_hf_high_l_reference as audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "tests/reference_data/ecp_raw_matrices/inputs.json"
MAX_BYTES = 256 * 1024
MATRIX_FIELDS = ("overlap", "ecp_raw", "ecp_normalized")
REPLAY_ATOL = 1e-13


def require_keys(value: Any, names: set[str]) -> None:
    if not isinstance(value, dict):
        raise TypeError("expected an object with the exact reference fields")
    if set(value) != names:
        raise ValueError("missing or unexpected reference fields")


def selectors(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"d{int(d_shell)}-{representation}",
            "representation": representation,
            "d_shell": d_shell,
        }
        for d_shell in inputs["d_shells"]
        for representation in inputs["representations"]
    ]


def validate_inputs(inputs: dict[str, Any], input_path: Path) -> None:
    require_keys(inputs, {
        "schema", "source", "audit_helper", "atoms", "unit", "charge", "spin",
        "basis_libraries", "ecp_libraries", "extra_d_shell", "representations",
        "d_shells", "precision", "matrix_unit", "overlap_unit", "normalization",
        "expected_core_electrons", "python", "packages", "preserved_gates",
        "basis_data_sources",
    })
    if inputs["packages"] != {
        "pyscf": "2.14.0", "numpy": "2.3.5", "scipy": "1.17.0",
        "h5py": "3.16.0", "threadpoolctl": "3.6.0",
    } or inputs["python"] != "3.12.12":
        raise ValueError("generation dependency pins changed")
    if inputs["schema"] != "generativeqc.ecp-raw-matrices.inputs.v1":
        raise ValueError("unsupported ECP input schema")
    if (
        inputs["atoms"] != [["Na", [0.13, -0.21, 0.17]], ["H", [0.43, 0.19, 3.2]]]
        or inputs["unit"] != "Bohr"
        or type(inputs["charge"]) is not int
        or type(inputs["spin"]) is not int
        or inputs["charge"] != 0
        or inputs["spin"] != 0
        or inputs["basis_libraries"] != {"Na": "lanl2dz", "H": "sto-3g"}
        or inputs["ecp_libraries"] != {"Na": "lanl2dz"}
        or inputs["extra_d_shell"] != [2, [0.35, 1.0]]
        or inputs["representations"] != ["spherical", "cartesian"]
        or inputs["d_shells"] != [False, True]
        or any(type(value) is not bool for value in inputs["d_shells"])
        or inputs["expected_core_electrons"] != [10, 0]
    ):
        raise ValueError("input is outside the four reviewed fixed ECP cases")
    if (
        inputs["precision"] != "float64"
        or inputs["matrix_unit"] != "Hartree"
        or inputs["overlap_unit"] != "dimensionless"
        or inputs["normalization"] != (
            "ECPscalar / sqrt(diag(int1e_ovlp))[:,None] / "
            "sqrt(diag(int1e_ovlp))[None,:]"
        )
    ):
        raise ValueError("scientific units or normalization mismatch")
    if platform.python_version() != inputs["python"]:
        raise ValueError("actual interpreter differs from the pinned manifest")
    for field in ("source", "audit_helper"):
        entry = inputs[field]
        path = ROOT / entry["path"]
        if not path.is_file() or audit.digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"reviewed source/helper byte identity mismatch: {field}")
    if Path(audit.__file__).resolve() != (
        ROOT / inputs["audit_helper"]["path"]
    ).resolve():
        raise ValueError("imported audit helper is not the reviewed checkout file")
    if input_path.resolve() != DEFAULT_INPUTS.resolve():
        raise ValueError("generation accepts only the reviewed input path")
    if len(selectors(inputs)) != 4:
        raise ValueError("expected exactly four case selectors")


def installed_integral_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("pyscf")
    basis_files = []
    libraries = []
    for entry in sorted(distribution.files or (), key=str):
        relative = str(entry)
        path = Path(relative)
        is_basis = (
            path.parent.as_posix() == "pyscf/gto/basis"
            and path.suffix == ".dat"
            and path.stem.replace("-", "").lower() in {"lanl2dz", "sto3g"}
        )
        is_library = ".so" in path.name and (
            path.name.startswith("libcgto") or path.name.startswith("libcint")
        )
        if is_basis or is_library:
            raw = Path(distribution.locate_file(entry)).read_bytes()
            record = {
                "path": relative,
                "bytes": len(raw),
                "sha256": audit.digest(raw),
            }
            (basis_files if is_basis else libraries).append(record)
    if len(basis_files) != 2 or not libraries:
        raise ValueError("missing allowlisted installed basis/integral identity")
    expected_basis = {entry["path"]: entry["sha256"] for entry in inputs["basis_data_sources"]}
    if {entry["path"]: entry["sha256"] for entry in basis_files} != expected_basis:
        raise ValueError("installed basis data differs from pinned upstream bytes")
    return {"basis_files": basis_files, "integral_libraries": libraries}


def array_values(record: dict[str, Any], shape: list[int]) -> Any:
    values = record["values"]
    if len(shape) == 2:
        if (
            not isinstance(values, list)
            or len(values) != shape[0]
            or any(
                not isinstance(row, list) or len(row) != shape[1]
                for row in values
            )
        ):
            raise ValueError("incorrect full matrix dimensions")
    elif (
        not isinstance(values, list)
        or len(values) != shape[0]
        or any(isinstance(value, list) for value in values)
    ):
        raise ValueError("incorrect vector dimensions")
    if record != audit.array_record(values, shape):
        raise ValueError("array metadata, finite FP64 values or byte hash mismatch")
    return values


def basis_layout(
    basis: dict[str, Any], cart: bool
) -> tuple[int, list[int], list[list[int]]]:
    require_keys(basis, {"Na", "H"})
    count = 0
    locations = [0]
    atom_slices = []
    for atom in ("Na", "H"):
        shells = basis[atom]
        start_shell, start_ao = len(locations) - 1, count
        if not isinstance(shells, list) or not 1 <= len(shells) <= 16:
            raise ValueError("invalid bounded resolved basis shell inventory")
        for shell in shells:
            if (
                not isinstance(shell, list)
                or len(shell) < 2
                or type(shell[0]) is not int
                or not 0 <= shell[0] <= 2
            ):
                raise ValueError("invalid resolved angular shell")
            angular = shell[0]
            contractions = len(shell[1]) - 1
            if not 1 <= contractions <= 8:
                raise ValueError("invalid resolved contraction count")
            for row in shell[1:]:
                if not isinstance(row, list) or len(row) != contractions + 1:
                    raise ValueError("ragged resolved contraction")
                audit.array_record(row, [len(row)])
                if row[0] <= 0:
                    raise ValueError("nonpositive resolved Gaussian exponent")
            components = (angular + 1) * (angular + 2) // 2 if cart else 2 * angular + 1
            count += contractions * components
            locations.append(count)
        atom_slices.append([start_shell, len(locations) - 1, start_ao, count])
    if not 1 <= count <= 20:
        raise ValueError("resolved AO count exceeds the fixed scope")
    return count, locations, atom_slices


def validate_ecp(ecp: dict[str, Any]) -> None:
    require_keys(ecp, {"Na"})
    potential = ecp["Na"]
    if (
        not isinstance(potential, list) or len(potential) != 2
        or type(potential[0]) is not int or potential[0] != 10
        or not isinstance(potential[1], list) or len(potential[1]) != 3
    ):
        raise ValueError("invalid resolved Na ECP inventory")
    angular_channels = []
    for channel in potential[1]:
        if (
            not isinstance(channel, list) or len(channel) != 2
            or type(channel[0]) is not int
            or not isinstance(channel[1], list) or not 3 <= len(channel[1]) <= 8
        ):
            raise ValueError("invalid resolved ECP channel")
        angular_channels.append(channel[0])
        term_count = 0
        for terms in channel[1]:
            if not isinstance(terms, list) or len(terms) > 32:
                raise ValueError("invalid bounded ECP radial terms")
            for term in terms:
                if not isinstance(term, list) or len(term) != 2:
                    raise ValueError("invalid scalar ECP exponent/coefficient pair")
                audit.array_record(term, [2])
                if term[0] <= 0:
                    raise ValueError("nonpositive ECP Gaussian exponent")
                term_count += 1
        if not term_count:
            raise ValueError("empty resolved ECP channel")
    if sorted(angular_channels) != [-1, 0, 1]:
        raise ValueError("incorrect local/s/p ECP channel inventory")


def validate_provenance(
    provenance: dict[str, Any], inputs: dict[str, Any], input_path: Path
) -> None:
    require_keys(
        provenance,
        {
            "configuration", "inputs_sha256", "generator_sha256",
            "audit_helper_sha256", "test_source_sha256", "execution",
            "python", "platform", "machine", "packages", "threadpools",
            "installed_integral_inputs",
        },
    )
    expected = {
        "configuration": {
            "policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")
        },
        "inputs_sha256": audit.digest(input_path.read_bytes()),
        "generator_sha256": audit.digest(Path(__file__).read_bytes()),
        "audit_helper_sha256": inputs["audit_helper"]["sha256"],
        "test_source_sha256": inputs["source"]["sha256"],
        "execution": audit.execution_identity(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for key, value in expected.items():
        if provenance[key] != value:
            raise ValueError(f"reference provenance mismatch: {key}")
    require_keys(provenance["packages"], set(inputs["packages"]))
    for name, version in inputs["packages"].items():
        package = provenance["packages"][name]
        if package["version"] != version:
            raise ValueError("pinned package version mismatch")
        for key in ("wheel_record_sha256", "installed_source_native_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", package[key]) is None:
                raise ValueError("invalid installed package identity")
        for key in ("installed_source_native_files", "installed_source_native_bytes"):
            if type(package[key]) is not int or package[key] < 1:
                raise ValueError("invalid installed package inventory")
    if provenance["installed_integral_inputs"] != installed_integral_inputs(inputs):
        raise ValueError("installed basis/native integral byte identity changed")
    if not provenance["threadpools"]:
        raise ValueError("missing native thread-pool identity")
    for pool in provenance["threadpools"]:
        require_keys(
            pool,
            {
                "user_api", "internal_api", "version", "num_threads",
                "architecture", "library", "sha256",
            },
        )
        if type(pool["num_threads"]) is not int or pool["num_threads"] != 1:
            raise ValueError("native thread pool is not single-threaded")
        for key in ("user_api", "internal_api", "library"):
            if not isinstance(pool[key], str) or not pool[key]:
                raise ValueError("invalid native library metadata")
        if Path(pool["library"]).name != pool["library"]:
            raise ValueError("only native library basenames may be recorded")
        if re.fullmatch(r"[0-9a-f]{64}", pool["sha256"]) is None:
            raise ValueError("invalid native library byte hash")
        for key in ("version", "architecture"):
            if pool[key] is not None and not isinstance(pool[key], str):
                raise ValueError("invalid optional native library metadata")


def validate(
    receipt: dict[str, Any], inputs: dict[str, Any], input_path: Path
) -> None:
    require_keys(
        receipt, {"schema", "replica", "process_id", "provenance", "inputs", "records"}
    )
    if (
        receipt["schema"] != "generativeqc.ecp-raw-matrices.receipt.v1"
        or receipt["inputs"] != inputs
        or receipt["replica"] not in ("first", "second")
        or type(receipt["process_id"]) is not int
        or receipt["process_id"] <= 0
    ):
        raise ValueError("invalid receipt identity or input manifest")
    validate_provenance(receipt["provenance"], inputs, input_path)
    if not isinstance(receipt["records"], list) or len(receipt["records"]) != 4:
        raise ValueError("incomplete fixed ECP case inventory")
    base_basis = None
    base_ecp = None
    for selector, record in zip(selectors(inputs), receipt["records"], strict=True):
        require_keys(
            record,
            {
                "input", "input_sha256", "nao", "ao_labels", "ao_locations",
                "atom_ao_slices", "core_electrons", "nelec",
                "molecule_basis", "molecule_ecp", "molecule_atoms",
                "effective_nuclear_charges", "overlap", "normalization",
                "ecp_raw", "ecp_normalized",
            },
        )
        case = record["input"]
        require_keys(
            case,
            {
                "id", "representation", "d_shell", "atoms", "unit",
                "charge", "spin", "cart", "basis", "ecp",
            },
        )
        for key, value in selector.items():
            if type(case[key]) is not type(value) or case[key] != value:
                raise ValueError("fixed case selector mismatch")
        for key in ("atoms", "unit", "charge", "spin"):
            if case[key] != inputs[key]:
                raise ValueError("explicit molecular input mismatch")
        cart = selector["representation"] == "cartesian"
        if type(case["cart"]) is not bool or case["cart"] != cart:
            raise ValueError("AO representation mismatch")
        if record["input_sha256"] != audit.digest(audit.canonical(case)):
            raise ValueError("resolved molecular input checksum mismatch")
        validate_ecp(case["ecp"])
        if type(case["charge"]) is not int or type(case["spin"]) is not int:
            raise TypeError("molecular charge and spin must be integers")
        if (
            record["molecule_atoms"] != case["atoms"]
            or record["effective_nuclear_charges"] != [1, 1]
            or any(type(value) is not int for value in record["effective_nuclear_charges"])
            or record["molecule_basis"] != case["basis"]
            or record["molecule_ecp"] != case["ecp"]
        ):
            raise ValueError("PySCF resolved molecule differs from explicit input")
        unextended = json.loads(audit.canonical(case["basis"]))
        if selector["d_shell"]:
            if unextended["Na"][-1] != inputs["extra_d_shell"]:
                raise ValueError("incorrect explicit d-shell extension")
            unextended["Na"].pop()
        if base_basis is None:
            base_basis, base_ecp = unextended, case["ecp"]
        if unextended != base_basis or case["ecp"] != base_ecp:
            raise ValueError("basis or ECP changed across the fixed four cases")
        nao, expected_locations, expected_slices = basis_layout(case["basis"], cart)
        if type(record["nao"]) is not int or record["nao"] != nao:
            raise ValueError("resolved AO count mismatch")
        labels = record["ao_labels"]
        if (
            not isinstance(labels, list) or len(labels) != nao
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != nao
        ):
            raise ValueError("incomplete or ambiguous AO ordering")
        locations = record["ao_locations"]
        shell_count = sum(len(shells) for shells in case["basis"].values())
        if (
            not isinstance(locations, list) or len(locations) != shell_count + 1
            or any(type(value) is not int for value in locations)
            or locations[0] != 0 or locations[-1] != nao
            or any(a >= b for a, b in zip(locations, locations[1:], strict=False))
        ):
            raise ValueError("invalid shell-to-AO ordering")
        if locations != expected_locations:
            raise ValueError("shell-to-AO map differs from resolved basis order")
        atom_slices = record["atom_ao_slices"]
        if (
            not isinstance(atom_slices, list) or len(atom_slices) != 2
            or any(
                not isinstance(row, list) or len(row) != 4
                or any(type(value) is not int for value in row)
                for row in atom_slices
            )
            or atom_slices[0][0] != 0 or atom_slices[-1][1] != shell_count
            or atom_slices[0][2] != 0 or atom_slices[-1][3] != nao
            or atom_slices[0][1] != atom_slices[1][0]
            or atom_slices[0][3] != atom_slices[1][2]
        ):
            raise ValueError("invalid atom-to-AO ordering")
        if atom_slices != expected_slices:
            raise ValueError("atom-to-AO map differs from explicit Na/H order")
        if (
            record["core_electrons"] != [10, 0] or record["nelec"] != [1, 1]
            or any(type(value) is not int for value in record["core_electrons"] + record["nelec"])
        ):
            raise ValueError("incorrect effective electron bookkeeping")
        matrices = {
            name: array_values(record[name], [nao, nao]) for name in MATRIX_FIELDS
        }
        norms = array_values(record["normalization"], [nao])
        for i in range(nao):
            if matrices["overlap"][i][i] <= 0 or norms[i] <= 0:
                raise ValueError("nonpositive AO overlap normalization")
            if abs(norms[i] - math.sqrt(matrices["overlap"][i][i])) > REPLAY_ATOL:
                raise ValueError("AO normalization does not match overlap diagonal")
            for j in range(nao):
                expected = matrices["ecp_raw"][i][j] / norms[i] / norms[j]
                if abs(matrices["ecp_normalized"][i][j] - expected) > REPLAY_ATOL:
                    raise ValueError("normalized ECP matrix does not match full raw data")
                for values in matrices.values():
                    if abs(values[i][j] - values[j][i]) > 1e-12:
                        raise ValueError("independent scalar matrix is not symmetric")


def generate(
    inputs: dict[str, Any], input_path: Path, replica: str
) -> dict[str, Any]:
    if any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules):
        raise ValueError("a fresh interpreter without PySCF is required")
    if any(name.startswith("PYSCF_") for name in os.environ):
        raise ValueError("unset external PySCF configuration before generation")
    inventory = audit.module_inventory(inputs)
    with tempfile.TemporaryDirectory(prefix="gqc-ecp-config-") as directory:
        config = Path(directory) / "empty.py"
        config.write_bytes(b"")
        os.environ["PYSCF_CONFIG_FILE"] = str(config)
        try:
            result = calculate(inputs, input_path, replica, config, inventory)
        finally:
            os.environ.pop("PYSCF_CONFIG_FILE", None)
    return result


def calculate(
    inputs: dict[str, Any], input_path: Path, replica: str,
    config: Path, inventory: dict[str, set[Path]],
) -> dict[str, Any]:
    import numpy as np
    import pyscf
    from pyscf import gto
    from threadpoolctl import threadpool_info

    selected = getattr(pyscf.__config__, "conf_file", None)
    if selected is None or Path(selected).resolve() != config.resolve():
        raise ValueError("PySCF did not select the explicit empty regular config")
    if config.read_bytes() != b"" or pyscf.__version__ != inputs["packages"]["pyscf"]:
        raise ValueError("PySCF configuration/version mismatch")
    audit.check_imported_modules(inventory)
    packages = {
        name: audit.distribution_identity(name, version)
        for name, version in inputs["packages"].items()
    }
    records = []
    with pyscf.lib.with_omp_threads(1):
        for selector in selectors(inputs):
            # Resolve and construct anew for every case in every process.
            basis = {
                atom: gto.basis.load(name, atom)
                for atom, name in inputs["basis_libraries"].items()
            }
            if selector["d_shell"]:
                basis["Na"].append(json.loads(audit.canonical(inputs["extra_d_shell"])))
            ecp = {
                atom: gto.basis.load_ecp(name, atom)
                for atom, name in inputs["ecp_libraries"].items()
            }
            case = {
                **selector,
                "atoms": inputs["atoms"], "unit": inputs["unit"],
                "charge": inputs["charge"], "spin": inputs["spin"],
                "cart": selector["representation"] == "cartesian",
                "basis": basis, "ecp": ecp,
            }
            molecule = gto.M(
                atom=case["atoms"], basis=basis, ecp=ecp, unit=case["unit"],
                charge=case["charge"], spin=case["spin"], cart=case["cart"],
                verbose=0,
            )
            if not np.array_equal(
                molecule.atom_coords(), [position for _, position in inputs["atoms"]]
            ):
                raise ValueError("actual molecule coordinates differ from Bohr inputs")
            overlap = molecule.intor("int1e_ovlp")
            raw = molecule.intor("ECPscalar")
            norms = np.sqrt(overlap.diagonal())
            normalized = raw / norms[:, None] / norms[None, :]
            nao = molecule.nao_nr()
            records.append({
                "input": case, "input_sha256": audit.digest(audit.canonical(case)),
                "nao": nao, "ao_labels": molecule.ao_labels(),
                "ao_locations": molecule.ao_loc_nr().tolist(),
                "atom_ao_slices": molecule.aoslice_by_atom().tolist(),
                "core_electrons": [molecule.atom_nelec_core(i) for i in range(2)],
                "nelec": list(molecule.nelec),
                "molecule_basis": molecule._basis, "molecule_ecp": molecule._ecp,
                "molecule_atoms": [[molecule.atom_symbol(i), molecule.atom_coord(i).tolist()] for i in range(2)],
                "effective_nuclear_charges": [molecule.atom_charge(i) for i in range(2)],
                "overlap": audit.array_record(overlap.tolist(), [nao, nao]),
                "normalization": audit.array_record(norms.tolist(), [nao]),
                "ecp_raw": audit.array_record(raw.tolist(), [nao, nao]),
                "ecp_normalized": audit.array_record(normalized.tolist(), [nao, nao]),
            })
        pools = []
        for pool in threadpool_info():
            path = Path(pool["filepath"])
            pools.append({
                key: pool.get(key)
                for key in (
                    "user_api", "internal_api", "version", "num_threads", "architecture"
                )
            } | {"library": path.name, "sha256": audit.digest(path.read_bytes())})
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("native pool is not single-threaded")
    audit.check_imported_modules(inventory)
    receipt = {
        "schema": "generativeqc.ecp-raw-matrices.receipt.v1",
        "replica": replica, "process_id": os.getpid(), "inputs": inputs,
        "provenance": {
            "configuration": {
                "policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")
            },
            "inputs_sha256": audit.digest(input_path.read_bytes()),
            "generator_sha256": audit.digest(Path(__file__).read_bytes()),
            "audit_helper_sha256": inputs["audit_helper"]["sha256"],
            "test_source_sha256": inputs["source"]["sha256"],
            "execution": audit.execution_identity(),
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "threadpools": pools, "installed_integral_inputs": installed_integral_inputs(inputs),
        },
        "records": records,
    }
    validate(receipt, inputs, input_path)
    return receipt


def verify(
    paths: list[Path], inputs: dict[str, Any], input_path: Path
) -> dict[str, Any]:
    first, second = [audit.read_json(path) for path in paths]
    for receipt in (first, second):
        validate(receipt, inputs, input_path)
    if (
        first["replica"] == second["replica"]
        or first["process_id"] == second["process_id"]
        or first["provenance"] != second["provenance"]
    ):
        raise ValueError("independent replay identity/provenance mismatch")
    maximum = 0.0
    numeric_fields = (*MATRIX_FIELDS, "normalization")
    for left, right in zip(first["records"], second["records"], strict=True):
        for key in set(left) - set(numeric_fields):
            if left[key] != right[key]:
                raise ValueError(f"replay molecular metadata mismatch: {key}")
        for key in numeric_fields:
            a, b = left[key]["values"], right[key]["values"]
            if key != "normalization":
                a = [value for row in a for value in row]
                b = [value for row in b for value in row]
            maximum = max(
                maximum, *(abs(x - y) for x, y in zip(a, b, strict=True))
            )
    if maximum > REPLAY_ATOL:
        raise ValueError(f"independent full-matrix replay disagrees: {maximum}")
    first["replay"] = {
        "schema": "generativeqc.independent-process-replay.v1",
        "receipt_sha256": [audit.digest(path.read_bytes()) for path in paths],
        "replicas": [first["replica"], second["replica"]],
        "process_ids": [first["process_id"], second["process_id"]],
        "absolute_tolerance": REPLAY_ATOL, "max_absolute_difference": maximum,
    }
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica", choices=("first", "second"))
    parser.add_argument("--verify", type=Path, nargs=2)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    inputs = audit.read_json(args.inputs)
    validate_inputs(inputs, args.inputs)
    if bool(args.verify) == bool(args.replica):
        parser.error("provide exactly one of --replica or --verify")
    result = (
        verify(args.verify, inputs, args.inputs)
        if args.verify else generate(inputs, args.inputs, args.replica)
    )
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("reference receipt exceeds its strict byte cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    identity = args.replica or "validated"
    print(f"Validated four matrices: replica={identity} bytes={len(raw)} sha256={audit.digest(raw)}")
    if args.emit:
        print(f"GQC_ECP_JSON_BEGIN replica={identity} sha256={audit.digest(raw)} bytes={len(raw)}")
        sys.stdout.write(raw.decode())
        print("GQC_ECP_JSON_END")


if __name__ == "__main__":
    main()
