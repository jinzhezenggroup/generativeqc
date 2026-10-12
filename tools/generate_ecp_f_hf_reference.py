"""Acquire four fixed NaH orbital-f HF endpoints in eight fresh CPU processes.

The generator imports only independently installed PySCF scientific code and
retains every SCF matrix and every force component. --check-inputs is a stdlib
preflight, not an acquisition. No output is committed until independently reviewed.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import generate_cpu_hf_high_l_reference as audit
import generate_ecp_raw_matrix_reference as ecp_audit

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "tests/reference_data/ecp_f_hf/inputs.json"
MAX_BYTES = 2 * 1024 * 1024
REPLAY_ATOL = 1e-13
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SCF_FIELDS = (
    "init_guess",
    "conv_tol",
    "conv_tol_grad",
    "max_cycle",
    "diis_space",
    "diis_start_cycle",
    "diis_file",
    "diis_space_rollback",
    "diis_damp",
    "conv_tol_cpscf",
    "direct_scf",
    "direct_scf_tol",
    "level_shift",
    "damp",
    "conv_check",
    "max_memory",
)


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("reference byte limit exceeded")
    result = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise TypeError("expected a JSON object")
    audit.canonical(result)
    return result


def profiles(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{state['method'].lower()}-{representation}": {
            **state,
            "representation": representation,
            "cart": representation == "cartesian",
            "nao": 19 if representation == "cartesian" else 16,
        }
        for state in inputs["states"]
        for representation in inputs["representations"]
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    """Validate scientific identity without executing scientific imports."""
    if set(inputs) != {
        "states",
        "basis_libraries",
        "schema",
        "preserved_gates",
        "scf",
        "unit",
        "basis_data_sources",
        "representations",
        "packages",
        "precision",
        "source_repository",
        "extra_f_shell",
        "source_commit",
        "resolved_ecp",
        "atoms",
        "resolved_basis",
        "python",
        "array_encoding",
        "upstream_scf_sources",
        "acquisition",
        "sources",
        "ecp_libraries",
    }:
        raise ValueError("unexpected or incomplete input manifest fields")
    if (
        inputs["schema"] != "generativeqc.ecp-f-hf.inputs.v1"
        or inputs["source_repository"] != "jinzhezenggroup/generativeqc"
        or inputs["source_commit"] != "44d8c91d80c5c341cd9ce98e86b52b512ce1d444"
        or inputs["atoms"] != [["Na", [0.13, -0.21, 0.17]], ["H", [0.43, 0.19, 3.2]]]
        or inputs["unit"] != "Bohr"
        or inputs["basis_libraries"] != {"Na": "lanl2dz", "H": "sto-3g"}
        or inputs["ecp_libraries"] != {"Na": "lanl2dz"}
        or inputs["extra_f_shell"] != [3, [0.55, 0.81], [0.17, 0.23]]
        or inputs["states"]
        != [
            {"method": "RHF", "charge": 0, "spin": 0, "nelec": [1, 1]},
            {"method": "UHF", "charge": 1, "spin": 1, "nelec": [1, 0]},
        ]
        or any(
            type(state[key]) is not int
            for state in inputs["states"]
            for key in ("charge", "spin")
        )
        or inputs["representations"] != ["cartesian", "spherical"]
        or inputs["scf"]
        != {
            "call": "(scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)",
            "conv_tol": 1e-12,
            "initial_guess": "minao",
            "one_electron_dispatch": "UHF factory selects HF1e; initial guess is not used; zero SCF cycles",
            "other_settings": "pinned PySCF defaults; no override, polish, or fallback",
        }
        or inputs["python"] != "3.12.12"
        or inputs["packages"]
        != {
            "pyscf": "2.14.0",
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "h5py": "3.16.0",
            "threadpoolctl": "3.6.0",
            "setuptools": "83.0.0",
        }
        or inputs["precision"] != "float64"
        or inputs["array_encoding"] != "base64 little-endian C-order FP64"
        or inputs["preserved_gates"]
        != {
            "energy_atol": 2e-8,
            "energy_rtol": 0,
            "force_atol": 2e-6,
            "force_rtol": 0,
            "net_force_atol": 2e-7,
        }
        or inputs["acquisition"]
        != {
            "repository": "njzjz-bot/vibeqc",
            "branch": "oracle/ecp-f-hf-reference-20261012",
            "profiles": 4,
            "fresh_processes_per_profile": 2,
            "replay_atol": REPLAY_ATOL,
            "device": "CPU",
            "native_threads": 1,
        }
    ):
        raise ValueError("outside the reviewed fixed orbital-f HF quartet")
    for key, expected in (
        (
            "resolved_basis",
            "8db2745d37cc6b11efa4a5d9836f6839d1d95dafeae85058fdf9f26604f593ff",
        ),
        (
            "resolved_ecp",
            "6645d9c803ef2894d5e2edd6b1fced43e48d4e5a6fe40c8fa2e15862b374fcd8",
        ),
    ):
        if audit.digest(audit.canonical(inputs[key])) != expected:
            raise ValueError("resolved independent basis/ECP input mismatch")
    if {entry["path"]: entry["sha256"] for entry in inputs["basis_data_sources"]} != {
        "pyscf/gto/basis/lanl2dz.dat": "8271af43b895a6b2d2b47f2c958fe86c3c990f76e982a54ac03cc3c6dd196b64",
        "pyscf/gto/basis/sto-3g.dat": "0514de5da5df92e477ce69ef906c81589a310260557bcda84531b144eed00cc3",
    }:
        raise ValueError("pinned upstream basis bytes changed")

    if {entry["path"]: entry["sha256"] for entry in inputs["upstream_scf_sources"]} != {
        "pyscf/scf/__init__.py": "42754aaf7518664076082bab3e4854a954cda79afe10c8cf7f48e7aa23c16e42",
        "pyscf/scf/hf.py": "34ea1a9057da3a2d5aa37fe51d4e7305580bcb8e0118a5f17ba95ba5bb36c22f",
        "pyscf/scf/uhf.py": "b3ee3e679ab64d51ad610f04d02f4ac332c3f9238aeede7ba6bd80ad962a8fee",
        "pyscf/grad/uhf.py": "cdcbe1afce1ab5447fcc2976b58b0ab8b8a0a6d039b1d902416ebabe2e6487c1",
    }:
        raise ValueError("pinned upstream SCF source identity changed")


def installed_scf_sources(inputs: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("pyscf")
    records = {}
    owned = {str(entry): entry for entry in distribution.files or ()}
    for entry in inputs["upstream_scf_sources"]:
        if entry["path"] not in owned:
            raise ValueError("missing pinned upstream SCF source")
        raw = Path(distribution.locate_file(owned[entry["path"]])).read_bytes()
        if audit.digest(raw) != entry["sha256"]:
            raise ValueError("installed upstream SCF source differs from reviewed file")
        records[entry["path"]] = {"bytes": len(raw), "sha256": audit.digest(raw)}
    return records


def validate_checkout(inputs: dict[str, Any], *, runtime: bool) -> None:
    paths = {
        "tests/python/test_ecp.py",
        "tests/python/test_ecp_f.py",
        "tools/generate_cpu_hf_high_l_reference.py",
        "tools/generate_ecp_raw_matrix_reference.py",
    }
    if {entry["path"] for entry in inputs["sources"]} != paths:
        raise ValueError("incomplete historical source inventory")
    for entry in inputs["sources"]:
        if audit.digest((ROOT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"reviewed acquisition source differs: {entry['path']}")
    for module in (audit, ecp_audit):
        if (
            Path(module.__file__).resolve()
            != ROOT / "tools" / Path(module.__file__).name
        ):
            raise ValueError("audit helper is outside reviewed checkout")
    if runtime:
        if platform.python_version() != inputs["python"]:
            raise ValueError("actual interpreter differs from pinned Python")
        if any(os.getenv(name) != "1" for name in THREAD_VARIABLES):
            raise ValueError("all native thread limits must be explicitly one")
        execution = audit.execution_identity()
        if (
            execution.get("github_repository") != inputs["acquisition"]["repository"]
            or execution.get("github_sha") != execution["checkout_sha"]
            or os.getenv("GITHUB_REF")
            != "refs/heads/" + inputs["acquisition"]["branch"]
            or not execution.get("github_run_id", "").isdigit()
            or not execution.get("github_run_attempt", "").isdigit()
            or os.getenv("CUDA_VISIBLE_DEVICES") != ""
        ):
            raise ValueError(
                "acquisition is restricted to the reviewed CPU-only fork branch"
            )


def shapes(profile: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    n = profile["nao"]
    spin_prefix = (2,) if profile["spin"] else ()
    return {
        **{name: (n, n) for name in ("overlap", "hcore", "ecp_raw", "ecp_normalized")},
        "normalization": (n,),
        **{
            name: (*spin_prefix, n, n)
            for name in (
                "density",
                "fock",
                "coefficients",
                "commutator_residual",
            )
        },
        **{name: (*spin_prefix, n) for name in ("orbital_energies", "occupations")},
        "orbital_gradient": (n - 1,),
        "energy": (),
        "nuclear_energy": (),
        "gradient": (2, 3),
        "forces": (2, 3),
    }


def encode_array(value: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    if (
        array.dtype != np.dtype("float64")
        or array.shape != shape
        or not np.isfinite(array).all()
    ):
        raise ValueError("scientific output is not complete finite real FP64")
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
    if (
        set(record)
        != {"dtype", "shape", "order", "encoding", "bytes", "sha256_le_f64", "data"}
        or record["dtype"] != "float64"
        or record["shape"] != list(shape)
        or any(type(n) is not int for n in record["shape"])
        or record["order"] != "C"
        or record["encoding"] != "base64-le-f64"
        or type(record["bytes"]) is not int
        or record["bytes"] != math.prod(shape) * 8
    ):
        raise ValueError("incomplete array or wrong shape/type/encoding")
    raw = base64.b64decode(record["data"], validate=True)
    if (
        len(raw) != record["bytes"]
        or audit.digest(raw) != record["sha256_le_f64"]
        or base64.b64encode(raw).decode("ascii") != record["data"]
    ):
        raise ValueError("scientific array byte identity mismatch")
    values = struct.unpack(f"<{math.prod(shape)}d", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite scientific data")
    return values


def validate_receipt(receipt: dict[str, Any], inputs: dict[str, Any]) -> None:
    if (
        set(receipt)
        != {
            "schema",
            "inputs",
            "profile",
            "replica",
            "process_id",
            "provenance",
            "molecule",
            "scf",
            "timings",
            "arrays",
        }
        or receipt["schema"] != "generativeqc.ecp-f-hf.receipt.v1"
        or receipt["inputs"] != inputs
        or receipt["profile"] not in profiles(inputs)
        or receipt["replica"] not in ("first", "second")
        or type(receipt["process_id"]) is not int
        or receipt["process_id"] <= 0
    ):
        raise ValueError("fixed receipt identity mismatch")
    profile = profiles(inputs)[receipt["profile"]]
    n = profile["nao"]
    mol = receipt["molecule"]
    cart = profile["cart"]
    if (
        mol["basis"] != inputs["resolved_basis"]
        or mol["ecp"] != inputs["resolved_ecp"]
        or mol["atoms_bohr"] != inputs["atoms"]
        or mol["unit"] != "Bohr"
        or type(mol["cart"]) is not bool
        or mol["cart"] != cart
        or type(mol["nao"]) is not int
        or mol["nao"] != n
        or type(mol["charge"]) is not int
        or mol["charge"] != profile["charge"]
        or type(mol["spin"]) is not int
        or mol["spin"] != profile["spin"]
        or mol["nelec"] != profile["nelec"]
        or mol["core_electrons"] != [10, 0]
        or mol["effective_nuclear_charges"] != [1, 1]
        or mol["ao_locations"] != [0, 1, 2, 5, 8, n - 1, n]
        or mol["atom_ao_slices"] != [[0, 5, 0, n - 1], [5, 6, n - 1, n]]
        or len(mol["ao_labels"]) != n
        or len(set(mol["ao_labels"])) != n
        or len(mol["ao_labels_structured"]) != n
    ):
        raise ValueError("resolved NaH shell/AO/electron identity mismatch")
    settings = receipt["scf"]
    if (
        settings["converged"] is not True
        or settings["method"] != profile["method"]
        or settings["resolved"]["init_guess"] != "minao"
        or settings["resolved"]["conv_tol"] != 1e-12
        or settings["resolved"]["conv_tol_grad"] is not None
        or settings["resolved"]["max_cycle"] != 50
        or settings["resolved"]["direct_scf"] is not True
        or set(settings["resolved"]) != {*SCF_FIELDS, "diis", "diis_class"}
        or type(settings["cycles"]) is not int
        or not 0 <= settings["cycles"] <= 50
        or (profile["spin"] == 1 and settings["cycles"] != 0)
        or (profile["spin"] == 0 and settings["cycles"] == 0)
        or settings["actual_class"]
        != ("pyscf.scf.uhf.HF1e" if profile["spin"] else "pyscf.scf.hf.RHF")
        or settings["initial_guess_used"] is not (profile["spin"] == 0)
    ):
        raise ValueError("SCF did not preserve the original default-guess protocol")
    if set(receipt["timings"]) != {"scf_seconds", "gradient_seconds"} or any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in receipt["timings"].values()
    ):
        raise ValueError("missing actual SCF/gradient timing")
    expected_shapes = shapes(profile)
    if set(receipt["arrays"]) != set(expected_shapes):
        raise ValueError("incomplete full SCF/force array inventory")
    arrays = {
        name: decode_array(receipt["arrays"][name], shape)
        for name, shape in expected_shapes.items()
    }
    if any(
        force != -grad
        for force, grad in zip(arrays["forces"], arrays["gradient"], strict=True)
    ):
        raise ValueError("force is not the complete negative analytic gradient")
    if any(
        abs(arrays["forces"][axis] + arrays["forces"][3 + axis]) > 2e-7
        for axis in range(3)
    ):
        raise ValueError("independent net force is not translationally invariant")
    occupations = (
        ((2.0,) + (0.0,) * (n - 1))
        if not profile["spin"]
        else ((1.0,) + (0.0,) * (2 * n - 1))
    )
    if arrays["occupations"] != occupations:
        raise ValueError("incorrect RHF/UHF occupations")
    if math.sqrt(sum(value * value for value in arrays["orbital_gradient"])) > 1e-5:
        raise ValueError(
            "independent SCF orbital gradient exceeds original convergence gate"
        )
    for name in ("overlap", "hcore", "ecp_raw", "ecp_normalized", "density", "fock"):
        values = arrays[name]
        for block in range(len(values) // (n * n)):
            offset = block * n * n
            if any(
                abs(values[offset + i * n + j] - values[offset + j * n + i]) > 1e-11
                for i in range(n)
                for j in range(n)
            ):
                raise ValueError("nonsymmetric independent one-electron matrix")
    for i in range(n):
        norm = arrays["normalization"][i]
        if norm <= 0 or abs(norm * norm - arrays["overlap"][i * n + i]) > 1e-12:
            raise ValueError("incorrect AO normalization")
        for j in range(n):
            if (
                abs(
                    arrays["ecp_raw"][i * n + j] / norm / arrays["normalization"][j]
                    - arrays["ecp_normalized"][i * n + j]
                )
                > REPLAY_ATOL
            ):
                raise ValueError("incomplete or incorrect normalized ECP")
    validate_scf_arrays(arrays, profile)
    validate_provenance(receipt["provenance"], inputs)


def validate_scf_arrays(
    arrays: dict[str, tuple[float, ...]], profile: dict[str, Any]
) -> None:
    """Check complete-array consistency without solving or altering the oracle."""
    import numpy as np

    values = {
        name: np.array(arrays[name]).reshape(shape)
        for name, shape in shapes(profile).items()
    }
    n = profile["nao"]
    overlap, hcore = values["overlap"], values["hcore"]
    blocks = (0, 1) if profile["spin"] else (None,)
    electronic = 0.0
    for block in blocks:
        density, fock, coefficients = (
            values[name] if block is None else values[name][block]
            for name in ("density", "fock", "coefficients")
        )
        occupations = (
            values["occupations"] if block is None else values["occupations"][block]
        )
        population = sum(profile["nelec"]) if block is None else profile["nelec"][block]
        for actual, expected, tolerance in (
            (coefficients.T @ overlap @ coefficients, np.eye(n), 1e-10),
            (density, (coefficients * occupations) @ coefficients.T, 1e-12),
            (np.trace(density @ overlap), population, 1e-10),
            (
                values["commutator_residual"]
                if block is None
                else values["commutator_residual"][block],
                fock @ density @ overlap - overlap @ density @ fock,
                1e-13,
            ),
        ):
            if not np.allclose(actual, expected, atol=tolerance, rtol=0):
                raise ValueError("full SCF state/residual is internally inconsistent")
        electronic += 0.5 * float(np.einsum("ij,ji->", density, hcore + fock))
    if abs(float(values["energy"] - values["nuclear_energy"]) - electronic) > 1e-10:
        raise ValueError("full density/Fock state does not reconstruct the HF energy")


def cpu_provider_records() -> list[dict[str, Any]]:
    """Observe the actual loaded CPU providers, not claims from a receipt."""
    from threadpoolctl import threadpool_info

    allowed = {
        "blas": {"openblas", "mkl", "blis", "flexiblas"},
        "openmp": {"openmp"},
    }
    records = []
    for pool in threadpool_info():
        if (
            pool.get("internal_api") not in allowed.get(pool.get("user_api"), set())
            or type(pool.get("num_threads")) is not int
            or pool["num_threads"] != 1
        ):
            raise ValueError(
                "loaded native provider is not an admitted one-thread CPU API"
            )
        path = Path(pool["filepath"])
        if not path.is_file():
            raise ValueError("loaded CPU provider has no readable native file")
        records.append(
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
    if not records:
        raise ValueError("no loaded CPU provider evidence")
    return sorted(records, key=audit.canonical)


def package_records(inputs: dict[str, Any]) -> dict[str, Any]:
    """Recompute complete wheel/source/native identities for every pinned package."""
    return {
        name: audit.distribution_identity(name, version)
        for name, version in inputs["packages"].items()
    }


def validate_provenance(provenance: dict[str, Any], inputs: dict[str, Any]) -> None:
    if (
        set(provenance)
        != {
            "inputs_sha256",
            "generator_sha256",
            "source_sha256",
            "execution",
            "configuration",
            "python",
            "platform",
            "machine",
            "packages",
            "threadpools",
            "installed_integral_inputs",
            "installed_scf_sources",
            "device",
            "cuda_visible_devices",
        }
        or provenance["platform"] != platform.platform()
        or provenance["machine"] != platform.machine()
        or provenance["inputs_sha256"] != audit.digest(INPUTS.read_bytes())
        or provenance["generator_sha256"] != audit.digest(Path(__file__).read_bytes())
        or provenance["source_sha256"]
        != {entry["path"]: entry["sha256"] for entry in inputs["sources"]}
        or provenance["configuration"]
        != {"policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")}
        or provenance["python"] != inputs["python"]
        or provenance["device"] != "CPU"
        or provenance["cuda_visible_devices"] != ""
        or provenance["execution"] != audit.execution_identity()
        or audit.canonical(provenance["packages"])
        != audit.canonical(package_records(inputs))
        or provenance["installed_integral_inputs"]
        != ecp_audit.installed_integral_inputs(inputs)
        or provenance["installed_scf_sources"] != installed_scf_sources(inputs)
        or audit.canonical(provenance["threadpools"])
        != audit.canonical(cpu_provider_records())
    ):
        raise ValueError("acquisition provenance mismatch")


def generate(inputs: dict[str, Any], profile_id: str, replica: str) -> dict[str, Any]:
    validate_checkout(inputs, runtime=True)
    if any(
        name == "pyscf" or name.startswith(("pyscf.", "generativeqc"))
        for name in sys.modules
    ):
        raise ValueError("requires fresh interpreter without PySCF or GenerativeQC")
    if any(name.startswith("PYSCF_") for name in os.environ):
        raise ValueError("unset external PySCF configuration")
    inventory = audit.module_inventory(inputs)
    with tempfile.TemporaryDirectory(prefix="gqc-ecp-f-config-") as directory:
        config = Path(directory) / "empty.py"
        config.write_bytes(b"")
        os.environ["PYSCF_CONFIG_FILE"] = str(config)
        try:
            return calculate(inputs, profile_id, replica, config, inventory)
        finally:
            os.environ.pop("PYSCF_CONFIG_FILE", None)


def calculate(
    inputs: dict[str, Any],
    profile_id: str,
    replica: str,
    config: Path,
    inventory: dict[str, set[Path]],
) -> dict[str, Any]:
    import numpy as np
    import pyscf
    from pyscf import gto, scf

    if (
        Path(pyscf.__config__.conf_file).resolve() != config.resolve()
        or config.read_bytes() != b""
        or pyscf.__version__ != inputs["packages"]["pyscf"]
    ):
        raise ValueError("PySCF configuration/version identity mismatch")
    audit.check_imported_modules(inventory)
    packages = package_records(inputs)
    installed = ecp_audit.installed_integral_inputs(inputs)
    upstream_scf = installed_scf_sources(inputs)
    profile = profiles(inputs)[profile_id]
    with pyscf.lib.with_omp_threads(1):
        basis = {
            atom: gto.basis.load(name, atom)
            for atom, name in inputs["basis_libraries"].items()
        }
        basis["Na"].append(json.loads(audit.canonical(inputs["extra_f_shell"])))
        ecp = {
            atom: gto.basis.load_ecp(name, atom)
            for atom, name in inputs["ecp_libraries"].items()
        }
        if basis != inputs["resolved_basis"] or ecp != inputs["resolved_ecp"]:
            raise ValueError(
                "independently resolved basis/ECP differs from exact input"
            )
        mol = gto.M(
            atom=inputs["atoms"],
            basis=basis,
            ecp=ecp,
            unit="Bohr",
            charge=profile["charge"],
            spin=profile["spin"],
            cart=profile["cart"],
            verbose=0,
        )
        reference = scf.UHF(mol) if profile["spin"] else scf.RHF(mol)
        # Exact original call: no forced dm0, alternative guess, polish, or retry.
        start = time.perf_counter()
        reference.run(conv_tol=1e-12)
        scf_seconds = time.perf_counter() - start
        if not reference.converged:
            raise ValueError("original independent HF endpoint did not converge")
        start = time.perf_counter()
        gradient = reference.nuc_grad_method().kernel()
        gradient_seconds = time.perf_counter() - start
        overlap = reference.get_ovlp()
        density = reference.make_rdm1()
        fock = reference.get_fock(dm=density)
        raw = mol.intor("ECPscalar")
        norms = np.sqrt(overlap.diagonal())
        values = {
            "overlap": overlap,
            "hcore": reference.get_hcore(),
            "ecp_raw": raw,
            "ecp_normalized": raw / norms[:, None] / norms[None, :],
            "normalization": norms,
            "density": density,
            "fock": fock,
            "coefficients": reference.mo_coeff,
            "orbital_energies": reference.mo_energy,
            "occupations": reference.mo_occ,
            "gradient": gradient,
            "forces": -gradient,
            "energy": reference.e_tot,
            "nuclear_energy": mol.energy_nuc(),
            "commutator_residual": fock @ density @ overlap - overlap @ density @ fock,
            "orbital_gradient": reference.get_grad(
                reference.mo_coeff, reference.mo_occ, fock
            ),
        }
        arrays = {
            name: encode_array(values[name], shape)
            for name, shape in shapes(profile).items()
        }
        pools = cpu_provider_records()
    audit.check_imported_modules(inventory)
    if any(
        name == "cupy" or name.startswith(("generativeqc", "cupy."))
        for name in sys.modules
    ):
        raise ValueError("unexpected GQC/GPU scientific import")
    resolved = {name: getattr(reference, name) for name in SCF_FIELDS}
    resolved.update(
        diis=bool(reference.diis),
        diis_class=f"{reference.DIIS.__module__}.{reference.DIIS.__name__}",
    )
    receipt = {
        "schema": "generativeqc.ecp-f-hf.receipt.v1",
        "inputs": inputs,
        "profile": profile_id,
        "replica": replica,
        "process_id": os.getpid(),
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
            "installed_integral_inputs": installed,
            "installed_scf_sources": upstream_scf,
            "device": "CPU",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "molecule": {
            "basis": mol._basis,
            "ecp": mol._ecp,
            "atoms_bohr": [
                [mol.atom_symbol(i), mol.atom_coord(i).tolist()] for i in range(2)
            ],
            "unit": "Bohr",
            "nao": mol.nao_nr(),
            "nelec": list(mol.nelec),
            "charge": mol.charge,
            "spin": mol.spin,
            "cart": mol.cart,
            "core_electrons": [mol.atom_nelec_core(i) for i in range(2)],
            "effective_nuclear_charges": [mol.atom_charge(i) for i in range(2)],
            "ao_labels": mol.ao_labels(),
            "ao_labels_structured": [list(label) for label in mol.ao_labels(fmt=False)],
            "ao_locations": mol.ao_loc_nr().tolist(),
            "atom_ao_slices": mol.aoslice_by_atom().tolist(),
        },
        "scf": {
            "method": profile["method"],
            "actual_class": f"{type(reference).__module__}.{type(reference).__name__}",
            "initial_guess_used": profile["spin"] == 0,
            "converged": bool(reference.converged),
            "cycles": int(reference.cycles),
            "resolved": resolved,
        },
        "timings": {"scf_seconds": scf_seconds, "gradient_seconds": gradient_seconds},
        "arrays": arrays,
    }
    validate_receipt(receipt, inputs)
    return receipt


def verify(paths: list[Path], inputs: dict[str, Any]) -> dict[str, Any]:
    if len(paths) != 8 or len({path.resolve() for path in paths}) != 8:
        raise ValueError("eight distinct original process receipts required")
    if any(
        name == "pyscf" or name.startswith(("pyscf.", "generativeqc"))
        for name in sys.modules
    ):
        raise ValueError(
            "verification requires a fresh interpreter without PySCF or GenerativeQC"
        )
    if any(name.startswith("PYSCF_") for name in os.environ):
        raise ValueError("unset external PySCF configuration")
    inventory = audit.module_inventory(inputs)
    with tempfile.TemporaryDirectory(prefix="gqc-ecp-f-verify-config-") as directory:
        config = Path(directory) / "empty.py"
        config.write_bytes(b"")
        os.environ["PYSCF_CONFIG_FILE"] = str(config)
        try:
            # Imports only: initialize the same CPU BLAS/OpenMP providers for
            # independent observation. No molecule, SCF or gradient is evaluated.
            for module in (
                "pyscf.gto",
                "pyscf.scf",
                "pyscf.grad.rhf",
                "pyscf.grad.uhf",
            ):
                importlib.import_module(module)
            pyscf = sys.modules["pyscf"]
            if (
                Path(pyscf.__config__.conf_file).resolve() != config.resolve()
                or config.read_bytes() != b""
                or pyscf.__version__ != inputs["packages"]["pyscf"]
            ):
                raise ValueError("verification configuration/version identity mismatch")
            audit.check_imported_modules(inventory)
            with pyscf.lib.with_omp_threads(1):
                result = verify_records(paths, inputs)
            audit.check_imported_modules(inventory)
            return result
        finally:
            os.environ.pop("PYSCF_CONFIG_FILE", None)


def verify_records(paths: list[Path], inputs: dict[str, Any]) -> dict[str, Any]:
    if len(paths) != 8 or len({path.resolve() for path in paths}) != 8:
        raise ValueError("eight distinct original process receipts required")
    records = {}
    provenance = None
    pids = set()
    for path in paths:
        record = read_json(path)
        validate_receipt(record, inputs)
        key = (record["profile"], record["replica"])
        if key in records or record["process_id"] in pids:
            raise ValueError("duplicate profile/replica or reused process")
        pids.add(record["process_id"])
        if provenance is None:
            provenance = record["provenance"]
        if record["provenance"] != provenance:
            raise ValueError(
                "independent processes do not share acquisition provenance"
            )
        records[key] = (record, path)
    expected = {
        (profile, replica)
        for profile in profiles(inputs)
        for replica in ("first", "second")
    }
    if set(records) != expected:
        raise ValueError("incomplete quartet/two-process inventory")
    bindings = []
    for profile_id, profile in profiles(inputs).items():
        first, first_path = records[profile_id, "first"]
        second, second_path = records[profile_id, "second"]
        if first["molecule"] != second["molecule"] or first["scf"] != second["scf"]:
            raise ValueError("independent reconstructed molecule/SCF protocol differs")
        differences = {}
        for name, shape in shapes(profile).items():
            left = decode_array(first["arrays"][name], shape)
            right = decode_array(second["arrays"][name], shape)
            differences[name] = max(
                abs(a - b) for a, b in zip(left, right, strict=True)
            )
        if max(differences.values()) > REPLAY_ATOL:
            raise ValueError("independent complete-array replay disagrees")
        bindings.append(
            {
                "profile": profile_id,
                "molecule": first["molecule"],
                "scf": first["scf"],
                "arrays": {
                    name: {key: value for key, value in array.items() if key != "data"}
                    for name, array in first["arrays"].items()
                },
                "replicas": [
                    {
                        "path": path.name,
                        "replica": record["replica"],
                        "process_id": record["process_id"],
                        "bytes": path.stat().st_size,
                        "sha256": audit.digest(path.read_bytes()),
                        "timings": record["timings"],
                    }
                    for record, path in ((first, first_path), (second, second_path))
                ],
                "replay": {
                    "absolute_tolerance": REPLAY_ATOL,
                    "max_absolute_difference_by_array": differences,
                },
            }
        )
    return {
        "schema": "generativeqc.ecp-f-hf.manifest.v1",
        "inputs_sha256": audit.digest(INPUTS.read_bytes()),
        "provenance": provenance,
        "profiles": bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--replica", choices=("first", "second"))
    parser.add_argument("--verify", nargs=8, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inputs = read_json(INPUTS)
    validate_inputs(inputs)
    if args.check_inputs:
        if args.profile or args.replica or args.verify or args.output:
            parser.error("--check-inputs must be used alone")
        validate_checkout(inputs, runtime=False)
        print(
            "Exact quartet input/source preflight passed; no scientific acquisition executed"
        )
        return
    if args.output is None or bool(args.verify) == bool(args.profile and args.replica):
        parser.error("provide --output with either --verify or --profile and --replica")
    validate_checkout(inputs, runtime=True)
    if args.verify:
        result = verify(args.verify, inputs)
    else:
        if args.profile not in profiles(inputs):
            parser.error("unknown fixed profile")
        result = generate(inputs, args.profile, args.replica)
    raw = (
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("bounded receipt exceeded byte cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(f"Retained {args.output.name}: bytes={len(raw)} sha256={audit.digest(raw)}")


if __name__ == "__main__":
    main()
