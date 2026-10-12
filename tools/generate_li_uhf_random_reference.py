"""Acquire the complete fixed Li/UHF seed-37 oracle in fresh PySCF processes.

No GenerativeQC implementation is imported. All scientific arrays are complete,
losslessly encoded C-order little-endian FP64, rather than selected entries.
"""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import itertools
import json
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import generate_cpu_hf_high_l_reference as audit

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "tests/reference_data/li_uhf_random/inputs.json"
MAX_BYTES = 256 * 1024
BLOCK_SHAPES = {
    "t1a": (2, 3), "t1b": (1, 4), "t2aa": (2, 2, 3, 3),
    "t2ab": (2, 1, 3, 4), "t2bb": (1, 1, 4, 4),
}
CHANNELS = ("aaa", "bbb", "aab", "abb")
SHAPES = {
    **{name: (5, 5) for name in (
        "overlap", "hcore", "density_alpha", "density_beta",
        "fock_ao_alpha", "fock_ao_beta", "coefficients_alpha", "coefficients_beta",
        "fock_mo_alpha", "fock_mo_beta",
    )},
    **{name: (5,) for name in (
        "orbital_energies_alpha", "orbital_energies_beta",
        "occupations_alpha", "occupations_beta", "eris_energies_alpha", "eris_energies_beta",
    )},
    "ao_eri": (5, 5, 5, 5), "fock_spin": (10, 10),
    "antisymmetrized_eri": (10, 10, 10, 10), "packed_amplitudes": (37,),
    **{f"{group}.{name}": shape for group in (
        "amplitudes", "updated_amplitudes", "denominators", "residuals"
    ) for name, shape in BLOCK_SHAPES.items()},
    **{name: () for name in (
        "hf_energy", "correlation_energy", "triples_slow_total", "triples_optimized_total"
    )},
    **{f"channels.{name}": () for name in CHANNELS},
}
REPLAY_ATOL = 1e-13


def encode_array(value: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.shape != shape:
        raise ValueError("scientific output is not complete real-FP64 of the declared shape")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite independent output")
    raw = array.astype("<f8", copy=False).tobytes(order="C")
    return {
        "dtype": "float64", "shape": list(shape), "order": "C",
        "encoding": "base64-le-f64", "bytes": len(raw),
        "sha256_le_f64": audit.digest(raw), "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(record: dict[str, Any], shape: tuple[int, ...]) -> tuple[float, ...]:
    if set(record) != {"dtype", "shape", "order", "encoding", "bytes", "sha256_le_f64", "data"}:
        raise ValueError("incorrect complete array envelope")
    if (
        record["dtype"] != "float64" or record["shape"] != list(shape)
        or record["order"] != "C" or record["encoding"] != "base64-le-f64"
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
        inputs["schema"] != "generativeqc.li-uhf-random.inputs.v1"
        or inputs["atoms"] != [["Li", [0, 0, 0]]]
        or inputs["unit"] != "Angstrom" or inputs["basis_name"] != "sto-3g"
        or type(inputs["charge"]) is not int or inputs["charge"] != 0
        or type(inputs["spin"]) is not int or inputs["spin"] != 1
        or inputs["cart"] is not False
        or inputs["populations"] != {"oa": 2, "ob": 1, "va": 3, "vb": 4}
        or inputs["scf"] != {"method": "UHF", "conv_tol": 1e-13, "initial_guess": "minao"}
        or type(inputs["random"]["seed"]) is not int
        or inputs["random"]["seed"] != 37 or inputs["random"]["scale"] != 0.04
        or inputs["random"]["generator"] != "numpy.random.default_rng"
        or inputs["random"]["same_spin_representatives"] != "i<j,a<b; occupied pair major, virtual pair minor; +ijab,-jiab,-ijba,+jiba"
        or inputs["random"]["mixed_spin_axes"] != ["occupied_alpha", "occupied_beta", "virtual_alpha", "virtual_beta"]
        or inputs["spin_orbital_order"] != inputs["random"]["mixed_spin_axes"]
        or inputs["precision"] != "float64"
        or inputs["array_encoding"] != "base64 little-endian C-order FP64"
        or inputs["eri_convention"] != "g[p,q,r,s]=(p r|q s)-(p s|q r); chemist AO uvwx and exact spin masks"
        or inputs["preserved_gates"] != {"residual_atol": 3e-11, "residual_rtol": 3e-11, "triples_atol": 3e-13, "triples_rtol": 3e-13, "impossible_channel_atol": 2e-13}
        or inputs["random"]["packed_block_order"] != list(BLOCK_SHAPES)
        or inputs["python"] != "3.12.12"
        or platform.python_version() != inputs["python"]
        or inputs["packages"] != {
            "pyscf": "2.14.0", "numpy": "2.3.5", "scipy": "1.17.0",
            "h5py": "3.16.0", "threadpoolctl": "3.6.0",
        }
    ):
        raise ValueError("outside the one reviewed fixed Li random-amplitude profile")
    expected_paths = {
        "tests/python/test_uccsd_equations.py", "tests/python/test_uccsd_triples.py",
        "tools/generate_cpu_hf_high_l_reference.py",
        "tools/generativeqc_cc/uccsd_triples_source_manifest.json",
    }
    if {entry["path"] for entry in inputs["sources"]} != expected_paths:
        raise ValueError("incomplete reviewed source inventory")
    for entry in inputs["sources"]:
        if audit.digest((ROOT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("reviewed source/helper changed")
    if Path(audit.__file__).resolve() != ROOT / "tools/generate_cpu_hf_high_l_reference.py":
        raise ValueError("unexpected imported audit helper")
    manifest = audit.read_json(ROOT / "tools/generativeqc_cc/uccsd_triples_source_manifest.json")
    if inputs["upstream_triples"] != manifest:
        raise ValueError("upstream scientific source pins changed")


def installed_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("pyscf")
    wanted = {inputs["basis_data"]["path"]: inputs["basis_data"]["sha256"]}
    wanted.update({entry["path"]: entry["sha256"] for entry in inputs["upstream_triples"]["files"]})
    files = {}
    for entry in distribution.files or ():
        relative = str(entry)
        if relative in wanted:
            raw = Path(distribution.locate_file(entry)).read_bytes()
            if audit.digest(raw) != wanted[relative]:
                raise ValueError("installed basis or upstream triples source differs")
            files[relative] = {"bytes": len(raw), "sha256": audit.digest(raw)}
    if set(files) != set(wanted):
        raise ValueError("missing pinned installed scientific source")
    return files


def independent_amplitudes() -> tuple[Any, dict[str, Any]]:
    import numpy as np

    packed = np.random.default_rng(37).normal(scale=0.04, size=37)
    blocks = {}
    offset = 0
    for name, shape in BLOCK_SHAPES.items():
        if name in ("t2aa", "t2bb"):
            values = np.zeros(shape, dtype=np.float64)
            for i, j in itertools.combinations(range(shape[0]), 2):
                for a, b in itertools.combinations(range(shape[2]), 2):
                    value = packed[offset]
                    offset += 1
                    values[i, j, a, b] = values[j, i, b, a] = value
                    values[j, i, a, b] = values[i, j, b, a] = -value
        else:
            size = math.prod(shape)
            values = packed[offset:offset + size].reshape(shape).copy()
            offset += size
        blocks[name] = values
    if offset != 37:
        raise ValueError("independent amplitude inventory mismatch")
    return packed, blocks


def calculate(inputs: dict[str, Any], replica: str, config: Path, inventory: dict) -> dict:
    import numpy as np
    import pyscf
    from pyscf import cc, gto, scf
    from pyscf.cc import uccsd_t_slow
    from threadpoolctl import threadpool_info

    if (
        Path(pyscf.__config__.conf_file).resolve() != config.resolve()
        or config.read_bytes() != b""
        or pyscf.__version__ != inputs["packages"]["pyscf"]
    ):
        raise ValueError("PySCF configuration/version identity mismatch")
    audit.check_imported_modules(inventory)
    pinned_files = installed_inputs(inputs)
    packages = {name: audit.distribution_identity(name, version) for name, version in inputs["packages"].items()}
    with pyscf.lib.with_omp_threads(1):
        basis = {"Li": gto.basis.load("sto-3g", "Li")}
        if basis != inputs["resolved_basis"]:
            raise ValueError("resolved STO-3G contractions differ from the reviewed input")
        molecule = gto.M(
            atom=inputs["atoms"], unit=inputs["unit"], basis=basis,
            charge=0, spin=1, cart=False, verbose=0,
        )
        if molecule.nao_nr() != 5 or molecule.nelec != (2, 1) or molecule._basis != basis:
            raise ValueError("unexpected resolved Li molecule")
        reference = scf.UHF(molecule)
        reference.conv_tol = 1e-13
        reference.kernel()
        if not reference.converged or reference.init_guess != "minao":
            raise ValueError("fixed UHF reference did not converge with the original guess")
        coupled_cluster = cc.CCSD(reference)
        eris = coupled_cluster.ao2mo()
        coeff_a, coeff_b = reference.mo_coeff
        coefficients = np.concatenate((coeff_a[:, :2], coeff_b[:, :1], coeff_a[:, 2:], coeff_b[:, 1:]), axis=1)
        spins = np.repeat((0, 1, 0, 1), (2, 1, 3, 4))
        ao_eri = molecule.intor("int2e")
        coulomb = np.einsum("uvwx,up,vr,wq,xs->pqrs", ao_eri, coefficients, coefficients, coefficients, coefficients, optimize=True)
        coulomb *= (spins[:, None, None, None] == spins[None, None, :, None]) & (spins[None, :, None, None] == spins[None, None, None, :])
        integrals = coulomb - coulomb.swapaxes(2, 3)
        fock = np.zeros((10, 10))
        for spin_value, source in ((0, eris.focka), (1, eris.fockb)):
            positions = np.flatnonzero(spins == spin_value)
            fock[np.ix_(positions, positions)] = source
        packed, amplitudes = independent_amplitudes()
        singles = (amplitudes["t1a"], amplitudes["t1b"])
        doubles = (amplitudes["t2aa"], amplitudes["t2ab"], amplitudes["t2bb"])
        new_singles, new_doubles = coupled_cluster.update_amps(singles, doubles, eris)
        updates = dict(zip(BLOCK_SHAPES, (*new_singles, *new_doubles), strict=True))
        domains = {
            "t1a": ("oa", "va"), "t1b": ("ob", "vb"),
            "t2aa": ("oa", "oa", "va", "va"), "t2ab": ("oa", "ob", "va", "vb"),
            "t2bb": ("ob", "ob", "vb", "vb"),
        }
        offsets = {"oa": 0, "ob": 2, "va": 3, "vb": 6}
        diagonal = np.diag(fock)
        denominators, residuals = {}, {}
        for name, shape in BLOCK_SHAPES.items():
            denominator = np.zeros(shape)
            for axis, domain in enumerate(domains[name]):
                axis_shape = [1] * len(shape)
                axis_shape[axis] = shape[axis]
                start = offsets[domain]
                denominator += (1 if domain.startswith("o") else -1) * diagonal[start:start + shape[axis]].reshape(axis_shape)
            denominators[name] = denominator
            residuals[name] = denominator * (updates[name] - amplitudes[name])
        correlation = float(coupled_cluster.energy(singles, doubles, eris))
        contributions = []

        def contraction(equation: str, *operands: Any, **kwargs: Any) -> Any:
            result = np.einsum(equation, *operands, **kwargs)
            if np.ndim(result) == 0 and all(np.ndim(value) == 6 for value in operands):
                contributions.append(float(result) / 4)
            return result

        with patch.object(uccsd_t_slow, "numpy", SimpleNamespace(einsum=contraction, asarray=np.asarray)):
            slow_total = float(uccsd_t_slow.kernel(coupled_cluster, eris, singles, doubles))
        optimized_total = float(coupled_cluster.ccsd_t(singles, doubles, eris))
        channels = dict(zip(CHANNELS, contributions, strict=True))
        np.testing.assert_allclose(optimized_total, slow_total, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(sum(channels.values()), slow_total, atol=3e-13, rtol=3e-13)
        density = reference.make_rdm1()
        fock_ao = reference.get_fock()
        arrays = {
            "overlap": reference.get_ovlp(), "hcore": reference.get_hcore(),
            "density_alpha": density[0], "density_beta": density[1],
            "fock_ao_alpha": fock_ao[0], "fock_ao_beta": fock_ao[1],
            "coefficients_alpha": coeff_a, "coefficients_beta": coeff_b,
            "orbital_energies_alpha": reference.mo_energy[0],
            "orbital_energies_beta": reference.mo_energy[1],
            "occupations_alpha": reference.mo_occ[0], "occupations_beta": reference.mo_occ[1],
            "fock_mo_alpha": eris.focka, "fock_mo_beta": eris.fockb,
            "eris_energies_alpha": eris.mo_energy[0], "eris_energies_beta": eris.mo_energy[1],
            "ao_eri": ao_eri, "fock_spin": fock, "antisymmetrized_eri": integrals,
            "packed_amplitudes": packed, "hf_energy": float(reference.e_tot),
            "correlation_energy": correlation, "triples_slow_total": slow_total,
            "triples_optimized_total": optimized_total,
            **{f"channels.{name}": value for name, value in channels.items()},
            **{f"{group}.{name}": value for group, blocks in (
                ("amplitudes", amplitudes), ("updated_amplitudes", updates),
                ("denominators", denominators), ("residuals", residuals)
            ) for name, value in blocks.items()},
        }
        encoded = {name: encode_array(arrays[name], shape) for name, shape in SHAPES.items()}
        pools = []
        for pool in threadpool_info():
            path = Path(pool["filepath"])
            pools.append({key: pool.get(key) for key in (
                "user_api", "internal_api", "version", "num_threads", "architecture"
            )} | {"library": path.name, "sha256": audit.digest(path.read_bytes())})
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("native scientific thread pool is not single-threaded")
    audit.check_imported_modules(inventory)
    receipt = {
        "schema": "generativeqc.li-uhf-random.receipt.v1", "replica": replica,
        "process_id": os.getpid(), "inputs": inputs,
        "provenance": {
            "inputs_sha256": audit.digest(INPUTS.read_bytes()),
            "generator_sha256": audit.digest(Path(__file__).read_bytes()),
            "source_sha256": {entry["path"]: entry["sha256"] for entry in inputs["sources"]},
            "execution": audit.execution_identity(),
            "configuration": {"policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")},
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "threadpools": pools, "installed_scientific_inputs": pinned_files,
        },
        "molecule": {
            "basis": basis, "atoms_bohr": [["Li", molecule.atom_coord(0).tolist()]],
            "ao_labels": molecule.ao_labels(),
            "ao_labels_structured": [list(label) for label in molecule.ao_labels(fmt=False)],
            "ao_locations": molecule.ao_loc_nr().tolist(),
            "nao": molecule.nao_nr(), "nelec": list(molecule.nelec),
            "charge": molecule.charge, "spin": molecule.spin, "cart": molecule.cart,
            "scf_converged": bool(reference.converged),
            "resolved_scf": {name: getattr(reference, name) for name in (
                "init_guess", "conv_tol", "conv_tol_grad", "max_cycle", "diis_space",
                "direct_scf", "direct_scf_tol", "level_shift", "damp",
            )},
        },
        "arrays": encoded,
    }
    validate(receipt, inputs)
    return receipt


def validate(receipt: dict[str, Any], inputs: dict[str, Any]) -> None:
    if (
        set(receipt) != {"schema", "replica", "process_id", "inputs", "provenance", "molecule", "arrays"}
        or receipt["schema"] != "generativeqc.li-uhf-random.receipt.v1"
        or receipt["replica"] not in ("first", "second")
        or type(receipt["process_id"]) is not int or receipt["process_id"] <= 0
        or receipt["inputs"] != inputs or set(receipt["arrays"]) != set(SHAPES)
    ):
        raise ValueError("invalid fixed receipt identity/inventory")
    provenance = receipt["provenance"]
    if (
        provenance["inputs_sha256"] != audit.digest(INPUTS.read_bytes())
        or provenance["generator_sha256"] != audit.digest(Path(__file__).read_bytes())
        or provenance["source_sha256"] != {entry["path"]: entry["sha256"] for entry in inputs["sources"]}
        or provenance["execution"] != audit.execution_identity()
        or provenance["python"] != inputs["python"]
        or provenance["configuration"] != {"policy": "explicit-empty-regular-file", "sha256": audit.digest(b"")}
        or provenance["installed_scientific_inputs"] != installed_inputs(inputs)
        or {name: entry["version"] for name, entry in provenance["packages"].items()} != inputs["packages"]
    ):
        raise ValueError("acquisition provenance mismatch")
    molecule = receipt["molecule"]
    if (
        type(molecule["nao"]) is not int or molecule["nao"] != 5
        or molecule["nelec"] != [2, 1]
        or any(type(value) is not int for value in molecule["nelec"])
        or molecule["basis"] != inputs["resolved_basis"]
        or molecule["ao_locations"] != inputs["ao_identity"]["locations"]
        or molecule["ao_labels_structured"] != inputs["ao_identity"]["labels"]
        or molecule["atoms_bohr"] != [["Li", [0, 0, 0]]]
        or type(molecule["charge"]) is not int or molecule["charge"] != 0
        or type(molecule["spin"]) is not int or molecule["spin"] != 1
        or molecule["cart"] is not False
        or molecule["scf_converged"] is not True or len(molecule["ao_labels"]) != 5
        or len(set(molecule["ao_labels"])) != 5
        or molecule["resolved_scf"]["conv_tol"] != 1e-13
        or molecule["resolved_scf"]["init_guess"] != "minao"
    ):
        raise ValueError("resolved Li UHF identity mismatch")
    arrays = {name: decode_array(receipt["arrays"][name], shape) for name, shape in SHAPES.items()}
    if arrays["occupations_alpha"] != (1, 1, 0, 0, 0) or arrays["occupations_beta"] != (1, 0, 0, 0, 0):
        raise ValueError("incorrect spin occupations")
    for name in ("overlap", "hcore", "fock_ao_alpha", "fock_ao_beta", "fock_mo_alpha", "fock_mo_beta", "fock_spin"):
        n = SHAPES[name][0]
        if any(abs(arrays[name][i * n + j] - arrays[name][j * n + i]) > 1e-12 for i in range(n) for j in range(n)):
            raise ValueError("nonsymmetric one-electron matrix")
    spins = (0, 0, 1, 0, 0, 0, 1, 1, 1, 1)
    for spin, name in ((0, "alpha"), (1, "beta")):
        positions = [index for index, value in enumerate(spins) if value == spin]
        for i, p in enumerate(positions):
            for j, q in enumerate(positions):
                if arrays["fock_spin"][10 * p + q] != arrays[f"fock_mo_{name}"][5 * i + j]:
                    raise ValueError("spin Fock ordering mismatch")
    for p, q in itertools.product(range(10), repeat=2):
        if spins[p] != spins[q] and arrays["fock_spin"][10 * p + q] != 0:
            raise ValueError("spin Fock contains a forbidden spin coupling")
    def g(p: int, q: int, r: int, s: int) -> float:
        return arrays["antisymmetrized_eri"][((p * 10 + q) * 10 + r) * 10 + s]
    for p, q, r, s in itertools.product(range(10), repeat=4):
        value = g(p, q, r, s)
        if abs(value + g(q, p, r, s)) > 1e-12 or abs(value + g(p, q, s, r)) > 1e-12:
            raise ValueError("full spin ERI lost fermionic exchange symmetry")
        if not ((spins[p] == spins[r] and spins[q] == spins[s]) or (spins[p] == spins[s] and spins[q] == spins[r])) and value != 0:
            raise ValueError("full spin ERI contains a forbidden spin coupling")
    offset = 0
    for name, shape in BLOCK_SHAPES.items():
        values = arrays[f"amplitudes.{name}"]
        if name in ("t2aa", "t2bb"):
            expected = [0.0] * math.prod(shape)
            for i, j in itertools.combinations(range(shape[0]), 2):
                for a, b in itertools.combinations(range(shape[2]), 2):
                    value = arrays["packed_amplitudes"][offset]
                    offset += 1
                    for ii, jj, aa, bb, sign in ((i,j,a,b,1), (j,i,b,a,1), (j,i,a,b,-1), (i,j,b,a,-1)):
                        index = ((ii * shape[1] + jj) * shape[2] + aa) * shape[3] + bb
                        expected[index] = sign * value
        else:
            expected = arrays["packed_amplitudes"][offset:offset + math.prod(shape)]
            offset += math.prod(shape)
        if tuple(expected) != values:
            raise ValueError("full random amplitude blocks do not preserve packed representatives")
    if offset != 37:
        raise ValueError("packed random amplitude inventory mismatch")
    for name in BLOCK_SHAPES:
        for value, denominator, new, old in zip(
            arrays[f"residuals.{name}"], arrays[f"denominators.{name}"],
            arrays[f"updated_amplitudes.{name}"], arrays[f"amplitudes.{name}"], strict=True
        ):
            if abs(value - denominator * (new - old)) > 1e-13:
                raise ValueError("independent update/residual relation mismatch")
    channels = {name: arrays[f"channels.{name}"][0] for name in CHANNELS}
    if any(abs(channels[name]) >= 2e-13 for name in ("aaa", "bbb", "abb")):
        raise ValueError("impossible Li spin channel is not zero")
    total = arrays["triples_slow_total"][0]
    tolerance = 3e-13 + 3e-13 * abs(total)
    if abs(sum(channels.values()) - total) > tolerance or abs(arrays["triples_optimized_total"][0] - total) > tolerance:
        raise ValueError("independent complete triples reductions disagree")


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
        first["replica"] != "first" or second["replica"] != "second"
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
        "absolute_tolerance": REPLAY_ATOL, "max_absolute_difference": maximum,
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
    receipt = verify(args.verify, inputs) if args.verify else generate(inputs, args.replica)
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("complete Li receipt exceeds its strict byte cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    identity = args.replica or "validated"
    print(f"Validated complete Li receipt: {identity} bytes={len(raw)} sha256={audit.digest(raw)}")
    if args.emit:
        print(f"GQC_LI_JSON_BEGIN replica={identity} sha256={audit.digest(raw)} bytes={len(raw)}")
        sys.stdout.write(raw.decode())
        print("GQC_LI_JSON_END")


if __name__ == "__main__":
    main()
