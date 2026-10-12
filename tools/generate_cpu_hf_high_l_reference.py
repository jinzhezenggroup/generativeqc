"""Generate independent mixed-spdf PySCF receipts; never import GenerativeQC.

Run twice in separate interpreters, then use --verify FIRST SECOND to validate
the complete inventory and replay agreement. Normal numerical CI consumes the
reviewed frozen receipt instead of importing this optional regeneration tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_BYTES = 256 * 1024
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "tests/reference_data/cpu_hf_high_l/inputs.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"receipt/input exceeds {MAX_BYTES} bytes")
    result = json.loads(raw)
    canonical(result)  # Reject nonfinite constants accepted by the JSON decoder.
    if not isinstance(result, dict):
        raise TypeError("expected a JSON object")
    return result


def cases(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for state in inputs["states"]:
        for representation in inputs["representations"]:
            for index, displacement in enumerate(inputs["displacements_bohr"]):
                atoms = [
                    [element, list(position)] for element, position in inputs["atoms"]
                ]
                atoms[inputs["displacement_atom"]][1][inputs["displacement_axis"]] += (
                    displacement
                )
                result.append(
                    {
                        "id": f"{state['method']}-{representation}-displacement-{index}",
                        "atom": atoms,
                        "basis": inputs["basis"],
                        "unit": inputs["unit"],
                        "method": state["method"],
                        "charge": state["charge"],
                        "spin": state["spin"],
                        "cart": representation == "cartesian",
                        "displacement_bohr": displacement,
                        "scf": inputs["scf"],
                    }
                )
    if len(result) != 8 or len({case["id"] for case in result}) != 8:
        raise ValueError("expected exactly eight distinct fixed cases")
    return result


def array_record(values: Any, shape: list[int]) -> dict[str, Any]:
    def flatten(value: Any) -> list[float]:
        if isinstance(value, list):
            return [number for item in value for number in flatten(item)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected real float64 data")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite scientific output")
        return [number]

    flat = flatten(values)
    if len(flat) != math.prod(shape):
        raise ValueError("incorrect array size")
    return {
        "dtype": "float64",
        "shape": shape,
        "values": values,
        "sha256_le_f64": digest(struct.pack(f"<{len(flat)}d", *flat)),
    }


def distribution_identity(name: str, expected: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    if distribution.version != expected:
        raise ValueError(f"{name}: expected {expected}, found {distribution.version}")
    records = []
    total_bytes = 0
    for entry in sorted(distribution.files or (), key=str):
        relative = str(entry)
        if relative.endswith(".py") or ".so" in Path(relative).name:
            raw = Path(distribution.locate_file(entry)).read_bytes()
            records.append([relative, len(raw), digest(raw)])
            total_bytes += len(raw)
    if not records:
        raise ValueError(f"missing installed source/native files for {name}")
    record = distribution.read_text("RECORD")
    if record is None:
        raise ValueError(f"missing wheel RECORD for {name}")
    return {
        "version": distribution.version,
        "wheel_record_sha256": digest(record.encode()),
        "installed_source_native_sha256": digest(canonical(records)),
        "installed_source_native_files": len(records),
        "installed_source_native_bytes": total_bytes,
    }


def execution_identity() -> dict[str, str]:
    checkout = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", checkout) is None:
        raise ValueError("invalid checkout identity")
    result = {"checkout_sha": checkout}
    for name in (
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    ):
        value = os.environ.get(name)
        if value is not None:
            result[name.lower()] = value
    if "github_sha" in result and result["github_sha"] != checkout:
        raise ValueError("GitHub event SHA does not match the actual checkout")
    return result


def module_inventory(inputs: dict[str, Any]) -> dict[str, set[Path]]:
    """Fail closed on shadow modules before executing scientific imports."""
    result = {}
    for name, version in inputs["packages"].items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise ValueError(f"unexpected installed version: {name}")
        owned = {
            Path(distribution.locate_file(entry)).resolve()
            for entry in distribution.files or ()
        }
        spec = importlib.util.find_spec(name)
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).resolve() not in owned
        ):
            raise ValueError(
                f"scientific import is outside its pinned distribution: {name}"
            )
        result[name] = owned
    return result


def check_imported_modules(inventory: dict[str, set[Path]]) -> None:
    for name, module in tuple(sys.modules.items()):
        package = name.split(".", 1)[0]
        if package not in inventory or module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None or Path(filename).resolve() not in inventory[package]:
            raise ValueError(
                f"loaded scientific module is outside pinned files: {name}"
            )
        for directory in getattr(module, "__path__", ()):
            if Path(directory).resolve() != Path(filename).resolve().parent:
                raise ValueError(f"unowned scientific package search path: {name}")


def generate(inputs: dict[str, Any], input_path: Path, replica: str) -> dict[str, Any]:
    """Use an actual empty regular config file; /dev/null permits fallback."""
    if any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules):
        raise ValueError("generation requires a fresh interpreter without PySCF")
    if any(name.startswith("PYSCF_") for name in os.environ):
        raise ValueError("unset external PYSCF_* configuration before generation")
    inventory = module_inventory(inputs)
    with tempfile.TemporaryDirectory(prefix="gqc-pyscf-config-") as directory:
        config = Path(directory) / "empty.py"
        config.write_bytes(b"")
        os.environ["PYSCF_CONFIG_FILE"] = str(config)
        try:
            result = _generate(inputs, input_path, replica, config, inventory)
        finally:
            os.environ.pop("PYSCF_CONFIG_FILE", None)
    return result


def _generate(
    inputs: dict[str, Any],
    input_path: Path,
    replica: str,
    config: Path,
    inventory: dict[str, set[Path]],
) -> dict[str, Any]:
    # This process has no import of the implementation under test.
    import numpy as np
    import pyscf
    from threadpoolctl import threadpool_info

    selected_config = getattr(pyscf.__config__, "conf_file", None)
    if selected_config is None or Path(selected_config).resolve() != config.resolve():
        raise ValueError("PySCF did not select the explicit empty regular config")
    if config.read_bytes() != b"":
        raise ValueError("explicit PySCF configuration is no longer empty")
    check_imported_modules(inventory)
    packages = {
        name: distribution_identity(name, version)
        for name, version in inputs["packages"].items()
    }
    if pyscf.__version__ != inputs["packages"]["pyscf"]:
        raise ValueError("imported PySCF does not match the pinned distribution")
    records = []
    with pyscf.lib.with_omp_threads(1):
        for case in cases(inputs):
            molecule = pyscf.gto.M(
                atom=case["atom"],
                unit=case["unit"],
                basis=case["basis"],
                charge=case["charge"],
                spin=case["spin"],
                cart=case["cart"],
                verbose=0,
            )
            reference = (
                pyscf.scf.RHF(molecule)
                if case["method"] == "rhf"
                else pyscf.scf.UHF(molecule)
            )
            for name, value in case["scf"].items():
                setattr(reference, name, value)
            reference.kernel()
            if not reference.converged:
                raise ValueError(f"SCF did not converge: {case['id']}")
            forces = np.asarray(-reference.nuc_grad_method().kernel(), dtype=np.float64)
            if forces.shape != (2, 3):
                raise ValueError("unexpected analytic-force shape")
            records.append(
                {
                    "input": case,
                    "input_sha256": digest(canonical(case)),
                    "nao": molecule.nao_nr(),
                    "ao_labels": molecule.ao_labels(),
                    "nelec": list(molecule.nelec),
                    "converged": bool(reference.converged),
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
                    "energy": array_record(float(reference.e_tot), []),
                    "forces": array_record(forces.tolist(), [2, 3]),
                }
            )
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
                | {"library": path.name, "sha256": digest(path.read_bytes())}
            )
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("a native thread pool is not single-threaded")
    check_imported_modules(inventory)
    result = {
        "schema": "generativeqc.cpu-hf-high-l.receipt.v1",
        "replica": replica,
        "process_id": os.getpid(),
        "provenance": {
            "configuration": {
                "policy": "explicit-empty-regular-file",
                "sha256": digest(b""),
            },
            "inputs_sha256": digest(input_path.read_bytes()),
            "generator_sha256": digest(Path(__file__).read_bytes()),
            "execution": execution_identity(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
            "threadpools": pools,
        },
        "inputs": inputs,
        "records": records,
    }
    validate(result, inputs, input_path)
    return result


def validate(receipt: dict[str, Any], inputs: dict[str, Any], input_path: Path) -> None:
    if receipt.get("schema") != "generativeqc.cpu-hf-high-l.receipt.v1":
        raise ValueError("incorrect receipt schema")
    if receipt.get("inputs") != inputs:
        raise ValueError("receipt input manifest mismatch")
    if receipt["replica"] not in ("first", "second"):
        raise ValueError("invalid replica identity")
    if type(receipt["process_id"]) is not int or receipt["process_id"] <= 0:
        raise ValueError("invalid interpreter process identity")
    provenance = receipt["provenance"]
    if provenance["configuration"] != {
        "policy": "explicit-empty-regular-file",
        "sha256": digest(b""),
    }:
        raise ValueError("scientific configuration identity mismatch")
    for name, value in (
        ("execution", execution_identity()),
        ("python", platform.python_version()),
        ("platform", platform.platform()),
        ("machine", platform.machine()),
    ):
        if provenance[name] != value:
            raise ValueError(f"execution/environment identity mismatch: {name}")
    if provenance["inputs_sha256"] != digest(input_path.read_bytes()):
        raise ValueError("input byte hash mismatch")
    if provenance["generator_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("generator byte hash mismatch")
    for name, version in inputs["packages"].items():
        package = provenance["packages"][name]
        if package["version"] != version:
            raise ValueError("pinned package version mismatch")
        for key in ("wheel_record_sha256", "installed_source_native_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", package[key]) is None:
                raise ValueError("invalid package byte identity")
        for key in ("installed_source_native_files", "installed_source_native_bytes"):
            if type(package[key]) is not int or package[key] < 1:
                raise ValueError("empty or invalid package byte inventory")
    if not provenance["threadpools"]:
        raise ValueError("missing native thread-pool identity")
    for pool in provenance["threadpools"]:
        if type(pool["num_threads"]) is not int or pool["num_threads"] != 1:
            raise ValueError("native thread-pool limit mismatch")
        if set(pool) != {
            "user_api",
            "internal_api",
            "version",
            "num_threads",
            "architecture",
            "library",
            "sha256",
        }:
            raise ValueError("incomplete native thread-pool identity")
        for key in ("user_api", "internal_api", "library"):
            if not isinstance(pool[key], str) or not pool[key]:
                raise ValueError("invalid native thread-pool identity")
        if Path(pool["library"]).name != pool["library"]:
            raise ValueError("expected a native library basename")
        for key in ("version", "architecture"):
            if pool[key] is not None and not isinstance(pool[key], str):
                raise ValueError("invalid native thread-pool metadata")
        if re.fullmatch(r"[0-9a-f]{64}", pool["sha256"]) is None:
            raise ValueError("invalid native library byte identity")
    expected = cases(inputs)
    records = receipt["records"]
    if len(records) != len(expected):
        raise ValueError("missing or extra reference cases")
    for case, record in zip(expected, records, strict=True):
        if record["input"] != case or record["input_sha256"] != digest(canonical(case)):
            raise ValueError("case input/order mismatch")
        nao = 21 if case["cart"] else 17
        nelec = [1, 1] if case["method"] == "rhf" else [2, 1]
        labels = record["ao_labels"]
        if (
            record["nao"] != nao
            or not isinstance(labels, list)
            or len(labels) != nao
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError("incorrect AO count/labels")
        if (
            record["nelec"] != nelec
            or any(type(count) is not int for count in record["nelec"])
            or record["converged"] is not True
        ):
            raise ValueError("electron count or convergence mismatch")
        resolved = record["resolved_scf"]
        if set(resolved) != {
            "init_guess",
            "conv_tol",
            "conv_tol_grad",
            "max_cycle",
            "diis_space",
            "direct_scf",
            "direct_scf_tol",
            "level_shift",
            "damp",
        }:
            raise ValueError("incomplete resolved SCF controls")
        for name in ("max_cycle", "diis_space"):
            if type(resolved[name]) is not int or resolved[name] < 1:
                raise ValueError("invalid resolved SCF iteration controls")
        if type(resolved["direct_scf"]) is not bool:
            raise ValueError("invalid resolved SCF direct-build control")
        for name in ("direct_scf_tol", "level_shift", "damp"):
            array_record(resolved[name], [])
        for name, value in case["scf"].items():
            if resolved[name] != value:
                raise ValueError("SCF settings mismatch")
        for name, shape in (("energy", []), ("forces", [2, 3])):
            array = record[name]
            if name == "energy" and isinstance(array["values"], list):
                raise ValueError("expected scalar energy")
            if name == "forces" and (
                not isinstance(array["values"], list)
                or len(array["values"]) != 2
                or any(
                    not isinstance(row, list) or len(row) != 3
                    for row in array["values"]
                )
            ):
                raise ValueError("incorrect force dimensions")
            if array != array_record(array["values"], shape):
                raise ValueError("scientific array metadata/hash mismatch")


def verify(
    paths: list[Path], inputs: dict[str, Any], input_path: Path
) -> dict[str, Any]:
    first, second = [read_json(path) for path in paths]
    for receipt in (first, second):
        validate(receipt, inputs, input_path)
    if (
        first["replica"] == second["replica"]
        or first["process_id"] == second["process_id"]
    ):
        raise ValueError("replay must come from two separate interpreter processes")
    if first["provenance"] != second["provenance"]:
        raise ValueError("replay provenance mismatch")
    max_energy = 0.0
    max_force = 0.0
    for left, right in zip(first["records"], second["records"], strict=True):
        for field in (
            "input",
            "input_sha256",
            "nao",
            "ao_labels",
            "nelec",
            "converged",
            "resolved_scf",
        ):
            if left[field] != right[field]:
                raise ValueError(f"replay metadata mismatch: {field}")
        max_energy = max(
            max_energy, abs(left["energy"]["values"] - right["energy"]["values"])
        )
        for row_a, row_b in zip(
            left["forces"]["values"], right["forces"]["values"], strict=True
        ):
            max_force = max(
                max_force, *(abs(a - b) for a, b in zip(row_a, row_b, strict=True))
            )
    if max_energy > 1e-12 or max_force > 1e-11:
        raise ValueError(
            f"independent replay disagrees: energy={max_energy}, force={max_force}"
        )
    first["replay"] = {
        "schema": "generativeqc.independent-process-replay.v1",
        "receipt_sha256": [digest(path.read_bytes()) for path in paths],
        "replicas": [first["replica"], second["replica"]],
        "process_ids": [first["process_id"], second["process_id"]],
        "energy_atol": 1e-12,
        "force_atol": 1e-11,
        "max_energy_difference": max_energy,
        "max_force_difference": max_force,
    }
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica", choices=("first", "second"))
    parser.add_argument("--verify", type=Path, nargs=2)
    parser.add_argument(
        "--emit", action="store_true", help="emit the bounded validated receipt"
    )
    args = parser.parse_args()
    inputs = read_json(args.inputs)
    if args.verify:
        result = verify(args.verify, inputs, args.inputs)
    elif args.replica:
        result = generate(inputs, args.inputs, args.replica)
    else:
        parser.error("provide --replica or --verify")
    raw = (
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("receipt exceeds the strict byte cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(f"Validated eight cases; bytes={len(raw)} sha256={digest(raw)}")
    if args.emit:
        print(f"GQC_ORACLE_JSON_BEGIN sha256={digest(raw)} bytes={len(raw)}")
        sys.stdout.write(raw.decode())
        print("GQC_ORACLE_JSON_END")


if __name__ == "__main__":
    main()
