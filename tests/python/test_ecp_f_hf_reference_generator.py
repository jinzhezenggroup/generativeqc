"""Pre-acquisition contract checks; synthetic arrays here are never reference data."""

from __future__ import annotations

import base64
import builtins
import copy
import importlib.util
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _generator() -> Any:
    tools = ROOT / "tools"
    # Load only the optional independent tools, never the GQC implementation.
    for name in (
        "generate_cpu_hf_high_l_reference",
        "generate_ecp_raw_matrix_reference",
        "generate_ecp_f_hf_reference",
    ):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, tools / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules["generate_ecp_f_hf_reference"]


def test_input_contract_and_full_inventory_do_not_import_scientific_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"pyscf", "generativeqc", "generativeqc_compiler"}:
            raise AssertionError(
                "live scientific engine imported during contract check"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    generator.validate_inputs(inputs)
    inventory = generator.profiles(inputs)
    assert list(inventory) == [
        "rhf-cartesian",
        "rhf-spherical",
        "uhf-cartesian",
        "uhf-spherical",
    ]
    for profile in inventory.values():
        shapes = generator.shapes(profile)
        n = 19 if profile["cart"] else 16
        assert len(shapes) == 16
        assert shapes["gradient"] == shapes["forces"] == (2, 3)
        assert shapes["energy"] == shapes["nuclear_energy"] == ()
        assert shapes["overlap"] == shapes["hcore"] == (n, n)
        assert (
            shapes["density"]
            == shapes["fock"]
            == shapes["coefficients"]
            == ((2, n, n) if profile["spin"] else (n, n))
        )
        assert (
            shapes["orbital_energies"]
            == shapes["occupations"]
            == ((2, n) if profile["spin"] else (n,))
        )
        assert shapes["orbital_gradient"] == (n - 1,)
        assert not any(len(shape) == 4 for shape in shapes.values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "Angstrom"),
        ("extra_f_shell", [3, [0.55, 1.0]]),
        ("representations", ["cartesian"]),
        ("python", "3.12.14"),
        ("precision", "float32"),
        ("source_repository", "other/repo"),
        ("source_commit", "0" * 40),
    ],
)
def test_scientific_input_drift_is_rejected(field: str, value: Any) -> None:
    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    inputs[field] = value
    with pytest.raises(ValueError):
        generator.validate_inputs(inputs)


@pytest.mark.parametrize(
    "mutation",
    ["guess", "charge", "basis", "ecp", "gate", "pins", "scope", "extra", "upstream"],
)
def test_changed_contract_is_not_silently_admitted(mutation: str) -> None:
    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    if mutation == "guess":
        inputs["scf"]["initial_guess"] = "1e"
    elif mutation == "charge":
        inputs["states"][1]["charge"] = 0
    elif mutation == "basis":
        inputs["resolved_basis"]["Na"][-1][1][1] += 0.1
    elif mutation == "ecp":
        inputs["resolved_ecp"]["Na"][0] = 8
    elif mutation == "gate":
        inputs["preserved_gates"]["force_atol"] = 2e-5
    elif mutation == "pins":
        inputs["packages"]["pyscf"] = "2.13.1"
    elif mutation == "scope":
        inputs["acquisition"]["repository"] = "jinzhezenggroup/generativeqc"
    elif mutation == "upstream":
        inputs["upstream_scf_sources"][0]["sha256"] = "0" * 64
    else:
        inputs["f_on_h"] = True
    with pytest.raises(ValueError):
        generator.validate_inputs(inputs)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 19, 19)])
def test_lossless_complete_fp64_transport(shape: tuple[int, ...]) -> None:
    generator = _generator()
    # Codec-only synthetic sequence, never a physical E/F or SCF reference.
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    envelope = generator.encode_array(values, shape)
    assert envelope["bytes"] == values.size * 8
    assert generator.decode_array(envelope, shape) == tuple(values.ravel())


@pytest.mark.parametrize(
    "mutation",
    ["dtype", "shape", "bytes", "hash", "base64", "nonfinite", "missing", "extra"],
)
def test_malformed_array_is_rejected(mutation: str) -> None:
    generator = _generator()
    record = generator.encode_array(
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), (2, 3)
    )
    if mutation == "dtype":
        record["dtype"] = "float32"
    elif mutation == "shape":
        record["shape"] = [6]
    elif mutation == "bytes":
        record["bytes"] = 8
    elif mutation == "hash":
        record["sha256_le_f64"] = "0" * 64
    elif mutation == "base64":
        record["data"] += "\n"
    elif mutation == "nonfinite":
        raw = struct.pack("<6d", float("nan"), 1, 2, 3, 4, 5)
        record["data"] = base64.b64encode(raw).decode()
        record["sha256_le_f64"] = generator.audit.digest(raw)
    elif mutation == "missing":
        del record["order"]
    else:
        record["selected_entries"] = [1, 2]
    with pytest.raises((ValueError, TypeError)):
        generator.decode_array(record, (2, 3))


@pytest.mark.parametrize(
    "value", [np.ones((2, 3), dtype=np.float32), np.ones((6,)), np.full((2, 3), np.inf)]
)
def test_wrong_or_nonfinite_scientific_output_is_rejected(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _generator().encode_array(value, (2, 3))


@pytest.mark.parametrize("raw", [b'{"a": 1, "a": 2}', b'{"a": NaN}', b"[]"])
def test_invalid_json_is_rejected(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError)):
        _generator().read_json(path)


def test_reference_byte_cap(tmp_path: Path) -> None:
    generator = _generator()
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (generator.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        generator.read_json(path)


def test_replay_requires_eight_distinct_files(tmp_path: Path) -> None:
    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    for paths in ([tmp_path / "first.json"] * 8, []):
        with pytest.raises(ValueError, match="eight distinct"):
            generator.verify(paths, inputs)


def test_canonical_route_and_nonpinned_python_are_rejected_before_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    # Avoid pinning current production source bytes in ordinary CI. Source
    # identity is checked only by the opt-in acquisition preflight itself.
    root = copy.deepcopy(inputs)
    for entry in root["sources"]:
        entry["sha256"] = generator.audit.digest(
            (generator.ROOT / entry["path"]).read_bytes()
        )
    monkeypatch.setattr(generator.platform, "python_version", lambda: "3.12.14")
    with pytest.raises(ValueError, match="interpreter"):
        generator.validate_checkout(root, runtime=True)
    monkeypatch.setattr(generator.platform, "python_version", lambda: "3.12.12")
    for name in generator.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/" + root["acquisition"]["branch"])
    monkeypatch.setattr(
        generator.audit,
        "execution_identity",
        lambda: {
            "checkout_sha": "1" * 40,
            "github_sha": "1" * 40,
            "github_repository": "jinzhezenggroup/generativeqc",
            "github_run_id": "1",
            "github_run_attempt": "1",
        },
    )
    with pytest.raises(ValueError, match="CPU-only fork branch"):
        generator.validate_checkout(root, runtime=True)


@pytest.fixture
def controlled_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """In-memory/dummy-file evidence; never an acquired scientific receipt."""
    from types import SimpleNamespace

    generator = _generator()
    inputs = generator.read_json(generator.INPUTS)
    library = tmp_path / "unit_test_openblas.so"
    library.write_bytes(b"unit-test provider bytes only; never loaded or a fixture\n")
    observed_pool = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "unit-test",
        "num_threads": 1,
        "architecture": "unit-test-cpu",
        "filepath": str(library),
    }
    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(threadpool_info=lambda: [observed_pool]),
    )
    identities = {
        name: {
            "version": version,
            "wheel_record_sha256": generator.audit.digest((name + " wheel").encode()),
            "installed_source_native_sha256": generator.audit.digest(
                (name + " files").encode()
            ),
            "installed_source_native_files": 20 + index,
            "installed_source_native_bytes": 1000 + index,
        }
        for index, (name, version) in enumerate(inputs["packages"].items())
    }

    def identity(name: str, expected: str) -> dict[str, Any]:
        assert expected == inputs["packages"][name]
        return copy.deepcopy(identities[name])

    monkeypatch.setattr(generator.audit, "distribution_identity", identity)
    execution = {
        "checkout_sha": "1" * 40,
        "github_sha": "1" * 40,
        "github_repository": inputs["acquisition"]["repository"],
        "github_run_id": "123",
        "github_run_attempt": "1",
    }
    monkeypatch.setattr(generator.audit, "execution_identity", lambda: execution)
    monkeypatch.setattr(
        generator.ecp_audit,
        "installed_integral_inputs",
        lambda _: {"controlled": "integrals"},
    )
    monkeypatch.setattr(
        generator, "installed_scf_sources", lambda _: {"controlled": "scf"}
    )
    provenance = {
        "inputs_sha256": generator.audit.digest(generator.INPUTS.read_bytes()),
        "generator_sha256": generator.audit.digest(
            Path(generator.__file__).read_bytes()
        ),
        "source_sha256": {
            entry["path"]: entry["sha256"] for entry in inputs["sources"]
        },
        "configuration": {
            "policy": "explicit-empty-regular-file",
            "sha256": generator.audit.digest(b""),
        },
        "python": inputs["python"],
        "platform": generator.platform.platform(),
        "machine": generator.platform.machine(),
        "device": "CPU",
        "cuda_visible_devices": "",
        "execution": execution,
        "packages": copy.deepcopy(identities),
        "installed_integral_inputs": {"controlled": "integrals"},
        "installed_scf_sources": {"controlled": "scf"},
        "threadpools": generator.cpu_provider_records(),
    }
    return generator, inputs, provenance, observed_pool


def test_exact_observed_provenance_is_accepted(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    generator.validate_provenance(provenance, inputs)
    assert set(provenance["packages"]) == set(inputs["packages"])
    assert provenance["threadpools"] == generator.cpu_provider_records()


@pytest.mark.parametrize(
    "field,value",
    [
        ("wheel_record_sha256", "0" * 64),
        ("installed_source_native_sha256", "0" * 64),
        ("installed_source_native_files", 1),
        ("installed_source_native_bytes", 1),
    ],
)
def test_matching_versions_do_not_authorize_invented_package_identity(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    for package in provenance["packages"].values():
        package[field] = value
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("library", "invented_provider.so"),
        ("internal_api", "cuda"),
        ("user_api", "cuda"),
        ("sha256", "0" * 64),
        ("num_threads", 2),
        ("num_threads", True),
        ("version", "invented-version"),
        ("architecture", "invented-architecture"),
    ],
)
def test_provider_claim_must_match_actual_loaded_metadata_and_bytes(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    provenance["threadpools"][0][field] = value
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)


def test_original_review_mutation_is_rejected(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    for package in provenance["packages"].values():
        package.update(
            wheel_record_sha256="0" * 64,
            installed_source_native_sha256="0" * 64,
            installed_source_native_files=1,
            installed_source_native_bytes=1,
        )
    provenance["threadpools"][0].update(
        library="invented_provider.so", internal_api="cuda", sha256="0" * 64
    )
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)


def test_provider_native_byte_change_invalidates_existing_receipt(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    generator, inputs, provenance, observed = controlled_provenance
    Path(observed["filepath"]).write_bytes(b"different controlled native bytes\n")
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("internal_api", "cuda"),
        ("user_api", "cuda"),
        ("num_threads", 2),
        ("num_threads", True),
    ],
)
def test_actual_loaded_non_cpu_or_multithreaded_provider_is_rejected(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    generator, _, _, observed = controlled_provenance
    observed[field] = value
    with pytest.raises(ValueError, match="one-thread CPU"):
        generator.cpu_provider_records()


def test_setuptools_is_pinned_and_cannot_be_omitted_from_provenance(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    assert inputs["packages"]["setuptools"] == "83.0.0"
    del provenance["packages"]["setuptools"]
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "platform", "machine", "extra"]
)
def test_provenance_inventory_and_runtime_identity_are_exact(
    controlled_provenance: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
    mutation: str,
) -> None:
    generator, inputs, provenance, _ = controlled_provenance
    if mutation == "missing":
        provenance["threadpools"] = []
    elif mutation == "duplicate":
        provenance["threadpools"] *= 2
    elif mutation in ("platform", "machine"):
        provenance[mutation] = "invented-runtime"
    else:
        provenance["undeclared"] = True
    with pytest.raises(ValueError, match="provenance"):
        generator.validate_provenance(provenance, inputs)
