"""CPU-only failure injection for the Fast anchor evidence runners."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import numpy as np
import pytest

RUNNERS = Path(
    os.environ.get(
        "FAST_ANCHOR_RUNNER_ROOT",
        Path(__file__).resolve().parents[2] / "benchmarks/fast_anchor_default",
    )
)
ARMS = ((12, "fast"), (96, "fast"), (12, "fp64"))
COUNTERS = (
    "mixed_stage_fock_builds",
    "strict_stage_fock_builds",
    "post_scf_fock_builds",
    "execution_retries",
    "fallback_count",
    "final_residual_audits",
)


def load_runner(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_record(atoms: int, policy: str, delta: str) -> dict[str, Any]:
    return {
        "atoms": atoms,
        "policy": policy,
        "delta": delta,
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]] * atoms,
        "finite_forces": True,
        "succeeded": True,
        "converged": True,
        "warm_start_used": False,
        "cold_endpoint_seconds": 2.0,
        "candidate_controls": {"density_tolerance": 1e-4},
        "atoms_bohr": [["H", [0.0, 0.0, float(i)]] for i in range(atoms)],
        "basis_identity": "fixture-basis",
        "precision": {
            "complete": True,
            "operator_inventory_complete": True,
            "operators": [{"name": "fixture"}],
            **dict.fromkeys(COUNTERS, 1),
        },
        "ks_diagnostic": {
            "nonlocal_work": {"count": 1},
            "coulomb_exchange_work": {"count": 1},
            "nonlocal_history_scopes": ["fixture"],
        },
        "independent_gate": {"gate": True},
    }


def write_sample(root: Path, record: dict[str, Any], name: str | None = None) -> Path:
    directory = root / (
        name or f"{record['policy']}{record['atoms']}-{record['delta']}"
    )
    directory.mkdir(parents=True)
    density = directory / "result.density.npy"
    np.save(density, np.array([1.0, 0.0, 0.0, 1.0]))
    record = dict(record)
    record["final_density"] = {
        "path": density.name,
        "count": 4,
        "sha256": hashlib.sha256(density.read_bytes()).hexdigest(),
    }
    result = directory / "result.json"
    result.write_text(json.dumps(record))
    for filename in ("library-before.sha256", "library-after.sha256"):
        (directory / filename).write_text("a" * 64 + " library.so\n")
    (directory / "exit-code.txt").write_text("0\n")
    (directory / "source-authentication.json").write_text(
        json.dumps({"native_identity": "b" * 64, "source": "fixture"})
    )
    return result


@pytest.fixture
def suite(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "receipts"
    build = tmp_path / "build"
    build.mkdir()
    (build / "authentication.json").write_text(
        json.dumps({"source_identity": "b" * 64, "library_sha256": "a" * 64})
    )
    (build / "rebuilt-inputs.json").write_text("{}")
    for name in ("cache-version.txt", "cache-before.txt", "cache-after.txt"):
        (build / name).write_text("fixture")
    for atoms, policy in ARMS:
        for delta in ("0", "auto"):
            write_sample(root, sample_record(atoms, policy, delta))
    return root, build, tmp_path / "summary.json"


def analyze(monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]) -> None:
    root, build, output = suite
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze", str(root), "--build-receipts", str(build), "--output", str(output)],
    )
    load_runner("analyze").main()


def rejected(output: Path) -> dict[str, Any]:
    diagnostics = list(output.parent.glob(f"{output.stem}.rejected-*.json"))
    assert diagnostics, "A rejected attempt must retain a diagnostic receipt"
    return json.loads(diagnostics[-1].read_text())


def test_complete_suite_qualifies(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    analyze(monkeypatch, suite)
    summary = json.loads(suite[2].read_text())
    assert len(summary["samples"]) == 6
    assert len(summary["pairs"]) == 3
    assert all(sample["gate"] for sample in summary["samples"])


@pytest.mark.parametrize(
    "target", ["arm_source", "arm_library", "build_source", "build_library"]
)
@pytest.mark.parametrize("defect", ["mismatch", "missing", "malformed"])
def test_reducer_requires_source_and_library_bindings(
    monkeypatch: pytest.MonkeyPatch,
    suite: tuple[Path, Path, Path],
    target: str,
    defect: str,
) -> None:
    root, build, output = suite
    value = "c" * 64 if defect == "mismatch" else "invalid-digest"
    if target == "arm_library":
        for name in ("library-before.sha256", "library-after.sha256"):
            (root / "fast12-auto" / name).write_text(
                "" if defect == "missing" else value + " library.so\n"
            )
    else:
        if target == "arm_source":
            path = root / "fast12-auto/source-authentication.json"
            field = "native_identity"
        else:
            path = build / "authentication.json"
            field = "source_identity" if target == "build_source" else "library_sha256"
        metadata = json.loads(path.read_text())
        if defect == "missing":
            metadata.pop(field)
        else:
            metadata[field] = value
        path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert not output.exists()
    diagnostic = rejected(output)
    assert len(diagnostic["samples"]) == 6
    assert len(diagnostic["raw_records"]) == 6


@pytest.mark.parametrize("field", ["source_identity", "library_sha256"])
def test_matching_malformed_bindings_cannot_qualify(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path], field: str
) -> None:
    root, build, output = suite
    path = build / "authentication.json"
    metadata = json.loads(path.read_text())
    metadata[field] = "matching-but-not-a-digest"
    path.write_text(json.dumps(metadata))
    for directory in root.iterdir():
        if field == "source_identity":
            (directory / "source-authentication.json").write_text(
                json.dumps({"native_identity": metadata[field], "source": "fixture"})
            )
        else:
            for name in ("library-before.sha256", "library-after.sha256"):
                (directory / name).write_text(metadata[field] + " library.so\n")
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert not output.exists()
    assert rejected(output)["qualification_errors"]


@pytest.mark.parametrize("mode", ["one_pair", "missing", "duplicate", "unexpected"])
def test_incomplete_or_ambiguous_suite_is_rejected(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path], mode: str
) -> None:
    root, _, output = suite
    if mode == "one_pair":
        for path in root.glob("*/result.json"):
            if not path.parent.name.startswith("fast12-"):
                path.unlink()
    elif mode == "missing":
        (root / "fast96-auto/result.json").unlink()
    else:
        write_sample(
            root,
            sample_record(12, "fast", "0" if mode == "duplicate" else "1"),
            "extra",
        )
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert not output.exists()
    assert rejected(output)["samples"]


def test_rejected_reduction_preserves_previous_summary(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    root, _, output = suite
    output.write_text("previous accepted summary")
    (root / "fast12-0/exit-code.txt").write_text("1\n")
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert output.read_text() == "previous accepted summary"
    assert any(not sample["gate"] for sample in rejected(output)["samples"])


@pytest.mark.parametrize("kind", ["file", "dangling"])
def test_accepted_reduction_never_replaces_output(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path], kind: str
) -> None:
    output = suite[2]
    target = output.with_name("missing-target")
    if kind == "file":
        output.write_text("previous accepted summary")
    else:
        output.symlink_to(target)
    with pytest.raises((SystemExit, FileExistsError)):
        analyze(monkeypatch, suite)
    if kind == "file":
        assert output.read_text() == "previous accepted summary"
    else:
        assert output.is_symlink() and not target.exists()


def test_early_failure_is_reduced_without_masking_original_error(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    result = suite[0] / "fast12-0/result.json"
    result.write_text(
        json.dumps(
            {
                "atoms": 12,
                "policy": "fast",
                "delta": "0",
                "error": "original early failure",
            }
        )
    )
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    diagnostic = rejected(suite[2])
    failed = next(
        item for item in diagnostic["samples"] if item["receipt"] == "fast12-0"
    )
    assert not failed["gate"]
    assert failed["error"] == "original early failure"


def test_attempt_without_endpoint_receipt_is_not_discarded(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    attempt = suite[0] / "timed-out-attempt"
    attempt.mkdir()
    (attempt / "exit-code.txt").write_text("124\n")
    (attempt / "endpoint.log").write_text("retained timeout diagnostics")
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    failed = next(
        item
        for item in rejected(suite[2])["samples"]
        if item["receipt"] == attempt.name
    )
    assert not failed["gate"] and failed["exit_code"] == 124
    assert (attempt / "endpoint.log").read_text() == "retained timeout diagnostics"


def test_repeated_rejections_are_retained_separately(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    (suite[0] / "fast12-0/exit-code.txt").write_text("1\n")
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    first = next(suite[2].parent.glob("summary.rejected-*.json"))
    original = first.read_bytes()
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert len(list(suite[2].parent.glob("summary.rejected-*.json"))) == 2
    assert first.read_bytes() == original


def test_reducer_preserves_output_created_during_comparison(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path]
) -> None:
    original_load = np.load

    def concurrent_publication(*args: Any, **kwargs: Any) -> Any:
        if not suite[2].exists():
            suite[2].write_text("concurrent accepted summary")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", concurrent_publication)
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert suite[2].read_text() == "concurrent accepted summary"
    assert rejected(suite[2])["qualification_errors"]


@pytest.mark.parametrize(
    "field", ["energy", "forces", "cold_endpoint_seconds", "density"]
)
def test_nonfinite_evidence_is_retained_and_rejected(
    monkeypatch: pytest.MonkeyPatch, suite: tuple[Path, Path, Path], field: str
) -> None:
    result = suite[0] / "fast12-0/result.json"
    record = json.loads(result.read_text())
    if field == "density":
        np.save(result.with_suffix(".density.npy"), np.array([float("nan")]))
    else:
        record[field] = (
            [[float("nan"), 0.0, 0.0]] if field == "forces" else float("nan")
        )
        result.write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        analyze(monkeypatch, suite)
    assert not suite[2].exists()
    assert rejected(suite[2])["samples"]


@pytest.fixture
def endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, Path, dict[str, Any]]:
    state: dict[str, Any] = {
        "execute_error": False,
        "cleanup_error": False,
        "entered": False,
    }
    record = sample_record(12, "fast", "auto")

    class Owner:
        _owner = object()
        ks_transport_diagnostics: ClassVar[dict[str, Any]] = {}
        resource_diagnostics: ClassVar[dict[str, Any]] = {}

        def __enter__(self) -> Self:
            state["entered"] = True
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, **kwargs: Any) -> Any:
            if state["execute_error"]:
                raise RuntimeError("original execute failure")
            item = SimpleNamespace(**record)
            item.forces = np.asarray(item.forces)
            return SimpleNamespace(items=[item], precision_validation={})

    class Calculator:
        _candidate = SimpleNamespace()

        def __init__(self, **kwargs: Any) -> None:
            return None

        def prepare_batch(self, *args: Any, **kwargs: Any) -> Owner:
            return Owner()

        def clear_cache(self) -> None:
            if state["cleanup_error"]:
                raise RuntimeError("cleanup failure")

    def export(_owner: Any, path: Path) -> dict[str, Any]:
        np.save(path, np.array([1.0]))
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": 1,
        }

    fakes = {
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(deviceSynchronize=lambda: None)
            )
        ),
        "generativeqc": SimpleNamespace(
            GridSpec=lambda *a: None,
            KsOptions=lambda **k: None,
            PrecisionCalculator=Calculator,
        ),
        "benchmarks.compare_gpu4pyscf_batch": SimpleNamespace(
            load_comparison_basis=lambda *a, **k: (
                SimpleNamespace(identity="fixture-basis"),
                None,
            ),
            native_build_metadata=lambda candidate: {},
        ),
        "benchmarks.pbe0_initial_guesses": SimpleNamespace(export_density=export),
        "benchmarks.readme_hf_scaling": SimpleNamespace(
            scaling_cases=lambda: {
                "water-12": SimpleNamespace(atoms=record["atoms_bohr"])
            }
        ),
    }
    for name, value in fakes.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fixture-device")
    monkeypatch.setenv("EXPERIMENT_ROOT", str(tmp_path))
    monkeypatch.delenv("GENERATIVEQC_KS_FAST_ANCHOR_DELTA", raising=False)
    (tmp_path / "def2-svp.json").write_text("{}")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "endpoint",
            "--atoms",
            "12",
            "--policy",
            "fast",
            "--delta",
            "auto",
            "--output",
            str(output),
        ],
    )
    module = load_runner("endpoint")
    monkeypatch.setattr(module, "reference_gate", lambda *a: {"gate": True})
    return module, output, state


def test_execution_and_cleanup_failures_both_survive(
    endpoint: tuple[Any, Path, dict[str, Any]],
) -> None:
    module, output, state = endpoint
    state.update(execute_error=True, cleanup_error=True)
    with pytest.raises(SystemExit):
        module.main()
    record = json.loads(output.read_text())
    assert "original execute failure" in record["error"]
    assert "cleanup failure" in record["cleanup_error"]


def test_final_cleanup_failure_still_publishes_a_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, endpoint: tuple[Any, Path, dict[str, Any]]
) -> None:
    module, output, state = endpoint

    def after_measured_endpoint(candidate: Any) -> dict[str, Any]:
        state["cleanup_error"] = True
        return {}

    monkeypatch.setattr(module, "native_build_metadata", after_measured_endpoint)
    with pytest.raises(SystemExit):
        module.main()
    record = json.loads(output.read_text())
    assert record["cold_endpoint_seconds"] > 0
    assert "cleanup failure" in record["cleanup_error"]


@pytest.mark.parametrize("filename", ["result.json", "result.density.npy"])
def test_endpoint_preserves_output_created_during_execution(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[Any, Path, dict[str, Any]],
    filename: str,
) -> None:
    module, output, _ = endpoint
    occupied = output.with_name(filename)
    original_export = module.export_density

    def concurrent_publication(owner: Any, path: Path) -> Any:
        result = original_export(owner, path)
        occupied.write_bytes(b"concurrent accepted evidence")
        return result

    monkeypatch.setattr(module, "export_density", concurrent_publication)
    with pytest.raises(SystemExit):
        module.main()
    assert occupied.read_bytes() == b"concurrent accepted evidence"
    failures = [path for path in output.parent.glob("result*.json") if path != occupied]
    assert any(json.loads(path.read_text()).get("error") for path in failures)


def test_endpoint_nonfinite_diagnostics_do_not_erase_receipt(
    monkeypatch: pytest.MonkeyPatch, endpoint: tuple[Any, Path, dict[str, Any]]
) -> None:
    module, output, _ = endpoint
    monkeypatch.setattr(
        module, "native_build_metadata", lambda candidate: {"invalid": float("inf")}
    )
    with pytest.raises(SystemExit):
        module.main()
    record = json.loads(output.read_text())
    assert record["nonfinite_fields"]
    assert record["native_build"]["invalid"] == "inf"


@pytest.mark.parametrize("filename", ["result.json", "result.density.npy"])
@pytest.mark.parametrize("kind", ["file", "dangling"])
def test_endpoint_preserves_occupied_outputs(
    endpoint: tuple[Any, Path, dict[str, Any]], filename: str, kind: str
) -> None:
    module, output, state = endpoint
    occupied = output.with_name(filename)
    target = output.with_name("missing-target")
    if kind == "file":
        occupied.write_bytes(b"accepted evidence")
    else:
        occupied.symlink_to(target)
    with pytest.raises((SystemExit, FileExistsError)):
        module.main()
    assert not state["entered"]
    if kind == "file":
        assert occupied.read_bytes() == b"accepted evidence"
    else:
        assert occupied.is_symlink() and not target.exists()
    failures = [path for path in output.parent.glob("result*.json") if path != occupied]
    assert any(json.loads(path.read_text()).get("error") for path in failures)


def test_endpoint_success_exports_both_files(
    endpoint: tuple[Any, Path, dict[str, Any]],
) -> None:
    module, output, _ = endpoint
    module.main()
    record = json.loads(output.read_text())
    density = output.parent / record["final_density"]["path"]
    assert density.name == "result.density.npy"
    assert (
        record["final_density"]["sha256"]
        == hashlib.sha256(density.read_bytes()).hexdigest()
    )
    assert np.isfinite(np.load(density)).all()


@pytest.mark.parametrize("relative", [True, False])
def test_shell_receipt_path_and_resident_budget(tmp_path: Path, relative: bool) -> None:
    tools = tmp_path / "bin"
    tools.mkdir()
    experiment = tmp_path / "experiment"
    (experiment / "source").mkdir(parents=True)
    (experiment / "library").mkdir()
    (experiment / "library/libgenerativeqc.so").write_text("fixture")
    for name in ("scontrol", "nvidia-smi", "sccache"):
        command = tools / name
        command.write_text("#!/bin/sh\nprintf 'fixture\\n'\n")
        command.chmod(0o755)
    python = tools / "fake-python"
    python.write_text(
        "#!/bin/sh\nif [ \"$1\" = - ]; then cat >/dev/null; printf '{}\\n'; exit 0; fi\n"
        'while [ $# -gt 0 ]; do if [ "$1" = --output ]; then shift; printf \'{}\\n\' > "$1"; fi; shift; done\n'
        "printf 'fixture endpoint budget=%s\\n' \"$GENERATIVEQC_KS_RESIDENT_MAX_BYTES\"\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "EXPERIMENT_ROOT": str(experiment),
        "PYTHON": str(python),
        "SLURM_JOB_ID": "fixture",
        "CUDA_VISIBLE_DEVICES": "fixture",
        "GENERATIVEQC_KS_RESIDENT_MAX_BYTES": "4294967296",
    }
    result = subprocess.run(
        [
            "bash",
            str(RUNNERS / "run-endpoint.sh"),
            "fast",
            "12",
            "auto",
            "relative-receipt" if relative else str(tmp_path / "relative-receipt"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = tmp_path / "relative-receipt"
    assert (receipt / "result.json").is_file()
    assert (receipt / "exit-code.txt").read_text().strip() == "0"
    assert "budget=2147483648" in (receipt / "endpoint.log").read_text()


@pytest.mark.parametrize("observed", ["fixture-source", "different-source"])
def test_optimized_python_still_authenticates_source_and_library(
    tmp_path: Path, observed: str
) -> None:
    commands = tmp_path / "bin"
    commands.mkdir()
    experiment = tmp_path / "experiment"
    source = experiment / "source"
    (source / "tools").mkdir(parents=True)
    (experiment / "library").mkdir()
    library = experiment / "library/libgenerativeqc.so"
    library.write_bytes(b"fixture library: never loaded by native ctypes")
    library_before = library.read_bytes()
    # Real optimized Python executes the wrapper's entire authentication script.
    # Only source inventory and ctypes are faked, so no binary or GPU is loaded.
    (source / "tools/__init__.py").write_text("")
    (source / "tools/generate_build_identity.py").write_text(
        "def _inventory(*args): return []\n"
        "def _source_identity(*args): return 'fixture-source'\n"
    )
    (source / "ctypes.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "c_char_p = object()\n"
        "class Identity:\n"
        "    def __call__(self): return os.environ['FIXTURE_OBSERVED'].encode()\n"
        "class CDLL:\n"
        "    def __init__(self, path):\n"
        "        self.generativeqc_get_source_identity = Identity()\n"
        "        Path(os.environ['FIXTURE_AUTH_LOG']).write_text(json.dumps({\n"
        "            'library': path, 'visibility': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        "            'optimized': sys.flags.optimize}))\n"
    )
    (experiment / "endpoint.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output', type=Path)\n"
        "args, _ = parser.parse_known_args()\n"
        "args.output.write_text('{}\\n')\n"
        "print('fixture endpoint completed')\n"
    )
    for name in ("scontrol", "nvidia-smi", "sccache"):
        command = commands / name
        command.write_text("#!/bin/sh\nprintf 'fixture\\n'\n")
        command.chmod(0o755)
    receipt = tmp_path / "receipt"
    auth_log = tmp_path / "authentication-observed.json"
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "EXPERIMENT_ROOT": str(experiment),
        "PYTHON": sys.executable,
        "PYTHONOPTIMIZE": "1",
        "SLURM_JOB_ID": "fixture",
        "CUDA_VISIBLE_DEVICES": "scheduler-owned-device",
        "FIXTURE_OBSERVED": observed,
        "FIXTURE_AUTH_LOG": str(auth_log),
    }
    result = subprocess.run(
        ["bash", str(RUNNERS / "run-endpoint.sh"), "fast", "12", "auto", str(receipt)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert json.loads(auth_log.read_text()) == {
        "library": str(library),
        "visibility": "scheduler-owned-device",
        "optimized": 1,
    }
    assert library.read_bytes() == library_before
    assert (receipt / "visibility.txt").read_text().strip() == "scheduler-owned-device"
    if observed == "fixture-source":
        assert result.returncode == 0, result.stderr
        assert (receipt / "result.json").is_file()
    else:
        assert result.returncode != 0
        assert "Source/library identity mismatch" in result.stderr
        assert not (receipt / "result.json").exists()
