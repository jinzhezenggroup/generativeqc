"""Host-only gates for the #2474 benchmark's DF identity and trace accounting."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks import ks_df_warm_factor as benchmark

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_full_fock_closure_binds_the_converted_df_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closure installed before density_fit could silently retain Direct K."""
    events = []

    class Reference:
        def __init__(self, fitted: bool = False) -> None:
            self.fitted = fitted

        def density_fit(self, *, auxbasis: Any, only_dfj: bool) -> Reference:
            assert auxbasis == "auxiliary"
            assert not only_dfj
            assert not hasattr(self, "get_veff")
            events.append("convert-df")
            return Reference(True)

    def engine(*arguments: Any, **keywords: Any) -> Reference:
        assert arguments == ("geometry", "orbital", "grid")
        assert keywords == {"xc": "M06_2X", "full_fock": False}
        events.append("prepare-grid")
        return Reference()

    def full_fock(reference: Reference) -> None:
        assert reference.fitted
        events.append("bind-df-full-fock")

    monkeypatch.setattr(benchmark, "reference_engine", engine)
    monkeypatch.setattr(benchmark, "configure_reference_full_fock", full_fock)
    reference = benchmark.prepare_reference(
        "geometry", "orbital", "auxiliary", "grid", "m06-2x"
    )
    assert events == ["prepare-grid", "convert-df", "bind-df-full-fock"]
    assert reference.fitted
    assert reference.conv_tol == 1e-12
    assert reference.conv_tol_grad == 1e-10
    assert reference.direct_scf_tol == 1e-14
    assert reference.max_cycle == 200


def test_ks_ledger_does_not_double_count_nested_provider_roots() -> None:
    """The top-level JK interval already contains RI-K/projection work."""
    rows = [
        {
            "operation": name,
            "counters": {"fock_build": 1},
            "regions": [{"name": name, "gpu_ms": 2.0}],
            "host_completion_ms": 3.0,
            "final_synchronization_ms": 0.1,
        }
        for name in (
            "ks_begin",
            "ri_k_occupied",
            "ks_iteration",
            "ks_finish",
            "force_response",
        )
    ]
    ledger = benchmark.ks_ledger(rows)
    assert [row["operation"] for row in ledger] == [
        "ks_begin",
        "ks_iteration",
        "ks_finish",
    ]
    assert all(row["diagnostic_synchronization_ms"] == 0.1 for row in ledger)


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"])
def test_runner_refuses_unscheduled_gpu_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    """Reject before importing GPU packages or preparing any scientific work."""
    monkeypatch.setenv("SLURM_JOB_ID", "host-test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-test-double")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ks_df_warm_factor",
            "reference",
            "--atoms",
            "48",
            "--basis-file",
            "orbital.json",
            "--auxiliary-file",
            "auxiliary.json",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["reference"],
        [
            "native",
            "--reference",
            "oracle.json",
            "--trace-directory",
            ".artifacts/trace",
        ],
        ["native", "--reference", "oracle.json", "--variant", "candidate"],
    ],
)
def test_first_k_audit_requires_traced_native_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, options: list[str]
) -> None:
    """The extra dense/replayed K work must never enter clean timing runs."""
    monkeypatch.setenv("SLURM_JOB_ID", "host-test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-test-double")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            *options,
            "--audit-first-k",
            "--atoms",
            "3",
            "--basis-file",
            "orbital.json",
            "--auxiliary-file",
            "auxiliary.json",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("fail_gate", [False, True])
def test_complete_protocol_and_failed_gate_journal_with_gpu_doubles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_gate: bool
) -> None:
    """Execute all runner control flow without a CUDA import, context or device."""
    import generativeqc

    monkeypatch.setenv("SLURM_JOB_ID", "host-test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-test-double")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "0")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT", "1")
    for variable in ("GENERATIVEQC_DF_TRACE", "GENERATIVEQC_DF_PROGRESS_TRACE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asnumpy=np.asarray,
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(deviceSynchronize=lambda: None)
            ),
        ),
    )
    atoms = [(1, (0.0, 0.0, float(index))) for index in range(3)]
    monkeypatch.setattr(
        benchmark, "scaling_cases", lambda: {"water-3": SimpleNamespace(atoms=atoms)}
    )
    monkeypatch.setattr(
        benchmark, "load_comparison_basis", lambda *args, **kwargs: ("basis", "basis")
    )
    monkeypatch.setattr(benchmark, "source_provenance", lambda: {"host_double": True})
    seeds = []

    class Reference:
        converged = True

        def density_fit(self, *, auxbasis: Any, only_dfj: bool) -> Reference:
            assert auxbasis == "basis"
            assert not only_dfj
            return self

        def kernel(self, *, dm0: Any) -> float:
            seeds.append(None if dm0 is None else dm0.copy())
            return -5.0

        def nuc_grad_method(self) -> Any:
            gradient = SimpleNamespace()

            def kernel() -> np.ndarray:
                assert gradient.grid_response and gradient.auxbasis_response
                return np.zeros((3, 3))

            gradient.kernel = kernel
            return gradient

        def make_rdm1(self) -> np.ndarray:
            return np.eye(2)

    @contextmanager
    def work_counter(engine: Any) -> Iterator[dict]:
        assert isinstance(engine, Reference)
        yield {"tracker": None, "scf_jk_builds": 1}

    monkeypatch.setattr(
        benchmark, "reference_engine", lambda *args, **kwargs: Reference()
    )
    monkeypatch.setattr(benchmark, "configure_reference_full_fock", lambda engine: None)
    monkeypatch.setattr(benchmark, "reference_work_counter", work_counter)
    monkeypatch.setattr(
        benchmark,
        "gpu_convergence_payload",
        lambda *args: [{"iterations": 1, "final_residuals": {}}],
    )
    basis_path, auxiliary_path = tmp_path / "orbital.json", tmp_path / "auxiliary.json"
    basis_path.write_text("orbital")
    auxiliary_path.write_text("auxiliary")
    reference_path, native_path = tmp_path / "reference.json", tmp_path / "native.json"
    common = [
        "--atoms",
        "3",
        "--basis-file",
        str(basis_path),
        "--auxiliary-file",
        str(auxiliary_path),
    ]
    monkeypatch.setattr(
        "sys.argv", ["benchmark", "reference", *common, "--output", str(reference_path)]
    )
    benchmark.main()
    oracle = json.loads(reference_path.read_text())
    assert oracle["status"] == "measured" and len(oracle["records"]) == 12
    assert "calculator_construction_seconds" not in oracle
    assert seeds[0] is None and seeds[6] is None
    for index, density in enumerate(seeds):
        if index not in (0, 6):
            np.testing.assert_array_equal(density, np.eye(2))

    class Owner:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False
            self.warm_updates: list[bool] = []

        def set_warm_start_updates(self, enabled: bool) -> None:
            self.warm_updates.append(enabled)

        def execute(self, coordinates: Any, **keywords: Any) -> Any:
            assert benchmark.os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT"] == "0"
            self.calls += 1
            assert keywords == {"strict": False, "properties": ("energy", "forces")}
            assert (coordinates is None) == (self.calls <= 6)
            forces = np.zeros((3, 3))
            if fail_gate and self.calls == 3:
                forces[0, 0] = 1e-6
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        succeeded=True,
                        converged=True,
                        energy=-5.0,
                        forces=forces,
                        iterations=1,
                        fock_builds=1,
                        status=0,
                        status_message="",
                        ks_diagnostic=None,
                        warm_start_used=self.calls != 1,
                        warm_start_fallback=False,
                    )
                ]
            )

        def last_density_fitting_metric_diagnostics(self) -> list:
            return []

        def close(self) -> None:
            self.closed = True

    owner = Owner()
    calculator = SimpleNamespace(prepare_batch=lambda *args, **kwargs: owner)
    monkeypatch.setattr(generativeqc, "Calculator", lambda **kwargs: calculator)
    monkeypatch.setattr(
        benchmark, "native_build_metadata", lambda calc: {"host_double": True}
    )
    monkeypatch.setattr(benchmark, "require_tuned_native_build", lambda metadata: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "native",
            *common,
            "--variant",
            "candidate",
            "--reference",
            str(reference_path),
            "--output",
            str(native_path),
        ],
    )
    if fail_gate:
        with pytest.raises(RuntimeError, match="gate failed"):
            benchmark.main()
    else:
        benchmark.main()
    native = json.loads(native_path.read_text())
    assert owner.closed
    assert native["status"] == ("failed" if fail_gate else "measured")
    assert len(native["records"]) == (3 if fail_gate else 12)
    assert native["records"][-1]["gate"] is not fail_gate
    if not fail_gate:
        assert [row["phase"] for row in native["records"]] == [
            "cold",
            *(["warm"] * 5),
            "moved",
            *(["moved-warm"] * 5),
        ]
        assert owner.warm_updates == [True, False, True, False]


@pytest.mark.parametrize("engine", ["reference", "native"])
@pytest.mark.parametrize("existing", ["file", "symlink", "dangling-symlink"])
def test_runner_reserves_output_before_scientific_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str, existing: str
) -> None:
    """Repeated runs cannot destroy successful or failed prior evidence."""
    monkeypatch.setenv("SLURM_JOB_ID", "host-test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-test-double")
    monkeypatch.delenv("GENERATIVEQC_DF_TRACE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DF_PROGRESS_TRACE", raising=False)
    output = tmp_path / "result.json"
    target = tmp_path / "previous.json"
    original = b'{"status":"failed","error":"retained diagnostic"}\n'
    if existing == "file":
        output.write_bytes(original)
    else:
        if existing == "symlink":
            target.write_bytes(original)
        output.symlink_to(target)

    def forbidden() -> None:
        raise AssertionError("existing output reached scientific preparation")

    monkeypatch.setattr(benchmark, "scaling_cases", forbidden)
    arguments = [
        "benchmark",
        engine,
        "--atoms",
        "3",
        "--basis-file",
        "orbital.json",
        "--auxiliary-file",
        "auxiliary.json",
        "--output",
        str(output),
    ]
    if engine == "native":
        arguments += ["--reference", "oracle.json"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(FileExistsError):
        benchmark.main()
    if existing == "dangling-symlink":
        assert output.is_symlink() and not target.exists()
    else:
        assert output.read_bytes() == original


def test_runner_journals_initialization_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An input failure after reservation keeps useful failure context."""
    monkeypatch.setenv("SLURM_JOB_ID", "host-test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-test-double")
    monkeypatch.delenv("GENERATIVEQC_DF_TRACE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_DF_PROGRESS_TRACE", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    error = RuntimeError("input preparation failed")

    def fail() -> None:
        raise error

    monkeypatch.setattr(benchmark, "scaling_cases", fail)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "reference",
            "--atoms",
            "3",
            "--basis-file",
            "orbital.json",
            "--auxiliary-file",
            "auxiliary.json",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="input preparation failed") as raised:
        benchmark.main()
    assert raised.value is error
    record = json.loads(output.read_text())
    assert record["status"] == "failed" and record["stage"] == "initialize"
    assert record["error"] == "RuntimeError: input preparation failed"
    assert record["engine"] == "reference" and record["records"] == []


def test_runner_records_loaded_library_not_requested_environment_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profile selection can load a different library than the requested path."""
    import hashlib

    import generativeqc.profiles

    actual = tmp_path / "loaded.so"
    requested = tmp_path / "requested.so"
    actual.write_bytes(b"loaded native library")
    requested.write_bytes(b"different requested library")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(requested))
    library = SimpleNamespace(_name=str(actual))

    def probe(loaded: Any, device: int) -> dict:
        assert loaded is library and device == 3
        return {"actual_device": device}

    monkeypatch.setattr(generativeqc.profiles, "probe_device", probe)
    metadata = benchmark.native_build_metadata(
        SimpleNamespace(_library=library, _device_id=3)
    )
    assert metadata["library_path"] == str(actual.resolve())
    assert metadata["library_sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()
    assert metadata["probe"] == {"actual_device": 3}
