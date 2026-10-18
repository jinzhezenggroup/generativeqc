"""Production comparison gates stay independent of either candidate's grid."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.grid_pruning_gpu import (
    accuracy_gates,
    errors,
    main,
    qualification_errors,
)


def test_production_pruning_gates_preserve_per_water_energy_and_max_force() -> None:
    assert accuracy_gates("lda", 3) == {"energy": 1e-6, "force": 5e-5}
    assert accuracy_gates("pbe", 3) == {"energy": 1.5e-6, "force": 3e-5}
    for method in ("lda", "pbe"):
        small, large = accuracy_gates(method, 3), accuracy_gates(method, 96)
        assert large["energy"] == 32 * small["energy"]
        assert large["force"] == small["force"]
    with pytest.raises(ValueError, match="only LDA and PBE"):
        accuracy_gates("wb97m-v", 96)


def test_production_comparison_uses_total_energy_and_max_absolute_force() -> None:
    reference = {"energy": -100.0, "forces": [[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]]}
    candidate = {
        "energy": -99.0,
        "forces": np.array([[1.0, -2.5, 3.0], [0.0, 0.0, 0.0]]),
    }
    assert errors(candidate, reference) == {"energy": 1.0, "force": 0.5}


def test_qualification_reserves_observed_reference_variation() -> None:
    gate = accuracy_gates("pbe", 96)
    measured = {"energy": 3.4e-5, "force": 2.98e-5}
    stability = {"energy": 6e-7, "force": 5e-7}
    admitted = qualification_errors(measured, stability)
    assert measured["force"] < gate["force"] < admitted["force"]
    assert admitted["energy"] < gate["energy"]


def test_gpu_qualification_requires_scheduler_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="srun"):
        main(["--output", str(tmp_path / "qualification.json")])


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling"])
def test_qualification_never_reuses_an_existing_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    from benchmarks import grid_pruning_gpu as runner

    output, target = tmp_path / "receipt.json", tmp_path / "target.json"
    if kind == "file":
        output.write_text("earlier receipt")
    elif kind == "directory":
        output.mkdir()
    else:
        if kind == "symlink":
            target.write_text("earlier target")
        output.symlink_to(target)
    monkeypatch.setenv("SLURM_JOB_ID", "test-only-no-gpu")

    def forbidden(*args: object) -> None:
        pytest.fail("existing output reached runtime initialization")

    monkeypatch.setattr(runner, "collect", forbidden)
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(output)])
    if kind == "file":
        assert output.read_text() == "earlier receipt"
    elif kind == "directory":
        assert output.is_dir() and not list(output.iterdir())
    else:
        assert output.is_symlink()
        assert (
            target.read_text() == "earlier target"
            if kind == "symlink"
            else not target.exists()
        )


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_gpu_qualification_rejects_retained_output_before_gpu_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, destination_kind: str
) -> None:
    """Reject retained paths before allocation checks, imports or evidence writes."""
    from benchmarks import _retention

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "qualification.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(destination / existing.name)])
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()


def test_qualification_retains_partial_failure_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from benchmarks import grid_pruning_gpu as runner

    def fail_after_sample(args: object, report: dict, save: object) -> None:
        report["methods"] = {"lda": {"failed_candidate": {"energy": -76.0}}}
        save()
        raise RuntimeError("injected endpoint failure")

    monkeypatch.setattr(runner, "collect", fail_after_sample)
    monkeypatch.setenv("SLURM_JOB_ID", "test-only-no-gpu")
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        runner.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "injected endpoint failure" in report["error"]
    assert report["methods"]["lda"]["failed_candidate"]["energy"] == -76.0


@pytest.mark.parametrize("changed", [None, "library_sha256", "python_source_sha256"])
def test_qualification_rechecks_source_identity_before_passing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str | None
) -> None:
    import json

    from benchmarks import grid_pruning_gpu as runner

    initial = {
        "library_path": "/selected.so",
        "library_sha256": "before",
        "python_source_sha256": {"policy.py": "before"},
    }
    final = dict(initial)
    if changed:
        final[changed] = "after"

    def collected(args: object, report: dict, save: object) -> None:
        report["execution_identity"] = initial
        report["methods"] = {"lda": {"samples": [1, 2]}}
        save()

    monkeypatch.setattr(runner, "collect", collected)
    monkeypatch.setattr(runner, "execution_identity", lambda: final)
    monkeypatch.setenv("SLURM_JOB_ID", "test-only-no-gpu")
    output = tmp_path / "identity.json"
    if changed:
        with pytest.raises(RuntimeError, match="sources changed"):
            runner.main(["--output", str(output)])
    else:
        runner.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == ("failed" if changed else "passed")
    assert report["execution_identity"] == initial
    assert report["execution_identity_after"] == final
    assert report["methods"]["lda"]["samples"] == [1, 2]


def test_identity_binds_selected_library_and_python_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    import sys
    from types import ModuleType, SimpleNamespace

    from benchmarks import grid_pruning_gpu as runner

    selected, requested = tmp_path / "selected.so", tmp_path / "requested.so"
    selected.write_bytes(b"selected library")
    requested.write_bytes(b"unused environment library")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(requested))
    package_roots = []
    for name in ("generativeqc", "generativeqc_compiler"):
        root = tmp_path / name
        root.mkdir()
        (root / "policy.py").write_text("value = 1\n")
        module = ModuleType(name)
        module.__path__ = [str(root)]
        if name == "generativeqc":
            module._native = SimpleNamespace(
                load_library=lambda **kwargs: SimpleNamespace(_name=str(selected))
            )
        monkeypatch.setitem(sys.modules, name, module)
        package_roots.append(root)
    benchmark_root = tmp_path / "benchmarks"
    benchmark_root.mkdir()
    driver = benchmark_root / "grid_pruning_gpu.py"
    driver.write_text("driver = 1\n")
    helper = benchmark_root / "readme_wb97mv.py"
    helper.write_text("helper = 1\n")
    import benchmarks

    monkeypatch.setattr(benchmarks, "__path__", [str(benchmark_root)])
    monkeypatch.setattr(runner, "__file__", str(driver))
    before = runner.execution_identity()
    assert before["library_path"] == str(selected)
    assert before["library_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert str(helper) in before["python_source_sha256"]
    assert all(
        str(root / "policy.py") in before["python_source_sha256"]
        for root in package_roots
    )
    helper.write_text("helper = 2\n")
    after = runner.execution_identity()
    assert (
        before["python_source_sha256"][str(helper)]
        != after["python_source_sha256"][str(helper)]
    )
    assert before["library_sha256"] == after["library_sha256"]


def test_qualification_rejects_a_different_loaded_native_library() -> None:
    from benchmarks import grid_pruning_gpu as runner

    with pytest.raises(RuntimeError, match="different library"):
        runner.verify_native_library(
            "native-radial-v1",
            {"library_path": "/other.so"},
            {"execution_identity": {"library_path": "/selected.so"}},
        )


def test_complete_receipt_preserves_gates_samples_and_alternating_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    import sys
    from types import SimpleNamespace

    from benchmarks import grid_pruning_gpu as runner

    identity = {
        "library_path": "/selected.so",
        "library_sha256": "same",
        "python_source_sha256": {},
    }
    monkeypatch.setattr(runner, "execution_identity", lambda: identity)
    monkeypatch.setenv("SLURM_JOB_ID", "test-only-no-gpu")
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    getDeviceProperties=lambda _: {"name": b"fake GPU"}
                )
            )
        ),
    )
    for name in ("pyscf", "gpu4pyscf"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="test"))
    monkeypatch.setitem(
        sys.modules,
        "benchmarks.readme_hf_scaling",
        SimpleNamespace(
            scaling_cases=lambda: {
                "water-3": SimpleNamespace(atoms=(("H", (0, 0, 0)),) * 3)
            }
        ),
    )
    calls = []

    def native(method: str, atoms: tuple, pruning: str) -> dict:
        calls.append("native-" + pruning)
        # Retain an explicitly unqualified baseline instead of upgrading it.
        return {
            "energy": 2e-6 if pruning == "none" else 1e-7,
            "forces": [[0, 0, 0]] * 3,
            "seconds": 2.0 if pruning == "none" else 1.0,
            "library_path": "/selected.so",
        }

    def reference(method: str, atoms: tuple, level: int, pruning: bool) -> dict:
        calls.append(f"reference-{level}-{pruning}")
        return {
            "energy": 1e-8 if level == 8 else 0.0,
            "forces": [[0, 0, 0]] * 3,
            "seconds": 0.5,
        }

    monkeypatch.setattr(runner, "native_endpoint", native)
    monkeypatch.setattr(runner, "reference_endpoint", reference)
    output = tmp_path / "complete.json"
    runner.main(
        ["--atoms", "3", "--methods", "lda", "--repeats", "2", "--output", str(output)]
    )
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    candidates = report["methods"]["lda"]["candidates"]
    assert not candidates["native-none"]["qualified"]
    assert candidates["native-radial-v1"]["qualified"]
    assert report["methods"]["lda"]["selected"]["native"] == "native-radial-v1"
    assert all(len(candidate["samples"]) == 2 for candidate in candidates.values())
    assert candidates["native-radial-v1"]["qualification_errors"][
        "energy"
    ] == pytest.approx(1.1e-7)
    forward = [
        "native-none",
        "native-radial-v1",
        "reference-3-False",
        "reference-3-True",
    ]
    assert calls == [
        "reference-8-True",
        "reference-9-True",
        *forward,
        *forward,
        *reversed(forward),
    ]


def test_fresh_reference_reports_no_converged_state_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from collections.abc import Iterator
    from contextlib import contextmanager
    from types import SimpleNamespace

    from benchmarks import (
        compare_df_direct_endpoint,
        compare_gpu4pyscf_batch,
        grid_pruning_gpu,
        readme_wb97mv,
    )

    engine = SimpleNamespace(
        grids=SimpleNamespace(coords=np.zeros((3, 3)), weights=np.ones(3)),
        converged=True,
        kernel=lambda: -1.0,
        nuc_grad_method=lambda: SimpleNamespace(kernel=lambda: np.zeros((3, 3))),
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(deviceSynchronize=lambda: None)
            ),
            asnumpy=np.asarray,
            count_nonzero=np.count_nonzero,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.dft",
        SimpleNamespace(
            gen_grid=SimpleNamespace(nwchem_prune=object()),
            rks=SimpleNamespace(RKS=lambda _: engine),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(gto=SimpleNamespace(M=lambda **kwargs: object())),
    )

    @contextmanager
    def work_counter(value: object) -> Iterator[dict]:
        yield {"scf_jk_builds": 2, "tracker": object()}

    def convergence(
        engines: list, trackers: list, *, warm_start_used: bool = True
    ) -> list:
        return [{"warm_start": {"used": warm_start_used}}]

    monkeypatch.setattr(
        compare_df_direct_endpoint, "reference_work_counter", work_counter
    )
    monkeypatch.setattr(compare_gpu4pyscf_batch, "gpu_convergence_payload", convergence)
    monkeypatch.setattr(readme_wb97mv, "configure_reference_full_fock", lambda _: None)
    result = grid_pruning_gpu.reference_endpoint("lda", (), 3, True)
    assert result["work"]["convergence"]["warm_start"]["used"] is False
    assert result["work"]["scf_jk_builds"] == 2
