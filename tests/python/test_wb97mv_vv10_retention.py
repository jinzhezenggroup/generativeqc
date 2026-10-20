"""Exercise qualification receipt ownership and late-failure retention without CUDA."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "benchmarks/results/wb97mv-vv10-default-20261018"


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _endpoint(atoms: int) -> dict[str, Any]:
    return {
        "succeeded": True,
        "converged": True,
        "energy": -1.0,
        "iterations": 2,
        "fock_builds": 2,
        "density_rms": 0.0,
        "physical_residual_rms": 0.0,
        "energy_change": 0.0,
        "initial_guess": {"target_attempts": 1},
        "warm_start_used": False,
        "warm_start_fallback": False,
        "precision": {
            "requested_mode": "fp64",
            "effective_bits": 64,
            "mixed_stage_fock_builds": 0,
            "fallback_count": 0,
            "execution_retries": 0,
        },
        "ks_diagnostic": {
            "history": [{"operator_scope": "full-target"}] * 2,
            "nonlocal_work": {"provisional_focks": 0, "target_focks": 2},
            "physical_residual_max": 0.0,
            "density_change_max": 0.0,
        },
        "forces": np.zeros((atoms, 3)).tolist(),
        "finite_forces": True,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "grid": [48, 16, 32],
        "density_fitting": "none",
        "basis_identity": "test-basis",
        "basis_sha256": "test-basis-sha",
        "coordinates_bohr": [[0.0, 0.0, 0.0]] * atoms,
        "production_defaults_changed": True,
        "experimental_proposal_policy": None,
        "native_source_identity": "test-identity",
        "native_build": {"probe": {"source_identity": "test-identity"}},
        "prepare_seconds": 8.0,
        "execute_seconds": 12.0,
        "cold_endpoint_seconds": 20.0,
    }


class _DensityState(ctypes.Structure):
    _fields_ = [
        ("present", ctypes.c_int),
        ("density_count", ctypes.c_size_t),
        ("coordinate_count", ctypes.c_size_t),
        ("density", ctypes.c_void_p),
        ("coordinates", ctypes.c_void_p),
    ]


class ProbeRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "result.json"
        self.basis = self.root / "basis.json"
        self.basis.write_text("basis")
        self.library = self.root / "library.so"
        self.library.write_bytes(b"stub-library")
        self.time = 0.0
        self.construct_error = self.execute_error = self.cleanup_error = None
        self.after_preflight = None
        self.constructed = False
        self.density_present, self.density_count = True, 4
        self.item = _endpoint(12)

        def query(batch: object, index: int, pointer: Any) -> int:
            state = pointer._obj
            state.present = self.density_present
            state.density_count, state.coordinate_count = self.density_count, 3
            if state.density:
                density = np.ctypeslib.as_array(
                    ctypes.cast(state.density, ctypes.POINTER(ctypes.c_double)),
                    shape=(4,),
                )
                density[:] = [1.0, 0.0, 0.0, 1.0]
                self.time += 11
            return 0

        fixture = self

        class Owner:
            _library = types.SimpleNamespace(generativeqc_batch_get_hf_warm_state=query)
            _batch = object()

            def __enter__(self) -> Self:
                fixture.time += 3
                return self

            def __exit__(self, *error: object) -> None:
                fixture.time += 13
                if fixture.cleanup_error:
                    raise RuntimeError(fixture.cleanup_error)

            def execute(self, **kwargs: Any) -> types.SimpleNamespace:
                fixture.time += 7
                if fixture.execute_error:
                    raise RuntimeError(fixture.execute_error)
                values = copy.deepcopy(fixture.item)
                values["forces"] = np.asarray(values["forces"])
                diagnostic = values["ks_diagnostic"]
                values["ks_diagnostic"] = types.SimpleNamespace(
                    to_payload=lambda: diagnostic
                )
                return types.SimpleNamespace(items=[types.SimpleNamespace(**values)])

        class Calculator:
            def __init__(self, **kwargs: Any) -> None:
                fixture.constructed = True
                fixture.time += 2
                if fixture.construct_error:
                    raise RuntimeError(fixture.construct_error)

            def prepare_batch(self, *args: object, **kwargs: Any) -> Owner:
                return Owner()

        def synchronize() -> None:
            self.time += 5

        def identity(*args: object) -> str:
            if self.after_preflight:
                self.after_preflight()
            return "test-identity"

        modules = {
            "cupy": types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    runtime=types.SimpleNamespace(deviceSynchronize=synchronize)
                )
            ),
            "generativeqc": types.SimpleNamespace(
                Calculator=Calculator,
                GridSpec=lambda *args: args,
                KsOptions=lambda **kwargs: kwargs,
                _native=types.SimpleNamespace(check=lambda *args: None),
            ),
            "generativeqc.checkpoint": types.SimpleNamespace(
                _descriptor=_DensityState, _pointer=lambda array: array.ctypes.data
            ),
            "benchmarks.compare_gpu4pyscf_batch": types.SimpleNamespace(
                load_comparison_basis=lambda *args, **kwargs: (
                    types.SimpleNamespace(identity="test-basis"),
                    None,
                ),
                native_build_metadata=lambda calculator: {
                    "probe": {"source_identity": "test-identity"}
                },
            ),
            "benchmarks.readme_hf_scaling": types.SimpleNamespace(
                scaling_cases=lambda: {
                    "water-12": types.SimpleNamespace(
                        atoms=[("H", [0.0, 0.0, 0.0])] * 12
                    )
                }
            ),
            "tools.generate_build_identity": types.SimpleNamespace(
                _inventory=lambda *args: (), _source_identity=identity
            ),
        }
        self.module_patch = mock.patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.probe = _load("vv10_test_probe", _SCRIPTS / "probe.py")
        function = mock.Mock(return_value=b"test-identity")
        self.patches = [
            mock.patch.object(
                self.probe.ctypes,
                "CDLL",
                return_value=types.SimpleNamespace(
                    generativeqc_get_source_identity=function
                ),
            ),
            mock.patch.object(
                self.probe, "perf_counter", side_effect=lambda: self.time
            ),
            mock.patch.dict(
                "os.environ",
                {
                    "SLURM_JOB_ID": "123",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "GENERATIVEQC_LIBRARY": str(self.library),
                },
                clear=True,
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    "probe",
                    "--atoms",
                    "12",
                    "--basis",
                    str(self.basis),
                    "--source",
                    str(self.root),
                    "--output",
                    str(self.output),
                ],
            ),
            mock.patch("builtins.print"),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_success_preserves_endpoint_timing_and_density_binding(self) -> None:
        self.probe.main()
        record = json.loads(self.output.read_text())
        density = self.output.with_suffix(".density.npy")
        self.assertNotIn("error", record)
        self.assertEqual(record["construction_seconds"], 2.0)
        self.assertEqual(record["prepare_seconds"], 8.0)
        self.assertEqual(record["execute_seconds"], 12.0)
        self.assertEqual(record["cold_endpoint_seconds"], 20.0)
        self.assertEqual(record["final_density"]["path"], density.name)
        self.assertEqual(
            record["final_density"]["sha256"],
            hashlib.sha256(density.read_bytes()).hexdigest(),
        )
        np.testing.assert_array_equal(np.load(density), [1.0, 0.0, 0.0, 1.0])

    def test_existing_receipt_paths_are_never_modified(self) -> None:
        for suffix in (".json", ".density.npy"):
            for kind in ("file", "directory", "symlink", "dangling-symlink"):
                with (
                    self.subTest(suffix=suffix, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    self.constructed = False
                    self.output = Path(directory) / "result.json"
                    path = self.output.with_suffix(suffix)
                    target = Path(directory) / "target"
                    if kind == "file":
                        path.write_bytes(b"retained")
                    elif kind == "directory":
                        path.mkdir()
                    else:
                        if kind == "symlink":
                            target.write_bytes(b"retained")
                        path.symlink_to(target)
                    with (
                        mock.patch.object(
                            sys,
                            "argv",
                            [
                                "probe",
                                "--atoms",
                                "12",
                                "--basis",
                                str(self.basis),
                                "--source",
                                str(self.root),
                                "--output",
                                str(self.output),
                            ],
                        ),
                        self.assertRaises(FileExistsError),
                    ):
                        self.probe.main()
                    self.assertFalse(self.constructed)
                    if kind == "file":
                        self.assertEqual(path.read_bytes(), b"retained")
                    elif kind == "directory":
                        self.assertTrue(path.is_dir())
                    else:
                        self.assertTrue(path.is_symlink())
                        if kind == "symlink":
                            self.assertEqual(target.read_bytes(), b"retained")
                        else:
                            self.assertFalse(target.exists())

    def test_paths_created_after_preflight_are_never_modified(self) -> None:
        for suffix in (".json", ".density.npy"):
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.constructed = False
                self.output = Path(directory) / "result.json"
                path = self.output.with_suffix(suffix)
                target = Path(directory) / "target"
                self.after_preflight = lambda path=path, target=target: path.symlink_to(
                    target
                )
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "probe",
                            "--atoms",
                            "12",
                            "--basis",
                            str(self.basis),
                            "--source",
                            str(self.root),
                            "--output",
                            str(self.output),
                        ],
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    self.probe.main()
                self.assertFalse(self.constructed)
                self.assertTrue(path.is_symlink())
                self.assertFalse(target.exists())

    def test_authentication_failure_still_blocks_execution(self) -> None:
        self.probe.ctypes.CDLL.return_value.generativeqc_get_source_identity.return_value = b"other-source"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.probe.main()
        self.assertFalse(self.constructed)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".density.npy").exists())

    def test_experiment_override_still_blocks_execution(self) -> None:
        with (
            mock.patch.dict(
                "os.environ", {"GENERATIVEQC_EXPERIMENT_VV10_INNER_POLICY": "override"}
            ),
            self.assertRaisesRegex(RuntimeError, "forbids an override"),
        ):
            self.probe.main()
        self.assertFalse(self.constructed)
        self.assertFalse(self.output.exists())

    def test_constructor_failure_is_retained(self) -> None:
        self.construct_error = "construction failed"
        with self.assertRaises(SystemExit) as failure:
            self.probe.main()
        self.assertEqual(failure.exception.code, 1)
        self.assertIn(
            "construction failed", json.loads(self.output.read_text())["error"]
        )

    def test_successful_endpoint_with_cleanup_failure_is_not_success(self) -> None:
        self.cleanup_error = "cleanup failed"
        with self.assertRaises(SystemExit) as failure:
            self.probe.main()
        self.assertEqual(failure.exception.code, 1)
        record = json.loads(self.output.read_text())
        self.assertTrue(record["succeeded"] and record["converged"])
        self.assertIn("cleanup failed", record["error"])
        self.assertTrue(self.output.with_suffix(".density.npy").exists())

    def test_missing_or_unbounded_density_is_rejected(self) -> None:
        for present, count in ((False, 4), (True, 0), (True, 768**2 + 1)):
            with (
                self.subTest(present=present, count=count),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.density_present, self.density_count = present, count
                output = Path(directory) / "result.json"
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "probe",
                            "--atoms",
                            "12",
                            "--basis",
                            str(self.basis),
                            "--source",
                            str(self.root),
                            "--output",
                            str(output),
                        ],
                    ),
                    self.assertRaises(SystemExit) as failure,
                ):
                    self.probe.main()
                self.assertEqual(failure.exception.code, 1)
                record = json.loads(output.read_text())
                self.assertIn("missing or unbounded final density", record["error"])
                self.assertNotIn("final_density", record)

    def test_density_export_and_cleanup_errors_both_survive(self) -> None:
        self.cleanup_error = "cleanup failed"
        with (
            mock.patch.object(
                self.probe.np, "save", side_effect=OSError("density write failed")
            ),
            self.assertRaises(SystemExit) as failure,
        ):
            self.probe.main()
        self.assertEqual(failure.exception.code, 1)
        error = json.loads(self.output.read_text())["error"]
        self.assertIn("density write failed", error)
        self.assertIn("cleanup failed", error)

    def test_execution_and_cleanup_errors_both_survive(self) -> None:
        self.execute_error = "execution failed"
        self.cleanup_error = "cleanup failed"
        with self.assertRaises(SystemExit):
            self.probe.main()
        error = json.loads(self.output.read_text())["error"]
        self.assertIn("execution failed", error)
        self.assertIn("cleanup failed", error)


class AnalysisRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyze = _load("vv10_test_analyze", _SCRIPTS / "analyze.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"
        self.historical = Path(self.temporary.name) / "historical"
        self.job = self.root / "job123"
        self.output = Path(self.temporary.name) / "evidence.json"
        self.root.mkdir()
        self.historical.mkdir()

        def write(path: Path, content: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        for name in (
            "srun-exit-code.txt",
            "restore-exit-code.txt",
            "expanded-tests-exit-code.txt",
        ):
            write(self.root / name, "0\n")
        write(self.root / "expanded-tests.log", "1 passed")
        write(
            self.root / "source/tools/generate_build_identity.py",
            "from pathlib import Path\ndef _inventory(*args): return [Path('src/dft/application_precision.hpp')]\n"
            "def _source_identity(*args): return 'test-identity'\n",
        )
        write(self.root / "source/src/dft/application_precision.hpp", "default")
        write(
            self.historical / "source-master/src/dft/application_precision.hpp",
            "master",
        )
        write(self.root / "library/libgenerativeqc.so", "stub-library")
        library_hash = self.analyze.digest(self.root / "library/libgenerativeqc.so")
        write(
            self.root / "build-receipts/source-authentication.json",
            json.dumps(
                {"native_identity": "test-identity", "base_revision": "test-base"}
            ),
        )
        write(self.root / "build-receipts/library.sha256", library_hash)
        write(self.job / "library-after.sha256", library_hash)
        write(self.job / "visibility.txt", "0")
        write(self.job / "pytest.log", "1 passed")
        write(self.job / "input-hashes.txt", library_hash + " libgenerativeqc.so\n")
        reference = {
            "records": [
                {"phase": "cold", "energy": -1.0, "forces": np.zeros((12, 3)).tolist()}
            ],
            "historical_binding": {},
            "protocol": {
                "basis_identity": "test-basis",
                "geometries_bohr": [[["H", [0.0, 0.0, 0.0]]] * 12],
            },
        }
        write(self.historical / "reference12.json", json.dumps(reference))
        for atoms, baseline in (
            (12, "screen-job1120/baseline/water12.json"),
            (96, "endpoint-baseline-job1121/water96.json"),
        ):
            for path in (
                self.job / f"water{atoms}/result.json",
                self.historical / baseline,
            ):
                record = _endpoint(atoms)
                record["library_sha256"] = library_hash
                path.parent.mkdir(parents=True, exist_ok=True)
                density = path.with_suffix(".density.npy")
                np.save(density, np.array([1.0, 0.0, 0.0, 1.0]))
                record["final_density"] = {"sha256": self.analyze.digest(density)}
                write(path, json.dumps(record))

    def run_analysis(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "analyze",
                    "--root",
                    str(self.root),
                    "--historical",
                    str(self.historical),
                    "--job-id",
                    "123",
                    "--output",
                    str(self.output),
                ],
            ),
            mock.patch("builtins.print"),
        ):
            self.analyze.main()

    def test_fresh_reduction_keeps_scientific_gates(self) -> None:
        self.run_analysis()
        report = json.loads(self.output.read_text())
        self.assertEqual(len(report["records"]), 2)
        self.assertTrue(
            all(
                row["physical_gate"] and row["paired_gate"] for row in report["records"]
            )
        )
        self.assertTrue(report["records"][0]["oracle_gate"])
        self.assertIsNone(report["records"][1]["oracle_gate"])
        self.assertFalse(report["water96_independent_reference_available"])

    def test_error_receipts_fail_the_physical_gate(self) -> None:
        record = _endpoint(12)
        self.assertTrue(self.analyze.physical_gate(record, 12))
        for error in ("cleanup failed", "", None):
            with self.subTest(error=error):
                record["error"] = error
                self.assertFalse(self.analyze.physical_gate(record, 12))
        self.assertFalse(
            self.analyze.physical_gate({"error": "constructor failed"}, 12)
        )

    def test_ordinary_physical_failures_remain_rejected(self) -> None:
        for field, value in (
            ("converged", False),
            ("warm_start_used", True),
            ("energy_tolerance", 1e-8),
            ("density_rms", 1e-5),
        ):
            with self.subTest(field=field):
                record = _endpoint(12)
                record[field] = value
                self.assertFalse(self.analyze.physical_gate(record, 12))

    def test_reduction_never_overwrites_consumed_receipts(self) -> None:
        for source, relative in (
            ("job", "water12/result.json"),
            ("historical", "reference12.json"),
            ("root", "build-receipts/source-authentication.json"),
        ):
            with self.subTest(source=source, relative=relative):
                # Isolate cases even when an unfixed runner corrupts its input.
                self.setUp()
                path = getattr(self, source) / relative
                before = path.read_bytes()
                self.output = path
                with self.assertRaises(FileExistsError):
                    self.run_analysis()
                self.assertEqual(path.read_bytes(), before)

    def test_reduction_does_not_follow_output_symlinks(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing):
                target = self.output.parent / f"target-{existing}"
                output = self.output.parent / f"link-{existing}.json"
                if existing:
                    target.write_bytes(b"retained")
                output.symlink_to(target)
                self.output = output
                with self.assertRaises(FileExistsError):
                    self.run_analysis()
                self.assertTrue(output.is_symlink())
                if existing:
                    self.assertEqual(target.read_bytes(), b"retained")
                else:
                    self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
