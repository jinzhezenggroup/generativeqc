"""Host-only protocol/observer failure tests; no runtime or oracle impersonation."""

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, NoReturn
from unittest import mock

import numpy as np

from benchmarks import direct_phase_moved as campaign


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = [["H", [float(i), 0.0, 0.0]] for i in range(48)]

    def test_cli_rejects_retained_output_before_reading_geometry(self) -> None:
        destination = (
            Path(campaign.__file__).resolve().parents[1]
            / "benchmarks/results/direct-phase-moved-forbidden/new.json"
        )
        with self.assertRaises(SystemExit) as error, mock.patch("sys.stderr") as stderr:
            campaign.main(
                ["--geometry", "missing-geometry.json", "--output", str(destination)]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid raw_output_path value", str(stderr.write.call_args_list))
        self.assertFalse(destination.parent.exists())

    def test_seeded_moves_preserve_order_every_atom_and_vector_rms(self) -> None:
        cold = json.loads(json.dumps(self.atoms))
        small = campaign.displace(self.atoms, 0.01)
        large = campaign.displace(self.atoms, 0.05)
        self.assertEqual(self.atoms, cold)
        self.assertEqual(small, campaign.displace(self.atoms, 0.01))
        self.assertEqual([a[0] for a in small["geometry_bohr"]], [a[0] for a in cold])
        self.assertAlmostEqual(small["actual_rms_bohr"], 0.01, places=14)
        self.assertAlmostEqual(large["actual_rms_bohr"], 0.05, places=14)
        self.assertTrue(np.all(np.linalg.norm(small["displacements_bohr"], axis=1) > 0))
        np.testing.assert_allclose(
            np.array(small["displacements_bohr"]) * 5,
            large["displacements_bohr"],
            atol=3e-14,
        )
        self.assertNotEqual(small["geometry_sha256"], large["geometry_sha256"])

    def test_bad_geometry_and_displacements_fail(self) -> None:
        for value in (
            None,
            [],
            [["H", [0, 1]]],
            [["H", [0, True, 0]]],
            [["H", [0, float("nan"), 0]]],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                campaign.displace(value, 0.01)
        for rms in (0, -1, float("inf"), True):
            with self.subTest(rms=rms), self.assertRaises(ValueError):
                campaign.displace(self.atoms, rms)

    def test_complete_declared_cohort_and_parent_remaining(self) -> None:
        plan = campaign.plan(self.atoms)
        self.assertEqual(plan["scientific_status"], "INCOMPLETE")
        self.assertEqual(len(plan["samples"]), 28)
        self.assertEqual(len(plan["oracle_geometries"]), 3)
        for engine in ("standard", "gpu4pyscf"):
            for population in ("clean", "instrumented"):
                rows = [
                    r
                    for r in plan["samples"]
                    if r["engine"] == engine and r["population"] == population
                ]
                self.assertEqual(
                    [r["phase"] for r in rows], ["cold", "warm", "warm", "warm"]
                )
            moved = [
                r
                for r in plan["samples"]
                if r["engine"] == engine and r["phase"] == "moved"
            ]
            self.assertEqual(len(moved), 6)
            self.assertTrue(
                all(r["seed_policy"] == "immutable_own_cold_density" for r in moved)
            )
        self.assertEqual(
            plan["remaining_issue_scope"],
            ["direct/pbe0/96", "direct/wb97m-v/48", "direct/wb97m-v/96"],
        )

    def test_plan_cannot_overwrite_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "geometry.json"
            output = Path(directory) / "plan.json"
            source.write_text(json.dumps(self.atoms), encoding="utf-8")
            args = ["--geometry", str(source), "--output", str(output)]
            campaign.main(args)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                campaign.main(args)
            self.assertEqual(output.read_bytes(), original)


class ObserverTests(unittest.TestCase):
    def test_arguments_aliases_return_and_nested_identity_preserved(self) -> None:
        observer = campaign.ApiObserver()
        operand = object()
        returned = object()
        seen = []

        class Engine:
            def inner(self, value: object, *, omega: float) -> object:
                seen.append((value, omega))
                return returned

            def outer(self, value: object, *, omega: float) -> object:
                return self.inner(value, omega=omega)

        engine = Engine()
        with observer.installed(engine, {"outer": "get_jk", "inner": "get_k"}):
            self.assertIs(engine.outer(operand, omega=-0.3), returned)
        self.assertEqual(vars(engine), {})
        self.assertIs(seen[0][0], operand)
        self.assertEqual(seen[0][1], -0.3)
        child, parent = observer.events
        self.assertEqual(child["parent"], parent["id"])
        self.assertIsNone(parent["parent"])
        self.assertTrue(
            all(
                not row["additive"] and not row["pure_kernel"]
                for row in observer.events
            )
        )

    def test_original_exception_and_instance_override_restored(self) -> None:
        observer = campaign.ApiObserver()
        failure = ValueError("scientific failure")

        def compute() -> NoReturn:
            raise failure

        class Engine:
            pass

        engine = Engine()
        engine.compute = compute
        with (
            self.assertRaises(ValueError) as caught,
            observer.installed(engine, {"compute": "force"}),
        ):
            engine.compute()
        self.assertIs(caught.exception, failure)
        self.assertIs(engine.compute, compute)
        self.assertEqual(observer.events[0]["status"], "exception")

    def test_partial_install_unwinds_and_missing_phase_is_unavailable(self) -> None:
        observer = campaign.ApiObserver()

        class Engine:
            def compute(self) -> int:
                return 3

        engine = Engine()
        with (
            self.assertRaises(AttributeError),
            observer.installed(engine, {"compute": "xc", "missing": "vv10"}),
        ):
            self.fail("must not enter")
        self.assertEqual(vars(engine), {})
        self.assertEqual(engine.compute(), 3)
        self.assertTrue(
            all(
                row == {"status": "UNAVAILABLE", "seconds": None}
                for row in observer.receipt()["phases"].values()
            )
        )

    def test_observer_clock_failure_does_not_replace_return_or_exception(self) -> None:
        def bad_clock() -> NoReturn:
            raise RuntimeError("clock unavailable")

        observer = campaign.ApiObserver(clock=bad_clock)
        value = object()
        self.assertIs(observer.wrap(lambda: value, "xc")(), value)
        failure = ValueError("original")

        def fail() -> NoReturn:
            raise failure

        with self.assertRaises(ValueError) as caught:
            observer.wrap(fail, "force")()
        self.assertIs(caught.exception, failure)
        self.assertTrue(observer.errors)


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]] * 48,
            "converged": True,
            "geometry_sha256": "geometry",
            "engine": "standard",
            "sdk_identity": "sdk",
            "gpu_uuid": "gpu",
            "seed_identity": "cold-density",
            "work": {
                "iterations": 2,
                "fock_builds": 4,
                "warm_start_used": True,
                "warm_start_fallback": False,
                "vv10_reuse": None,
            },
        }

    def test_incomplete_force_shapes_and_nonfinite_energy_rejected(self) -> None:
        for changes in (
            {"forces": [[0, 0, 0]]},
            {"forces": None},
            {"energy": float("nan")},
            {"energy": True},
            {"converged": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                campaign.observable_errors({**self.sample, **changes}, self.sample, 48)

    def test_all_observable_gates_and_quarter_margin(self) -> None:
        changed = {**self.sample, "energy": self.sample["energy"] + 48 * 5e-6}
        self.assertTrue(campaign.observable_errors(changed, self.sample, 48)["passed"])
        self.assertFalse(
            campaign.observable_errors(changed, self.sample, 48, fraction=0.25)[
                "passed"
            ]
        )
        changed = {**self.sample, "forces": [[0.0, 0.0, 1e-3]] * 48}
        self.assertFalse(campaign.observable_errors(changed, self.sample, 48)["passed"])

    def test_parity_is_not_independent_science_and_work_changes_fail(self) -> None:
        report = campaign.observer_parity(self.sample, self.sample, 48)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scientific_acceptance"], "INCOMPLETE")
        instrumented = {
            **self.sample,
            "work": {**self.sample["work"], "fock_builds": 5},
        }
        self.assertFalse(
            campaign.observer_parity(self.sample, instrumented, 48)["passed"]
        )
        for key in ("geometry_sha256", "seed_identity", "gpu_uuid", "sdk_identity"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                campaign.observer_parity(
                    self.sample, {**self.sample, key: "different"}, 48
                )
        with self.assertRaises(ValueError):
            campaign.observer_parity(self.sample, {**self.sample, "work": {}}, 48)


class PopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.0]]]
        self.events = []
        self.rows = []
        self.tick = 0
        self.fail_call = None
        self.fallback = False
        self.fail_close = False
        self.failure = ValueError("scientific endpoint failed")

    def clock(self) -> float:
        self.tick += 1
        self.events.append("clock")
        return float(self.tick)

    def factory(self) -> Any:
        parent = self
        self.events.append("construct")

        class Adapter:
            def __init__(self) -> None:
                self.calls = 0
                self.frozen = False

            def execute(self, coordinates: Any) -> int:
                parent.events.append(
                    ("execute", copy.deepcopy(coordinates), self.frozen)
                )
                self.calls += 1
                if parent.fail_call == self.calls:
                    raise parent.failure
                return self.calls

            def freeze_cold(self) -> None:
                self.frozen = True
                parent.events.append("freeze")

            def record(self, result: int) -> dict:
                parent.events.append("serialize")
                return {
                    "energy": -1.0,
                    "converged": True,
                    "forces": [[0.0, 0.0, 0.0]] * 2,
                    "work": {
                        "warm_start_used": result != 1,
                        "warm_start_fallback": parent.fallback,
                    },
                }

            def close(self) -> None:
                parent.events.append("close")
                if parent.fail_close:
                    raise RuntimeError("cleanup failed")

        return Adapter()

    def run_population(self, moved: Any = None) -> None:
        campaign.measure_population(
            self.factory,
            self.atoms,
            moved=moved,
            synchronize=lambda: self.events.append("sync"),
            retain=self.rows.append,
            clock=self.clock,
        )

    def test_complete_endpoint_boundaries_and_frozen_seed(self) -> None:
        self.run_population()
        self.assertEqual(
            [row["phase"] for row in self.rows], ["cold", "warm", "warm", "warm"]
        )
        self.assertEqual(self.events[:3], ["sync", "clock", "construct"])
        calls = [event for event in self.events if isinstance(event, tuple)]
        self.assertEqual([call[2] for call in calls], [False, True, True, True])
        self.assertTrue(all(call[1] is None for call in calls))
        for index, event in enumerate(self.events):
            if event == "serialize":
                self.assertEqual(self.events[index - 2 : index], ["sync", "clock"])
        self.assertEqual(self.events[-1], "close")

    def test_bounded_pilot_is_one_complete_cold_call(self) -> None:
        campaign.measure_population(
            self.factory,
            self.atoms,
            synchronize=lambda: self.events.append("sync"),
            retain=self.rows.append,
            clock=self.clock,
            warm_repeats=0,
        )
        self.assertEqual([row["phase"] for row in self.rows], ["cold"])
        self.assertEqual(self.events[-1], "close")
        with self.assertRaises(ValueError):
            campaign.measure_population(
                self.factory,
                self.atoms,
                synchronize=lambda: None,
                retain=self.rows.append,
                warm_repeats=-1,
            )

    def test_moved_calls_do_not_chain_densities_or_move_cold_setup(self) -> None:
        target = [["H", [0.01, 0.0, 0.0]], ["H", [0.0, 0.02, 1.0]]]
        self.run_population(target)
        self.assertEqual(
            [row["phase"] for row in self.rows],
            ["cold_setup", "moved", "moved", "moved"],
        )
        calls = [event for event in self.events if isinstance(event, tuple)]
        self.assertIsNone(calls[0][1])
        self.assertTrue(
            all(call[1] == [row[1] for row in target] for call in calls[1:])
        )
        self.assertEqual(self.events.count("freeze"), 1)
        self.assertEqual(self.rows[0]["geometry_bohr"], self.atoms)
        self.assertTrue(all(row["geometry_bohr"] == target for row in self.rows[1:]))

    def test_scientific_failure_retained_and_cleanup_cannot_mask_it(self) -> None:
        self.fail_call = 3
        self.fail_close = True
        with self.assertRaises(ValueError) as caught:
            self.run_population()
        self.assertIs(caught.exception, self.failure)
        self.assertEqual(
            [row["status"] for row in self.rows],
            ["measured", "measured", "failed", "cleanup_failed"],
        )
        self.assertEqual(self.events[-1], "close")

    def test_warm_fallback_stops_population_and_retains_failed_observables(
        self,
    ) -> None:
        self.fallback = True
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            self.run_population()
        self.assertEqual(
            len([event for event in self.events if isinstance(event, tuple)]), 2
        )
        self.assertEqual(self.rows[1]["status"], "failed")
        self.assertIn("forces", self.rows[1])
        self.assertEqual(self.events[-1], "close")


class OracleContractTests(unittest.TestCase):
    """Synthetic oracle fixtures test rejection logic, not chemical validity."""

    def setUp(self) -> None:
        self.contract = {
            "method": "pbe0",
            "mode": "direct",
            "basis": "def2-svp",
            "basis_representation": "spherical",
            "charge": 0,
            "spin": 0,
            "geometry_bohr": [["H", [float(i), 0.0, 0.0]] for i in range(48)],
        }
        self.sample = {
            **self.contract,
            "energy": -1.0,
            "converged": True,
            "forces": [[0.0, 0.0, 0.0]] * 48,
        }
        common = {
            **self.contract,
            "status": "complete",
            "allocation": {"test_fixture": True},
            "source_revision": "a" * 40,
            "driver_sha256": "b" * 64,
            "controls": {
                "energy_tolerance": 1e-11,
                "orbital_gradient_tolerance": 1e-7,
                "screening_tolerance": 1e-14,
                "full_density_fock": True,
                "force_grid_response": True,
            },
        }
        self.gpu = {
            **common,
            "backend": "cuda",
            "producer": "gpu4pyscf",
            "package_versions": {"gpu4pyscf": "fixture"},
            "samples": [
                {**self.sample, "level": level, "xc_grid": {"level": level}}
                for level in (4, 5)
            ],
        }
        self.cpu = {
            **common,
            "backend": "cpu",
            "producer": "pyscf",
            "package_versions": {"pyscf": "fixture"},
            "samples": [{**self.sample, "level": 5, "xc_grid": {"level": 5}}],
        }

    def qualify(
        self,
        samples: list[dict] | None = None,
        gpu: dict | None = None,
        cpu: dict | None = None,
    ) -> dict[str, Any]:
        return campaign.qualify_geometry(
            [self.sample] if samples is None else samples,
            self.gpu if gpu is None else gpu,
            self.cpu if cpu is None else cpu,
            self.contract,
        )

    def test_quarter_gates_recomputed_and_all_repeats_checked(self) -> None:
        self.assertEqual(self.qualify()["status"], "PASS")
        changed = {**self.sample, "energy": 0.0, "passed": True}
        self.assertEqual(self.qualify([self.sample, changed])["status"], "FAIL")
        oracle = copy.deepcopy(self.gpu)
        oracle["samples"][0]["energy"] += 48 * 5e-6
        oracle["grid_stability"] = {"passed": True}
        self.assertEqual(self.qualify(gpu=oracle)["status"], "FAIL")

    def test_old_geometry_cannot_qualify_a_displaced_sample(self) -> None:
        sample = copy.deepcopy(self.sample)
        sample["geometry_bohr"][0][1][0] += 0.01
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.qualify([sample])

    def test_independent_cpu_grid_stability_is_recomputed(self) -> None:
        oracle = copy.deepcopy(self.cpu)
        oracle["samples"].insert(
            0, {**copy.deepcopy(self.sample), "level": 4, "xc_grid": {"level": 4}}
        )
        self.assertEqual(self.qualify(cpu=oracle)["status"], "PASS")
        oracle["samples"][0]["energy"] += 48 * 5e-6
        result = self.qualify(cpu=oracle)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["cpu_grid_stability"]["passed"])

    def test_missing_cpu_bad_controls_and_coarse_grid_fail_closed(self) -> None:
        for changes in (
            {"samples": []},
            {"producer": "standard"},
            {"controls": {}},
            {"driver_sha256": "unknown"},
            {"status": "running"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.qualify(cpu={**self.cpu, **changes})
        oracle = copy.deepcopy(self.gpu)
        oracle["samples"][0]["level"] = 3
        with self.assertRaises(ValueError):
            self.qualify(gpu=oracle)
        with self.assertRaises(ValueError):
            self.qualify([])


class CpuWorkerTests(unittest.TestCase):
    def test_absent_allocation_rejected_before_scientific_import_or_output(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            output = Path(directory) / "not-started.json"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                campaign.run_cpu_oracle(
                    [["H", [0.0, 0.0, 0.0]]] * 48, output, source_revision="a" * 40
                )
            self.assertFalse(output.exists())

    def test_cpu_checkpoint_failure_retention_and_no_duplicate_solves(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "fixture.py"
                source.write_text("# synthetic dependency fixture\n", encoding="utf-8")
                output = root / "oracle.json"
                reference = types.ModuleType("benchmarks.readme_defaults_accuracy")
                reference.__file__ = str(source)
                executed = []

                def solve(
                    method: str,
                    atoms: list,
                    *,
                    backend: str,
                    level: int,
                    calls: list = executed,
                    should_fail: bool = fail,
                    destination: Path = output,
                ) -> dict:
                    calls.append((method, backend, level))
                    if should_fail and level == 5:
                        raise ValueError("oracle did not converge")
                    if level == 5:
                        self.assertEqual(
                            len(json.loads(destination.read_text())["samples"]), 1
                        )
                    return {
                        "energy": -1.0,
                        "forces": [[0.0, 0.0, 0.0]] * len(atoms),
                        "converged": True,
                        "level": level,
                        "xc_grid": {"level": level},
                        "controls": {
                            "energy": 1e-11,
                            "orbital_gradient": 1e-7,
                            "screening": 1e-14,
                            "full_density_fock": True,
                            "force_grid_response": True,
                        },
                    }

                reference.solve = solve
                pyscf = types.ModuleType("pyscf")
                pyscf.__version__ = "fixture"
                dft = types.ModuleType("pyscf.dft")
                dft.libxc = types.SimpleNamespace(libxc_version=lambda: "fixture")
                modules = {
                    "benchmarks.readme_defaults_accuracy": reference,
                    "pyscf": pyscf,
                    "pyscf.dft": dft,
                }
                allocation = types.SimpleNamespace(
                    stdout="JobId=123 JobState=RUNNING UserId=fixture(456) TimeLimit=01:00:00"
                )
                with (
                    mock.patch.dict("sys.modules", modules),
                    mock.patch.dict("os.environ", {"SLURM_JOB_ID": "123"}),
                    mock.patch.object(
                        campaign.os, "getuid", return_value=456, create=True
                    ),
                    mock.patch.object(
                        campaign.subprocess, "run", return_value=allocation
                    ),
                ):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "converge"):
                            campaign.run_cpu_oracle(
                                [["H", [0.0, 0.0, 0.0]]] * 48,
                                output,
                                source_revision="a" * 40,
                            )
                    else:
                        campaign.run_cpu_oracle(
                            [["H", [0.0, 0.0, 0.0]]] * 48,
                            output,
                            source_revision="a" * 40,
                        )
                    with self.assertRaises(FileExistsError):
                        campaign.run_cpu_oracle(
                            [["H", [0.0, 0.0, 0.0]]] * 48,
                            output,
                            source_revision="a" * 40,
                        )
                record = json.loads(output.read_text())
                self.assertEqual(executed, [("pbe0", "cpu", 4), ("pbe0", "cpu", 5)])
                self.assertEqual(record["status"], "failed" if fail else "complete")
                self.assertEqual(len(record["samples"]), 1 if fail else 2)
                self.assertEqual(record["scientific_status"], "INCOMPLETE")


class GpuGuardTests(unittest.TestCase):
    def test_device_query_uses_physical_pci_not_remapped_zero(self) -> None:
        cp = types.ModuleType("cupy")
        cp.cuda = types.SimpleNamespace(
            Device=lambda index: types.SimpleNamespace(pci_bus_id="0000:45:00.0"),
            runtime=types.SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDeviceProperties=lambda index: {"major": 12, "minor": 0},
            ),
        )
        result = types.SimpleNamespace(
            stdout="GPU-fixture, 00000000:45:00.0, RTX fixture, 32607, 580.95.05\n"
        )
        with (
            mock.patch.dict("sys.modules", {"cupy": cp}),
            mock.patch.object(
                campaign.subprocess, "run", return_value=result
            ) as command,
        ):
            self.assertEqual(campaign.gpu_device()["uuid"], "GPU-fixture")
        self.assertEqual(command.call_args.args[0][1:3], ["-i", "0000:45:00.0"])

    def test_unowned_gpu_preflight_never_starts_device_or_output(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            root = Path(directory)
            proof = root / "preflight.json"
            proof.write_text(json.dumps({"status": "verified", "job_id": "other"}))
            output = root / "not-started.json"
            with self.assertRaisesRegex(RuntimeError, "allocation"):
                campaign.run_gpu_worker(
                    [["H", [0.0, 0.0, 0.0]]] * 48,
                    output,
                    preflight=proof,
                    engine="standard",
                    kind="clean",
                    reference_path=root / "absent-reference.json",
                )
            self.assertFalse(output.exists())


class ObserverAdmissionTests(unittest.TestCase):
    """Synthetic JSON records test admission logic, never chemical validity."""

    def case(self, engine: str = "standard") -> tuple[dict, dict, dict, dict]:
        atoms = [["H", [float(i), 0.0, 0.0]] for i in range(48)]
        scientific = {
            "method": "pbe0",
            "mode": "direct",
            "basis": "def2-svp",
            "basis_representation": "spherical",
            "charge": 0,
            "spin": 0,
        }
        gpu = {
            "uuid": "GPU-fixture",
            "pci_bus_id": "00000000:01:00.0",
            "name": "fixture",
            "memory_mib": 32607,
            "driver_version": "fixture",
            "compute_capability": [12, 0],
            "assigned_visibility": "0",
        }
        settings = {
            "energy_tolerance": 1e-9,
            "screening_tolerance": 1e-12,
            "max_iterations": 100,
            "diis_history": 8,
            "initial_guess": "stock",
            "force_grid_response": True,
        }
        work = {
            "iterations": 2,
            "fock_builds": 2,
            "warm_start_used": False,
            "warm_start_fallback": False,
        }
        physical = {
            "components": {
                "nuclear": 0.0,
                "one_electron": -2.0,
                "hartree": 0.5,
                "xc": 0.5,
            },
            "density_change_max": 1e-9,
            "physical_residual_max": 1e-9,
            "electrons": [1.0, 1.0],
        }
        if engine == "standard":
            settings.update(
                density_tolerance=1e-5,
                xc_grid_points=200,
                nonlocal_grid=None,
                nonlocal_grid_inherits_xc=True,
                nonlocal_density_policy=None,
                xc_grid={
                    "radial_points": 3,
                    "angular_polar": 0,
                    "angular_azimuth": 0,
                    "version": 4,
                    "units": "Bohr",
                },
            )
            work["ks_diagnostic"] = {
                **copy.deepcopy(physical),
                "occupations": [1, 1],
                "grid_points": 200,
                "tile_points": 32,
                "ao_order": 1,
                "scf_domain": "fixture",
                "initial_density_used": False,
                "fock_builds": 2,
                "history": [
                    {
                        **copy.deepcopy(physical),
                        "iteration": i,
                        "occupation_stabilized": False,
                        "energy_change": None if i == 1 else 0.0,
                    }
                    for i in (1, 2)
                ],
                "coulomb_exchange_work": {
                    "complete": False,
                    "coulomb_applications": None,
                    "exchange_applications": None,
                },
                "nonlocal_work": None,
                "nonlocal_history_scopes": None,
            }
        else:
            settings.update(
                orbital_gradient_tolerance=None,
                orbital_gradient_tolerance_resolved=1e-5,
                direct_scf=True,
                small_rho_cutoff=0.0,
                xc_grid={
                    "level": 3,
                    "points": 200,
                    "radial": "fixture",
                    "angular_pruning": "fixture",
                    "radii_adjust": "fixture",
                    "becke": "fixture",
                },
            )
            work.update(
                fock_builds=None,
                effective_potential_builds=3,
                effective_potential_build_semantics="executed get_veff invocations, including initialization/audits",
                vv10_reuse=None,
                vv10_applicability="not applicable to PBE0",
            )
        sample = {
            "status": "measured",
            "phase": "cold",
            "repeat": 0,
            "seconds": 1.0,
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]] * 48,
            "converged": True,
            "engine": engine,
            "sdk_identity": "c" * 64,
            "gpu_uuid": gpu["uuid"],
            "seed_identity": campaign.identity(
                {"cold_geometry": atoms, "policy": "immutable own cold"}
            ),
            "seed_identity_kind": "declared policy and cold geometry; no cross-engine AO density identity asserted",
            "seed_policy": "stock_guess",
            "geometry_bohr": atoms,
            "geometry_sha256": campaign.identity(atoms),
            "settings": settings,
            "work": work,
        }
        if engine == "standard":
            sample["initial_guess_diagnostic"] = {
                "kind": "huckel",
                "outcome": "used",
                "work_counters_complete": True,
                "preliminary_iterations": 1,
                "preliminary_fock_builds": 1,
                "target_attempts": 1,
                "discarded_target_iterations": 0,
                "discarded_target_fock_builds": 0,
                "preparation_numeric_capacity": 1000,
                "preparation_seconds": 0.1,
            }
            base = {
                "policy_version": 1,
                "requested_mode": "fp64",
                "effective_bits": 64,
                "mixed_precision_fock_threshold": 0.0,
                "strict_refinement_applied": False,
                "mixed_precision_reserved_error": 0.0,
                "refinement_iterations": 0,
                "mixed_stage_fock_builds": 0,
                "strict_stage_fock_builds": 2,
                "post_scf_fock_builds": 0,
                "execution_retries": 0,
                "mixed_admission_census": 0,
                "final_residual_audits": 1,
                "skipped_final_fock_builds": 0,
                "operator_work_counters_valid": True,
            }
            sample["precision"] = {
                **base,
                "native_provenance": copy.deepcopy(base),
                "detail_version": 1,
                "complete": True,
                "operator_inventory_complete": True,
                "conversion_count": 0,
                "fallback_count": 0,
                "returned_state_identity": "cuda-ks:1:1:2",
                "scf_fock_timeline": [
                    {
                        "kind": "strict_fock",
                        "sequence": 0,
                        "iteration": 2,
                        "state": "cuda-ks:1:1:2",
                        "phase": "scf",
                        "count": 1,
                    },
                    {
                        "kind": "final_audit",
                        "sequence": 1,
                        "iteration": 2,
                        "state": "cuda-ks:1:1:2",
                        "phase": "finalization",
                        "count": 1,
                    },
                ],
                "operators": [
                    {
                        "name": name,
                        "storage": "fp64",
                        "compute": "fp64",
                        "accumulation": "fp64",
                        "reduction": "fp64",
                        "arithmetic_mode": "strict",
                        "count": 2,
                    }
                    for name in (
                        "coulomb_j",
                        "exchange_k",
                        "xc",
                        "fock_assembly",
                        "physical_residual",
                    )
                ],
            }
        common = {
            **scientific,
            "engine": engine,
            "source_revision": "a" * 40,
            "driver_sha256": "b" * 64,
            "sdk_identity": "c" * 64,
            "native_source_identity": "d" * 64,
            "preflight_sha256": "e" * 64,
        }
        expected = {
            **common,
            "gpu": copy.deepcopy(gpu),
            "geometry_bohr": copy.deepcopy(atoms),
            "settings": copy.deepcopy(settings),
        }
        clean = {
            **common,
            "schema": "generativeqc.direct-phase-moved.gpu.v1",
            "status": "complete",
            "population": "clean",
            "gpu": gpu,
            "geometry_bohr": atoms,
            "geometry_sha256": campaign.identity(atoms),
            "cold_geometry_bohr": atoms,
            "timing_role": "qualification_pilot",
            "samples": [sample],
        }
        observed = copy.deepcopy(clean)
        observed["population"] = "instrumented"
        reference = {
            **scientific,
            "schema": "generativeqc.direct-phase-moved.oracle.v1",
            "status": "complete",
            "geometry_bohr": copy.deepcopy(atoms),
            "geometry_sha256": campaign.identity(atoms),
            "producer": "pyscf",
            "backend": "cpu",
            "source_revision": "a" * 40,
            "driver_sha256": "f" * 64,
            "reference_helper_sha256": "1" * 64,
            "allocation": "synthetic fixture",
            "package_versions": {
                "pyscf": "fixture",
                "numpy": "fixture",
                "libxc": "fixture",
            },
            "samples": [
                {
                    "energy": -1.0,
                    "forces": [[0.0, 0.0, 0.0]] * 48,
                    "converged": True,
                    "level": 5,
                    "xc_grid": {"level": 5, "points": 300},
                    "controls": {
                        "energy": 1e-11,
                        "orbital_gradient": 1e-7,
                        "screening": 1e-14,
                        "full_density_fock": True,
                        "force_grid_response": True,
                    },
                }
            ],
        }
        return clean, observed, reference, expected

    def refused(
        self, clean: dict, observed: dict, reference: dict, expected: dict
    ) -> None:
        try:
            result = campaign.admit_observer_pair(
                clean, observed, reference, expected=expected
            )
        except ValueError:
            return
        self.assertFalse(result["admitted"])

    def test_roundoff_and_numeric_history_are_not_work_but_bitwise_fail_is_retained(
        self,
    ) -> None:
        for engine in ("standard", "gpu4pyscf"):
            with self.subTest(engine=engine):
                a, b, ref, expected = self.case(engine)
                b["samples"][0]["forces"][0] = [1e-13, 0.0, 0.0]
                if engine == "standard":
                    b["samples"][0]["work"]["ks_diagnostic"]["history"][0][
                        "components"
                    ]["hartree"] += 1e-13
                    b["samples"][0]["initial_guess_diagnostic"][
                        "preparation_seconds"
                    ] = 0.2
                original = copy.deepcopy((a, b, ref, expected))
                result = campaign.admit_observer_pair(a, b, ref, expected=expected)
                self.assertTrue(result["admitted"])
                self.assertFalse(
                    result["samples"][0]["original_bitwise_diagnostic"]["passed"]
                )
                self.assertEqual(
                    result["full_primitive_work_neutrality"], "UNAVAILABLE"
                )
                self.assertEqual(result["actual_D0_byte_identity"], "UNAVAILABLE")
                self.assertEqual((a, b, ref, expected), original)

    def test_both_missing_required_is_never_verified(self) -> None:
        paths = [
            ("source_revision",),
            ("gpu", "uuid"),
            ("preflight_sha256",),
            ("samples", 0, "sdk_identity"),
            ("samples", 0, "seed_identity"),
            ("samples", 0, "settings", "energy_tolerance"),
            ("samples", 0, "settings", "xc_grid"),
            ("samples", 0, "work", "fock_builds"),
            ("samples", 0, "work", "ks_diagnostic", "grid_points"),
            (
                "samples",
                0,
                "work",
                "ks_diagnostic",
                "history",
                0,
                "occupation_stabilized",
            ),
            ("samples", 0, "initial_guess_diagnostic", "preliminary_fock_builds"),
            ("samples", 0, "precision", "operators"),
            (
                "samples",
                0,
                "precision",
                "native_provenance",
                "strict_stage_fock_builds",
            ),
        ]
        for path in paths:
            with self.subTest(path=path):
                a, b, ref, expected = self.case()
                for record in (a, b):
                    parent = record
                    for key in path[:-1]:
                        parent = parent[key]
                    del parent[path[-1]]
                self.refused(a, b, ref, expected)

    def test_changed_required_counters_are_rejected(self) -> None:
        for engine in ("standard", "gpu4pyscf"):
            a, b, ref, expected = self.case(engine)
            key = (
                "fock_builds" if engine == "standard" else "effective_potential_builds"
            )
            b["samples"][0]["work"][key] += 1
            self.refused(a, b, ref, expected)
        a, b, ref, expected = self.case()
        b["samples"][0]["precision"]["operators"][0]["count"] += 1
        self.refused(a, b, ref, expected)

    def test_changed_history_branch_length_and_initial_guess_are_rejected(self) -> None:
        for change in ("branch", "length", "guess"):
            a, b, ref, expected = self.case()
            if change == "branch":
                b["samples"][0]["work"]["ks_diagnostic"]["history"][0][
                    "occupation_stabilized"
                ] = True
            elif change == "length":
                b["samples"][0]["work"]["ks_diagnostic"]["history"].pop()
            else:
                b["samples"][0]["initial_guess_diagnostic"]["outcome"] = (
                    "target_retried"
                )
            self.refused(a, b, ref, expected)

    def test_settings_null_missing_or_changed_are_rejected(self) -> None:
        for value in (None, 1e-8):
            a, b, ref, expected = self.case()
            b["samples"][0]["settings"]["energy_tolerance"] = value
            self.refused(a, b, ref, expected)
        a, b, ref, expected = self.case()
        for record in (a, b):
            record["samples"][0]["settings"]["energy_tolerance"] = None
        expected["settings"]["energy_tolerance"] = None
        self.refused(a, b, ref, expected)

    def test_source_gpu_or_driver_identity_cannot_drift(self) -> None:
        for key, value in (
            ("source_revision", "2" * 40),
            ("driver_sha256", "2" * 64),
            ("sdk_identity", "2" * 64),
            ("native_source_identity", "2" * 64),
        ):
            a, b, ref, expected = self.case()
            for record in (a, b):
                record[key] = value
            self.refused(a, b, ref, expected)
        a, b, ref, expected = self.case()
        b["gpu"]["uuid"] = "GPU-other"
        self.refused(a, b, ref, expected)

    def test_unknown_schema_unclassified_work_and_boolean_counter_are_rejected(
        self,
    ) -> None:
        for change in ("schema", "extra", "bool", "precision"):
            a, b, ref, expected = self.case()
            for record in (a, b):
                if change == "schema":
                    record["schema"] = "unknown.v99"
                elif change == "extra":
                    record["samples"][0]["work"]["unknown_counter"] = 1
                elif change == "bool":
                    record["samples"][0]["work"]["iterations"] = True
                else:
                    record["samples"][0]["precision"]["detail_version"] = 99
            self.refused(a, b, ref, expected)
        a, b, ref, expected = self.case()
        ref["schema"] = "unknown-oracle.v99"
        self.refused(a, b, ref, expected)

    def test_nonfinite_values_and_incomplete_force_shapes_are_rejected(self) -> None:
        for change in ("energy", "force", "history", "shape", "string", "bool"):
            a, b, ref, expected = self.case()
            row = b["samples"][0]
            if change == "energy":
                row["energy"] = float("nan")
            elif change == "force":
                row["forces"][0] = [float("inf"), 0.0, 0.0]
            elif change == "history":
                row["work"]["ks_diagnostic"]["history"][0]["components"]["hartree"] = (
                    float("nan")
                )
            elif change == "shape":
                row["forces"] = [[0.0, 0.0, 0.0]]
            else:
                row["forces"][0] = ["0" if change == "string" else False, 0.0, 0.0]
            self.refused(a, b, ref, expected)

    def test_each_existing_quarter_gate_is_enforced_without_new_tolerances(
        self,
    ) -> None:
        for change in ("energy", "force_max", "force_rms"):
            a, b, ref, expected = self.case()
            row = b["samples"][0]
            if change == "energy":
                row["energy"] += 48 * 3e-6
            elif change == "force_max":
                row["forces"][0] = [4e-5, 0.0, 0.0]
            else:
                row["forces"] = [[8e-6] * 3 for _ in range(48)]
            self.assertTrue(
                campaign.observable_errors(row, ref["samples"][-1], 48)["passed"]
            )
            self.refused(a, b, ref, expected)

    def test_pair_self_agreement_does_not_replace_independent_scientific_gates(
        self,
    ) -> None:
        a, b, ref, expected = self.case()
        for record in (a, b):
            record["samples"][0]["forces"] = [[4e-5] * 3 for _ in range(48)]
            record["samples"][0]["independent_cpu_errors"] = {"passed": True}
        self.refused(a, b, ref, expected)

    def test_nullable_coverage_is_preserved_as_unknown_never_zero(self) -> None:
        a, b, ref, expected = self.case()
        result = campaign.admit_observer_pair(a, b, ref, expected=expected)
        self.assertTrue(result["admitted"])
        self.assertIsNone(
            a["samples"][0]["work"]["ks_diagnostic"]["coulomb_exchange_work"][
                "coulomb_applications"
            ]
        )
        for record in (a, b):
            record["samples"][0]["work"]["ks_diagnostic"]["coulomb_exchange_work"][
                "coulomb_applications"
            ] = 0
        self.refused(a, b, ref, expected)


if __name__ == "__main__":
    unittest.main()
