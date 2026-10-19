"""Python-free runtime contracts; all GPU identities/evidence here are synthetic."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
PROBE = Path(sys.argv.pop(1)).resolve()
PROFILE_PROBE = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(__file__).resolve().parents[2]
INCREMENTAL_SELECTORS = (
    "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
    "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK",
)
KS_INVENTORY_METHODS = ("lda-rks", "lda-uks", "pbe-rks", "pbe-uks")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gqc-native-runtime-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = {
            **{
                name: value
                for name, value in os.environ.items()
                if not name.startswith(INCREMENTAL_SELECTORS)
            },
            "PATH": str(self.root / "no-interpreters"),
            "GENERATIVEQC_PROFILE_CACHE": str(self.root / "cache"),
            "GENERATIVEQC_PROFILE": "off",
        }
        self.env.pop("GQC_TEST_CAPTURE_SCF", None)

    def cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CLI), *arguments],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def resources(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self.cli(
            "resources",
            str(SOURCE / "tests/data/h2_bohr.xyz"),
            "--units",
            "bohr",
            *arguments,
        )

    def bundle(self) -> Path:
        directory = self.root / "bundle"
        directory.mkdir()
        schedule = {
            "kind": "shell_task",
            "block_threads": 128,
            "unicode": "雪🐈",
            "float": 1.0e-7,
        }
        device = {
            "name": "synthetic-only",
            "major": 12,
            "minor": 0,
            "runtime_version": 12090,
            "driver_version": 13000,
            "release_build": 1,
            "fast_compile": 0,
        }
        kernel = {
            "shell_class": "ssss",
            "consumer": "fock",
            "schedule": schedule,
            "schedule_hash": digest(schedule),
            "source_hash": "b" * 64,
            "gates": dict.fromkeys(
                ("target", "resources", "numerical", "performance", "endpoint"), "pass"
            ),
        }
        evidence = {
            "endpoint": {"passed": True},
            "candidates": [
                {
                    "shell_class": "ssss",
                    "consumer": "fock",
                    "accepted": True,
                    "selected_trial": "fixture",
                    "tuning": {
                        "winners": [
                            {
                                "trial_key": "fixture",
                                "schedule": schedule,
                                "production_validation": {"accepted": True},
                            }
                        ]
                    },
                    "isolated": {
                        "passed": True,
                        "source_hash": "b" * 64,
                        "schedule_hash": digest(schedule),
                        "device": {
                            "name": device["name"],
                            "major": 12,
                            "minor": 0,
                            "runtime": 12090,
                            "driver": 13000,
                        },
                        "errors": {"fixture": {"passed": True}},
                    },
                    "endpoint": {"passed": True},
                }
            ],
        }
        (directory / "libgenerativeqc.so").write_bytes(
            b"synthetic artifact; not executable"
        )
        (directory / "manifest.json").write_text("{}")
        (directory / "evidence.json").write_text(json.dumps(evidence))
        profile = {
            "schema": "generativeqc.local_profile",
            "schema_version": 1,
            "identity": {
                "schema": 1,
                "source_identity": "a" * 64,
                "policy": {
                    "precision": "fp64",
                    "spin": ["rhf", "uhf"],
                    "consumers": ["fock", "force"],
                },
                "device": device,
            },
            "toolchain": {"nvcc": "synthetic", "ptxas": "synthetic"},
            "kernels": [kernel],
            "artifacts": {},
        }
        self.write_profile(directory, profile)
        return directory

    @staticmethod
    def write_profile(directory: Path, profile: dict) -> None:
        profile["artifacts"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("libgenerativeqc.so", "manifest.json", "evidence.json")
        }
        (directory / "profile.json").write_text(json.dumps(profile))

    def test_run_and_optimize_scf_controls_reach_native_preparation(self) -> None:
        """Capture public descriptors, then exit before any scientific execution."""
        if (
            sys.platform != "linux"
            or os.environ.get("GENERATIVEQC_TEST_MONOLITHIC") == "1"
        ):
            self.skipTest("SCF capture requires the Linux shared-core CLI")
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        self.env["GQC_TEST_CAPTURE_SCF"] = "1"
        for command in ("run", "optimize"):
            for method in (
                "rhf",
                "uhf",
                "pbe-rks",
                "pbe-uks",
                "mp2",
                "ump2",
                "rccsd",
                "gfn2-xtb",
            ):
                correlated = method in ("mp2", "ump2", "rccsd")
                energy, density = (1e-12, 1e-10) if correlated else (1e-10, 1e-8)
                screening = 0.0 if correlated or method == "gfn2-xtb" else 1e-14
                cases = (
                    ((), energy, density, screening, 100),
                    (("--energy-tolerance", "3e-11"), 3e-11, density, screening, 100),
                    (("--density-tolerance", "7e-9"), energy, 7e-9, screening, 100),
                    (
                        (
                            "--energy-tolerance",
                            "3e-11",
                            "--density-tolerance",
                            "7e-9",
                            "--max-iterations",
                            "17",
                        ),
                        3e-11,
                        7e-9,
                        screening,
                        17,
                    ),
                )
                for (
                    arguments,
                    expected_energy,
                    expected_density,
                    expected_screening,
                    iterations,
                ) in cases:
                    with self.subTest(
                        command=command, method=method, arguments=arguments
                    ):
                        run = self.cli(
                            command,
                            str(SOURCE / "tests/data/h2_bohr.xyz"),
                            "--units",
                            "bohr",
                            "--method",
                            method,
                            "--backend",
                            "cpu",
                            *(("--stages", method) if command == "optimize" else ()),
                            *arguments,
                        )
                        self.assertEqual(run.returncode, 86, run.stdout + run.stderr)
                        self.assertEqual(
                            json.loads(run.stderr),
                            {
                                "energy_tolerance": expected_energy,
                                "density_tolerance": expected_density,
                                "screening_tolerance": expected_screening,
                                "max_iterations": iterations,
                            },
                        )

    def test_resources_scf_controls_and_cuda_admission(self) -> None:
        """Check dry-run controls and inputs to both CUDA HF admission paths."""
        if (
            sys.platform != "linux"
            or os.environ.get("GENERATIVEQC_TEST_MONOLITHIC") == "1"
        ):
            self.skipTest("SCF capture requires the Linux shared-core CLI")
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        self.env["GQC_TEST_CAPTURE_SCF"] = "1"
        cases = (
            ((), 1e-10, 1e-8),
            (("--energy-tolerance", "3e-11"), 3e-11, 1e-8),
            (("--density-tolerance", "7e-9"), 1e-10, 7e-9),
            (
                ("--energy-tolerance", "3e-11", "--density-tolerance", "7e-9"),
                3e-11,
                7e-9,
            ),
        )
        for method in ("rhf", "uhf", *KS_INVENTORY_METHODS):
            backends = (
                ("cpu", "cuda", "cuda-df") if method in ("rhf", "uhf") else ("cpu",)
            )
            for backend in backends:
                for arguments, energy, density in cases:
                    with self.subTest(
                        method=method, backend=backend, arguments=arguments
                    ):
                        run = self.resources(
                            "--method",
                            method,
                            "--backend",
                            "cpu" if backend == "cpu" else "cuda",
                            *(
                                ("--density-fitting", "cuda")
                                if backend == "cuda-df"
                                else ()
                            ),
                            *arguments,
                        )
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        controls = json.loads(run.stdout)["controls"]
                        self.assertEqual(controls["energy_tolerance"], energy)
                        self.assertEqual(controls["density_tolerance"], density)
                        self.assertEqual(controls["screening_tolerance"], 1e-14)
                        self.assertEqual(controls["max_iterations"], 100)
                        if backend != "cpu":
                            self.assertEqual(
                                json.loads(run.stderr),
                                {
                                    "energy_tolerance": energy,
                                    "screening_tolerance": 1e-14,
                                },
                            )

    def test_canonical_json_matches_python(self) -> None:
        generator = random.Random(20261011)
        numbers = [
            0.0,
            -0.0,
            1e-4,
            1e-5,
            1e15,
            1e16,
            1e-7,
            5e-324,
            sys.float_info.max,
            9007199254740991.0,
        ]
        for _ in range(1000):
            value = math.ldexp(generator.random(), generator.randrange(-1074, 1024))
            if math.isfinite(value):
                numbers.append(value)
        values = [
            {"snow": "雪🐈", "value": value, "nested": [True, None, 7]}
            for value in numbers
        ]
        run = subprocess.run(
            [str(PROBE)],
            input="\n".join(map(json.dumps, values)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        rows = run.stdout.splitlines()
        self.assertEqual(len(rows), 2 * len(values))
        for index, value in enumerate(values):
            self.assertEqual(rows[2 * index], canonical(value))
            self.assertEqual(rows[2 * index + 1], digest(value))

    def test_hf_phase_inventory_and_fleet_scaling(self) -> None:
        one = self.resources()
        self.assertEqual(one.returncode, 0, one.stderr)
        plan = json.loads(one.stdout)
        self.assertEqual(
            plan["item_inventory"],
            {
                "preparation": 60360,
                "iteration": 5312,
                "finalization": 5312,
                "integral_state": 1392,
            },
        )
        self.assertEqual(plan["peak_bytes"]["host"], 62152)
        many = json.loads(self.resources("--batch", "4").stdout)
        self.assertEqual(many["peak_bytes"]["host"], 60360 + 4 * 1792)
        self.assertEqual(many["resident_bytes"]["pageable"], 4 * 1792)
        payload = copy.deepcopy(plan)
        identity = payload.pop("identity")
        self.assertEqual(identity, digest(payload))

    def test_cpu_hf_cli_uses_shared_native_host_inventory(self) -> None:
        """The installed CLI and Python bridge use the same phase capacities."""
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_hf_cpu_host_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        for method in ("rhf", "uhf"):
            for fitted in (False, True):
                with self.subTest(method=method, fitted=fitted):
                    arguments = ["--method", method]
                    if fitted:
                        arguments += ["--density-fitting", "cpu"]
                    run = self.resources(*arguments)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    plan = json.loads(run.stdout)
                    auxiliary = plan.get("auxiliary") or {}
                    shape = (ctypes.c_uint64 * 10)(
                        2,
                        2,
                        2,
                        1 if method == "rhf" else 2,
                        0,
                        8,
                        auxiliary.get("nbf", 0),
                        auxiliary.get("cartesian_nbf", 0),
                        auxiliary.get("maximum_angular", 0),
                        0,
                    )
                    values = (ctypes.c_uint64 * 4)()
                    self.assertEqual(query(shape, len(shape), values, len(values)), 0)
                    self.assertEqual(
                        plan["item_inventory"],
                        dict(
                            zip(
                                (
                                    "preparation",
                                    "iteration",
                                    "finalization",
                                    "integral_state",
                                ),
                                values,
                                strict=True,
                            )
                        ),
                    )

    def test_cpu_hf_cli_fleet_uses_shared_native_aggregation(self) -> None:
        """Homogeneous CLI batches and native ragged descriptors have equal bounds."""
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_hf_cpu_fleet_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        for method in ("rhf", "uhf"):
            for fitted in (False, True):
                with self.subTest(method=method, fitted=fitted):
                    arguments = ["--method", method, "--batch", "3"]
                    if fitted:
                        arguments += ["--density-fitting", "cpu"]
                    run = self.resources(*arguments)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    plan = json.loads(run.stdout)
                    orbital, auxiliary = plan["orbital"], plan.get("auxiliary") or {}
                    item = (
                        2,
                        orbital["nbf"],
                        orbital["cartesian_nbf"],
                        1 if method == "rhf" else 2,
                        orbital["maximum_angular"],
                        8,
                        auxiliary.get("nbf", 0),
                        auxiliary.get("cartesian_nbf", 0),
                        auxiliary.get("maximum_angular", 0),
                        0,
                        orbital["shells"],
                        orbital["primitives"],
                        auxiliary.get("shells", 0),
                        auxiliary.get("primitives", 0),
                    )
                    shapes = (ctypes.c_uint64 * 42)(*(item * 3))
                    values = (ctypes.c_uint64 * 15)()
                    self.assertEqual(query(shapes, len(shapes), values, len(values)), 0)
                    self.assertEqual(values[0], plan["resident_bytes"]["pageable"])
                    self.assertEqual(values[2], plan["peak_bytes"]["host"])
                    self.assertEqual(values[1], values[2] - values[0])
                    self.assertEqual(list(values[3:7]), list(values[7:11]))
                    self.assertEqual(list(values[7:11]), list(values[11:15]))

    def test_budget_reserve_headroom_and_zero(self) -> None:
        for arguments in (
            ("--host-bytes", "0"),
            ("--host-bytes", "62151"),
            ("--host-bytes", "62152", "--host-reserve-bytes", "1"),
            ("--host-bytes", "62152", "--headroom-fraction", "0.01"),
        ):
            result = self.resources(*arguments)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "infeasible")
        self.assertEqual(self.resources("--host-bytes", "62152").returncode, 0)
        self.assertEqual(
            self.resources(
                "--pinned-host-bytes", "0", "--device-bytes", "0"
            ).returncode,
            0,
        )

    def test_shared_composition_owner_preserves_exact_method_limits(self) -> None:
        for method in ("rhf", "uhf", *KS_INVENTORY_METHODS):
            with self.subTest(method=method):
                options = ("--method", method)
                initial = self.resources(*options)
                self.assertEqual(initial.returncode, 0, initial.stderr)
                plan = json.loads(initial.stdout)
                self.assertEqual(
                    plan["composition_inventory"], "native-resource-composition-v1"
                )
                peak = plan["peak_bytes"]["host"]
                for difference in (-1, 0):
                    checked = self.resources(
                        *options, "--host-bytes", str(peak + difference)
                    )
                    self.assertEqual(
                        checked.returncode, 0 if difference == 0 else 2, checked.stderr
                    )
                    self.assertEqual(
                        json.loads(checked.stdout)["peak_bytes"], plan["peak_bytes"]
                    )

    def test_large_budget_headroom_matches_exact_ieee_rational(self) -> None:
        for capacity, fraction in (
            (2**63 - 1, 0.5),
            (2**53 + 1, 0.1),
            (2**63 - 1, 1e-16),
            (2**63 - 1, 5e-324),
        ):
            run = self.resources(
                "--host-bytes", str(capacity), "--headroom-fraction", str(fraction)
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            numerator, denominator = fraction.as_integer_ratio()
            expected = (
                capacity - (capacity * numerator + denominator - 1) // denominator
            )
            self.assertEqual(json.loads(run.stdout)["limits"]["host"], expected)

    def test_cuda_hf_fleet_matches_compiled_abi_and_exact_budgets(self) -> None:
        """Interpose only synthetic GPU leaves; execute the real CLI/ABI owner."""
        if sys.platform != "linux":
            self.skipTest("host-only interposition fixture requires Linux")
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_hf_cuda_fleet_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        for method in ("rhf", "uhf"):
            for history in (0, 8, 64):
                for batch in (1, 3):
                    with self.subTest(method=method, history=history, batch=batch):
                        arguments = (
                            "--backend",
                            "cuda",
                            "--method",
                            method,
                            "--diis-history",
                            str(history),
                            "--batch",
                            str(batch),
                        )
                        run = self.resources(*arguments)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        plan = json.loads(run.stdout)
                        orbital = plan["orbital"]
                        spins = 1 if method == "rhf" else 2
                        effective_history = plan["controls"]["diis_history"]
                        arena = (
                            4096
                            + 128 * orbital["nbf"]
                            + 16 * effective_history
                            + 32 * spins
                        )
                        staging = 256 + 64 * orbital["nbf"]
                        shape = (
                            2,
                            orbital["nbf"],
                            orbital["cartesian_nbf"],
                            orbital["shells"],
                            orbital["primitives"],
                            spins,
                            arena,
                            staging,
                            0,
                        )
                        shapes = (ctypes.c_uint64 * (9 * batch))(*(shape * batch))
                        values = (ctypes.c_uint64 * (4 + 3 * batch))()
                        self.assertEqual(
                            query(shapes, len(shapes), values, len(values)), 0
                        )
                        self.assertEqual(
                            plan["cuda_hf_fleet_composition"],
                            "native-declared-capacity",
                        )
                        self.assertEqual(plan["resident_bytes"]["device:0"], values[0])
                        self.assertEqual(plan["peak_bytes"]["device:0"], values[2])
                        self.assertEqual(plan["peak_bytes"]["host"], values[3])
                        self.assertEqual(
                            plan["item_inventory"],
                            {
                                "arena": values[4],
                                "host": values[5],
                                "ecp_workspace": values[6],
                            },
                        )
                        for space in ("host", "device"):
                            peak = plan["peak_bytes"][
                                "host" if space == "host" else "device:0"
                            ]
                            for difference in (0, -1):
                                checked = self.resources(
                                    *arguments,
                                    f"--{space}-bytes",
                                    str(peak + difference),
                                )
                                self.assertEqual(
                                    checked.returncode,
                                    0 if difference == 0 else 2,
                                    checked.stderr,
                                )
                                self.assertEqual(
                                    json.loads(checked.stdout)["peak_bytes"],
                                    plan["peak_bytes"],
                                )

    def test_cuda_df_native_capacities_match_python_inventory_without_gpu(self) -> None:
        if sys.platform != "linux":
            self.skipTest("host-only interposition fixture requires Linux")
        sys.path.insert(0, str(SOURCE / "python"))
        try:
            from generativeqc.resources_hf_df import cuda_df_candidates
        except ImportError:
            self.skipTest(
                "optional Python resource reference dependencies are unavailable"
            )
        from unittest.mock import patch

        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        orbital = {"nbf": 2, "cartesian_nbf": 2, "shells": 2, "primitives": 6}
        item = {
            "atoms": 2,
            "orbital": orbital,
            "auxiliary": orbital,
            "electrons": {"nalpha": 1, "nbeta": 1},
            "spins": 1,
        }
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        for storage in ("auto", "packed", "packed-single"):
            self.env["GENERATIVEQC_DF_VALUE_STORAGE"] = storage
            with patch.dict(os.environ, {"GENERATIVEQC_DF_VALUE_STORAGE": storage}):
                reference = cuda_df_candidates(
                    library,
                    [item] * 4,
                    diis_history=8,
                    device=0,
                    first_phase=0,
                    last_phase=0,
                )
            result = self.resources(
                "--backend", "cuda", "--density-fitting", "cuda", "--batch", "4"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(len(plan["candidates"]), len(reference))
            for actual, expected in zip(plan["candidates"], reference, strict=True):
                self.assertEqual(actual["name"], expected.name)
                self.assertEqual(actual["retained_device"], expected.estimates[0].bytes)
                self.assertEqual(
                    actual["workspace_device"], expected.estimates[1].bytes
                )
                self.assertEqual(actual["retained_host"], expected.estimates[2].bytes)
                self.assertEqual(actual["workspace_host"], expected.estimates[3].bytes)
                self.assertEqual(
                    actual["runtime_allowance_device"], expected.estimates[4].bytes
                )
            rejected = self.resources(
                "--backend",
                "cuda",
                "--density-fitting",
                "cuda",
                "--batch",
                "4",
                "--host-bytes",
                "0",
                "--device-bytes",
                "0",
            )
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            diagnostic = json.loads(rejected.stdout)
            least_excess = min(
                enumerate(reference),
                key=lambda entry: (
                    sum(estimate.bytes for estimate in entry[1].estimates),
                    entry[0],
                    entry[1].name,
                ),
            )[1]
            self.assertEqual(diagnostic["status"], "infeasible")
            self.assertEqual(diagnostic["selection"], least_excess.name)

    def test_cuda_df_shared_source_owner_and_byte_caps_without_gpu(self) -> None:
        """Actual CLI/DF composition; only the CUDA availability gate is synthetic."""
        if sys.platform != "linux":
            self.skipTest("host-only interposition fixture requires Linux")
        sys.path.insert(0, str(SOURCE / "python"))
        try:
            from generativeqc.resources_hf_df import cuda_df_candidates
        except ImportError:
            self.skipTest("optional Python inventory dependencies are unavailable")
        from unittest.mock import patch

        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        orbital = {"nbf": 2, "cartesian_nbf": 2, "shells": 2, "primitives": 6}
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        for storage in ("auto", "dense", "packed", "packed-single"):
            for exchange in ("auto", "occupied"):
                controls = {
                    "GENERATIVEQC_DF_VALUE_STORAGE": storage,
                    "GENERATIVEQC_DF_EXCHANGE": exchange,
                }
                self.env.update(controls)
                for method, spins in (("rhf", 1), ("uhf", 2)):
                    item = {
                        "atoms": 2,
                        "orbital": orbital,
                        "auxiliary": orbital,
                        "electrons": {"nalpha": 1, "nbeta": 1},
                        "spins": spins,
                    }
                    for batch in (1, 3):
                        with self.subTest(
                            storage=storage,
                            exchange=exchange,
                            method=method,
                            batch=batch,
                        ):
                            with patch.dict(os.environ, controls):
                                reference = cuda_df_candidates(
                                    library,
                                    [item] * batch,
                                    diis_history=8,
                                    device=0,
                                    first_phase=0,
                                    last_phase=0,
                                    requested_budget=64 << 20,
                                )
                            arguments = (
                                "--backend",
                                "cuda",
                                "--density-fitting",
                                "cuda",
                                "--method",
                                method,
                                "--batch",
                                str(batch),
                                "--density-fitting-memory-bytes",
                                str(64 << 20),
                            )
                            result = self.resources(*arguments)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            plan = json.loads(result.stdout)
                            self.assertEqual(len(reference), 1)
                            self.assertEqual(len(plan["candidates"]), 1)
                            actual, expected = plan["candidates"][0], reference[0]
                            self.assertEqual(actual["name"], expected.name)
                            for key, estimate in zip(
                                (
                                    "retained_device",
                                    "workspace_device",
                                    "retained_host",
                                    "workspace_host",
                                    "runtime_allowance_device",
                                ),
                                expected.estimates,
                                strict=True,
                            ):
                                self.assertEqual(actual[key], estimate.bytes)
                            for space in ("host", "device"):
                                peak = plan["peak_bytes"][space]
                                for difference in (-1, 0):
                                    checked = self.resources(
                                        *arguments,
                                        f"--{space}-bytes",
                                        str(peak + difference),
                                    )
                                    self.assertEqual(
                                        checked.returncode,
                                        0 if difference == 0 else 2,
                                        checked.stderr,
                                    )
                                    self.assertEqual(
                                        json.loads(checked.stdout)["peak_bytes"],
                                        plan["peak_bytes"],
                                    )

    def test_df_and_ks_have_real_inventories(self) -> None:
        fitted = self.resources(
            "--density-fitting", "cpu", "--auxiliary-basis", "def2-svp"
        )
        self.assertEqual(fitted.returncode, 0, fitted.stderr)
        self.assertGreater(json.loads(fitted.stdout)["peak_bytes"]["host"], 62152)
        for method in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"):
            run = self.resources("--method", method)
            self.assertEqual(run.returncode, 0, run.stderr)
            plan = json.loads(run.stdout)
            self.assertGreater(plan["item_inventory"]["retained"], 1000)
        unsupported = self.resources("--method", "pbe-rks", "--forces")
        self.assertEqual(unsupported.returncode, 2)
        self.assertEqual(json.loads(unsupported.stdout)["status"], "unsupported")

    def test_cpu_ks_cli_uses_shared_native_fleet_inventory(self) -> None:
        """Homogeneous repeats retain each owner but share the phase workspace."""
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_ks_cpu_fleet_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        for method in KS_INVENTORY_METHODS:
            with self.subTest(method=method):
                run = self.resources(
                    "--method", method, "--batch", "3", "--max-iterations", "250"
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                plan = json.loads(run.stdout)
                radial, polar, azimuth = plan["controls"]["grid"]
                item = (
                    2,
                    2,
                    2,
                    2,
                    6,
                    2 if method.endswith("uks") else 1,
                    0,
                    0,
                    2 * radial * polar * azimuth,
                    radial,
                    polar,
                    256,
                    8,
                    250,
                    int(method.startswith("pbe")),
                    0,
                    0,
                )
                shapes = (ctypes.c_uint64 * 51)(*(item * 3))
                values = (ctypes.c_uint64 * 63)()
                self.assertEqual(query(shapes, len(shapes), 0, values, len(values)), 0)
                self.assertEqual(plan["resident_bytes"]["pageable"], values[12])
                self.assertEqual(plan["peak_bytes"]["host"], values[14])
                self.assertEqual(plan["item_inventory"]["retained"] * 3, values[12])
                for budget, status in ((values[14], 0), (values[14] - 1, 2)):
                    bounded = self.resources(
                        "--method",
                        method,
                        "--batch",
                        "3",
                        "--max-iterations",
                        "250",
                        "--host-bytes",
                        str(budget),
                    )
                    self.assertEqual(bounded.returncode, status, bounded.stderr)

    def test_cuda_ks_cli_uses_shared_native_host_fleet_without_gpu(self) -> None:
        """Synthetic device leaves retain both real native fleet owners."""
        if sys.platform != "linux":
            self.skipTest("host-only interposition fixture requires Linux")
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_ks_cuda_host_fleet_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        device_query = library.generativeqc_resource_ks_cuda_fleet_v1
        device_query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        device_query.restype = ctypes.c_int
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        for method in KS_INVENTORY_METHODS:
            for batch in (1, 3):
                with self.subTest(method=method, batch=batch):
                    arguments = (
                        "--backend",
                        "cuda",
                        "--method",
                        method,
                        "--batch",
                        str(batch),
                        "--max-iterations",
                        "250",
                    )
                    run = self.resources(*arguments)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    plan = json.loads(run.stdout)
                    radial, polar, azimuth = plan["controls"]["grid"]
                    item = (
                        2,
                        2,
                        2,
                        2,
                        6,
                        2 if method.endswith("uks") else 1,
                        0,
                        0,
                        2 * radial * polar * azimuth,
                        radial,
                        polar,
                        256,
                        8,
                        250,
                        int(method.startswith("pbe")),
                        0,
                        0,
                        0,
                    )
                    shapes = (ctypes.c_uint64 * (18 * batch))(*(item * batch))
                    values = (ctypes.c_uint64 * (15 + 16 * batch))()
                    self.assertEqual(
                        query(shapes, len(shapes), 0, 0, values, len(values)), 0
                    )
                    self.assertEqual(
                        plan["host_fleet_composition"], "native-declared-capacity"
                    )
                    self.assertEqual(plan["resident_bytes"]["pageable"], values[12])
                    self.assertEqual(plan["peak_bytes"]["host"], values[14])
                    self.assertEqual(
                        plan["item_inventory"]["retained"] * batch, values[12]
                    )
                    self.assertEqual(
                        plan["item_inventory"]["matrix_binding_host"], 16 << 10
                    )
                    self.assertEqual(
                        plan["item_inventory"]["structured_binding_host"], 32 << 10
                    )
                    self.assertEqual(plan["item_inventory"]["ao_grid_cache"], 0)
                    device_item = (2, 2, 2, 6, 11, 13, 17, 19, 23, 0, 1, 0, 0)
                    device_shapes = (ctypes.c_uint64 * (13 * batch))(
                        *(device_item * batch)
                    )
                    device_values = (ctypes.c_uint64 * (9 + 6 * batch))()
                    self.assertEqual(
                        device_query(
                            device_shapes,
                            len(device_shapes),
                            device_values,
                            len(device_values),
                        ),
                        0,
                    )
                    self.assertEqual(
                        plan["device_fleet_composition"], "native-declared-capacity"
                    )
                    self.assertEqual(plan["peak_bytes"]["device"], device_values[7])
                    self.assertEqual(
                        plan["resident_bytes"]["device:0"],
                        device_values[7] - device_values[4],
                    )
                    self.assertEqual(
                        plan["device_inventory"]["setup"], device_values[12]
                    )
                    for capacity, status in (
                        (device_values[7], 0),
                        (device_values[7] - 1, 2),
                    ):
                        bounded = self.resources(
                            *arguments, "--device-bytes", str(capacity)
                        )
                        self.assertEqual(bounded.returncode, status, bounded.stderr)
                    peak = values[14]
                    for capacity, status in ((peak, 0), (peak - 1, 2)):
                        bounded = self.resources(
                            *arguments, "--host-bytes", str(capacity)
                        )
                        self.assertEqual(bounded.returncode, status, bounded.stderr)
                    allowance = 384 << 20
                    self.assertEqual(
                        query(shapes, len(shapes), allowance, 0, values, len(values)), 0
                    )
                    self.assertEqual(values[14], peak + allowance)
                    self.assertEqual(
                        values[12], plan["resident_bytes"]["pageable"] + allowance
                    )

    def test_cpu_ks_cli_uses_shared_native_host_inventory(self) -> None:
        """The Python-free command exposes each authoritative host owner."""
        library = ctypes.CDLL(str(CLI.parent / "libgenerativeqc.so"))
        query = library.generativeqc_resource_ks_cpu_host_v1
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        for method in KS_INVENTORY_METHODS:
            with self.subTest(method=method):
                run = self.resources("--method", method)
                self.assertEqual(run.returncode, 0, run.stderr)
                plan = json.loads(run.stdout)
                radial, polar, azimuth = plan["controls"]["grid"]
                shape = (ctypes.c_uint64 * 17)(
                    2,
                    2,
                    2,
                    2,
                    6,
                    2 if method.endswith("uks") else 1,
                    0,
                    0,
                    2 * radial * polar * azimuth,
                    radial,
                    polar,
                    256,
                    8,
                    100,
                    int(method.startswith("pbe")),
                    0,
                    0,
                )
                values = (ctypes.c_uint64 * 16)()
                self.assertEqual(query(shape, 17, values, 16), 0)
                self.assertEqual(plan["item_inventory"]["retained"], values[13])
                self.assertEqual(plan["item_inventory"]["setup_workspace"], values[14])
                self.assertEqual(plan["item_inventory"]["scf_workspace"], values[15])
                self.assertEqual(plan["item_inventory"]["ao_grid_cache"], values[12])
                peak = plan["peak_bytes"]["host"]
                self.assertEqual(
                    self.resources(
                        "--method", method, "--host-bytes", str(peak)
                    ).returncode,
                    0,
                )
                self.assertEqual(
                    self.resources(
                        "--method", method, "--host-bytes", str(peak - 1)
                    ).returncode,
                    2,
                )

    def assert_incremental_inventory_unsupported(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stderr, "")
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "unsupported")
        self.assertIn("incremental Direct-J/K resource inventory", plan["diagnostic"])
        self.assertIn("anchor/delta storage", plan["diagnostic"])
        self.assertIn("closure history", plan["diagnostic"])
        for selector in INCREMENTAL_SELECTORS:
            self.assertIn(selector, plan["diagnostic"])
        self.assertEqual(plan["peak_bytes"], {})
        self.assertEqual(plan["resident_bytes"], {})
        self.assertNotIn("orbital", plan)
        self.assertNotIn("item_inventory", plan)
        identity = plan.pop("identity")
        self.assertEqual(identity, digest(plan))

    def test_ks_incremental_inventory_rejects_either_enabled_selector(self) -> None:
        for method in KS_INVENTORY_METHODS:
            for backend in ("cpu", "cuda"):
                for selector in INCREMENTAL_SELECTORS:
                    for value in ("1", "on"):
                        with self.subTest(
                            method=method,
                            backend=backend,
                            selector=selector,
                            value=value,
                        ):
                            for name in INCREMENTAL_SELECTORS:
                                self.env.pop(name, None)
                            self.env[selector] = value
                            result = self.resources(
                                "--method", method, "--backend", backend
                            )
                            self.assert_incremental_inventory_unsupported(result)

    def test_ks_incremental_inventory_combines_valid_selectors_with_or(self) -> None:
        generic, legacy = INCREMENTAL_SELECTORS
        for first, second in (("off", "on"), ("on", "off"), ("1", "1")):
            with self.subTest(generic=first, legacy=second):
                self.env.update({generic: first, legacy: second})
                self.assert_incremental_inventory_unsupported(
                    self.resources("--method", "pbe-uks")
                )

    def test_ks_incremental_inventory_validates_both_selectors_eagerly(self) -> None:
        for selector in INCREMENTAL_SELECTORS:
            other = next(name for name in INCREMENTAL_SELECTORS if name != selector)
            for value in ("", "false", "true", "ON", "OFF", "1 ", " on", "0.0", "2"):
                for other_value in (None, "on"):
                    with self.subTest(
                        selector=selector, value=value, other_value=other_value
                    ):
                        self.env.pop(other, None)
                        if other_value is not None:
                            self.env[other] = other_value
                        self.env[selector] = value
                        result = self.resources("--method", "pbe-uks")
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual(result.stdout, "")
                        self.assertIn(
                            f"{selector} must be 0/off or 1/on", result.stderr
                        )

    def test_ks_disabled_incremental_controls_preserve_plan_bytes(self) -> None:
        generic, legacy = INCREMENTAL_SELECTORS
        for method in KS_INVENTORY_METHODS:
            for backend in ("cpu", "cuda"):
                for name in INCREMENTAL_SELECTORS:
                    self.env.pop(name, None)
                arguments = ("--method", method, "--backend", backend)
                baseline = self.resources(*arguments)
                if backend == "cpu":
                    self.assertEqual(baseline.returncode, 0, baseline.stderr)
                for first in (None, "0", "off"):
                    for second in (None, "0", "off"):
                        with self.subTest(
                            method=method, backend=backend, generic=first, legacy=second
                        ):
                            for name, value in ((generic, first), (legacy, second)):
                                self.env.pop(name, None)
                                if value is not None:
                                    self.env[name] = value
                            result = self.resources(*arguments)
                            self.assertEqual(result.returncode, baseline.returncode)
                            self.assertEqual(result.stdout, baseline.stdout)
                            self.assertEqual(result.stderr, baseline.stderr)

    def test_ks_incremental_inventory_does_not_parse_tuning_suffixes(self) -> None:
        arguments = ("--method", "pbe-uks")
        baseline = self.resources(*arguments)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        for selector in INCREMENTAL_SELECTORS:
            self.env[f"{selector}_REBUILD_INTERVAL"] = "not-an-unsigned-integer"
            self.env[f"{selector}_DENSITY_RMS_THRESHOLD"] = "nan"
            self.env[selector] = "off"
        inactive = self.resources(*arguments)
        self.assertEqual(inactive.returncode, baseline.returncode)
        self.assertEqual(inactive.stdout, baseline.stdout)
        self.assertEqual(inactive.stderr, baseline.stderr)
        for selector in INCREMENTAL_SELECTORS:
            with self.subTest(selector=selector):
                self.env[selector] = "on"
                self.assert_incremental_inventory_unsupported(
                    self.resources(*arguments)
                )
                self.env[selector] = "off"

    def test_ks_incremental_inventory_rejects_before_basis_and_other_gates(
        self,
    ) -> None:
        self.env[INCREMENTAL_SELECTORS[0]] = "on"
        for backend in ("cpu", "cuda"):
            for extra in (
                (),
                ("--host-bytes", "0", "--max-iterations", "1"),
                ("--precision", "auto"),
                ("--forces",),
                ("--density-fitting", "cpu"),
                ("--basis", str(self.root / "missing-basis.json")),
            ):
                with self.subTest(backend=backend, extra=extra):
                    self.assert_incremental_inventory_unsupported(
                        self.resources(
                            "--method", "pbe-uks", "--backend", backend, *extra
                        )
                    )

    def test_hf_inventory_ignores_ks_incremental_controls(self) -> None:
        for method in ("rhf", "uhf"):
            for selector in INCREMENTAL_SELECTORS:
                self.env.pop(selector, None)
            arguments = ("--method", method)
            baseline = self.resources(*arguments)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            for value in ("on", "1", "false", ""):
                with self.subTest(method=method, value=value):
                    for selector in INCREMENTAL_SELECTORS:
                        self.env[selector] = value
                    result = self.resources(*arguments)
                    self.assertEqual(result.returncode, baseline.returncode)
                    self.assertEqual(result.stdout, baseline.stdout)
                    self.assertEqual(result.stderr, baseline.stderr)

    def test_registered_ks_inventory_uses_native_execution_grid(self) -> None:
        probe = subprocess.run(
            [str(PROBE), "--native-grid"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        native_grid = json.loads(probe.stdout)
        for method in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"):
            for backend in ("cpu", "cuda"):
                with self.subTest(method=method, backend=backend):
                    result = self.resources("--method", method, "--backend", backend)
                    self.assertIn(result.returncode, (0, 2), result.stderr)
                    if backend == "cpu":
                        self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan["controls"]["grid"], native_grid)
                    if method == "pbe-rks" and backend == "cpu":
                        # Eligible restricted GGA retains 32 bytes per point/AO.
                        grid_cache = 32 * 2 * math.prod(native_grid) * 2
                        self.assertGreaterEqual(
                            plan["item_inventory"]["scf_workspace"], grid_cache
                        )

    def test_invalid_resource_controls_fail_before_output(self) -> None:
        for arguments in (
            ("--host-bytes", "-1"),
            ("--device-bytes", str(2**63)),
            ("--headroom-fraction", "nan"),
            ("--headroom-fraction", "1"),
            ("--batch", "0"),
            ("--batch", "100001"),
            ("--multiplicity", "2"),
            ("--diis-history", "-1"),
        ):
            result = self.resources(*arguments)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stdout, "")

    def test_zero_diis_history_uses_the_native_default_in_resource_planning(
        self,
    ) -> None:
        baseline = self.resources()
        default_sentinel = self.resources("--diis-history", "0")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(default_sentinel.stdout, baseline.stdout)
        self.assertEqual(
            json.loads(default_sentinel.stdout)["controls"]["diis_history"], 8
        )

    def test_all_registered_correlated_selectors_reach_native_execution(self) -> None:
        for method, reference in (
            ("mp2", -1.129872195222),
            ("rccsd", -1.137275943783),
            ("rccsd(t)", -1.137275943783),
        ):
            run = self.cli(
                "run",
                str(SOURCE / "tests/data/h2_bohr.xyz"),
                "--units",
                "bohr",
                "--method",
                method,
                "--json",
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertAlmostEqual(
                json.loads(run.stdout)["energy_hartree"], reference, places=9
            )
        invalid = self.cli(
            "run",
            str(SOURCE / "tests/data/h2_bohr.xyz"),
            "--method",
            "mp2",
            "--screening-tolerance",
            "1e-14",
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn(
            "unscreened",
            invalid.stderr.replace("--screening-tolerance 0", "unscreened"),
        )

    def test_bulk_libxc_discovery_aliases_and_cpu_execution(self) -> None:
        sys.path.insert(0, str(SOURCE / "python"))
        from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks

        listing = self.cli("methods", "--json", "--libxc")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        table = {entry["name"]: entry for entry in json.loads(listing.stdout)}
        self.assertEqual(table["libxc:gga_x_apbe"]["aliases"], ["libxc-rks:gga_x_apbe"])
        self.assertFalse(table["libxc-uks:gga_x_apbe"]["cuda"])
        energies = []
        for selector in (
            "libxc:GGA_X_APBE",
            "LIBXC-RKS:GGA_X_APBE",
            "libxc-uks:GGA_X_APBE",
        ):
            result = self.cli(
                "run",
                str(SOURCE / "tests/data/h2_bohr.xyz"),
                "--units",
                "bohr",
                "--method",
                selector,
                "--json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            spin = "polarized" if "uks" in selector else "unpolarized"
            self.assertEqual(
                payload["method_ir_identity"],
                resolve_bulk_ks("GGA_X_APBE", spin=spin).method.identity,
            )
            self.assertTrue(math.isfinite(payload["energy_hartree"]))
            energies.append(payload["energy_hartree"])
        self.assertAlmostEqual(energies[0], energies[1], places=12)
        self.assertAlmostEqual(energies[0], energies[2], places=8)
        rejected = self.cli(
            "run",
            str(SOURCE / "tests/data/h2_bohr.xyz"),
            "--method",
            "libxc:GGA_X_APBE",
            "--backend",
            "cuda",
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("CPU backend", rejected.stderr)

    def test_profile_validate_and_zip_export_preserve_python_identity(self) -> None:
        directory = self.bundle()
        original = json.loads((directory / "profile.json").read_text())
        run = self.cli("profile", "validate", str(directory))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["identity"], digest(original))
        archive = self.root / "export.zip"
        run = self.cli("profile", "export", str(directory), str(archive))
        self.assertEqual(run.returncode, 0, run.stderr)
        with zipfile.ZipFile(archive) as exported:
            self.assertEqual(
                set(exported.namelist()),
                {
                    "profile.json",
                    "libgenerativeqc.so",
                    "manifest.json",
                    "evidence.json",
                },
            )
            for name in exported.namelist():
                self.assertEqual(exported.read(name), (directory / name).read_bytes())

    def test_tampered_ungated_and_foreign_evidence_fail_closed(self) -> None:
        directory = self.bundle()
        original = json.loads((directory / "profile.json").read_text())
        for mutation in ("schedule", "gate", "policy", "source", "array"):
            profile = copy.deepcopy(original)
            if mutation == "schedule":
                profile["kernels"][0]["schedule"]["block_threads"] = 64
            if mutation == "gate":
                profile["kernels"][0]["gates"]["endpoint"] = "fail"
            if mutation == "policy":
                profile["identity"]["policy"]["precision"] = "fp32"
            if mutation == "source":
                profile["kernels"][0]["source_hash"] = "z" * 64
            if mutation == "array":
                profile["kernels"] = {}
            (directory / "profile.json").write_text(json.dumps(profile))
            self.assertNotEqual(
                self.cli("profile", "validate", str(directory)).returncode, 0
            )
        (directory / "profile.json").write_text(json.dumps(original))
        (directory / "libgenerativeqc.so").write_bytes(b"tampered")
        self.assertNotEqual(
            self.cli("profile", "validate", str(directory)).returncode, 0
        )

    def test_artifact_digests_require_exact_strings(self) -> None:
        directory = self.bundle()
        original = json.loads((directory / "profile.json").read_text())
        for digest_value in (None, 0, True, [], {}, "0" * 64):
            with self.subTest(digest=digest_value):
                profile = copy.deepcopy(original)
                profile["artifacts"]["libgenerativeqc.so"] = digest_value
                (directory / "profile.json").write_text(json.dumps(profile))
                result = self.cli("profile", "validate", str(directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_archive_traversal_duplicates_and_failed_install_preserve_index(
        self,
    ) -> None:
        cache = self.root / "cache"
        cache.mkdir()
        active = cache / "active.json"
        active.write_text('{"keep":"previous"}')
        for entries in (
            ("../escape",),
            ("profile.json", "profile.json"),
            ("profile.json",),
        ):
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name in entries:
                    output.writestr(name, "{}")
            result = self.cli("profile", "install", str(archive))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(active.read_text()), {"keep": "previous"})
            self.assertFalse((self.root / "escape").exists())
        self.assertFalse(list(cache.glob(".gqc-profile-*")))

    def test_profile_argument_validation_and_diagnostics(self) -> None:
        for arguments in (
            ("profile", "export"),
            ("profile", "install"),
            ("profile", "validate"),
            ("profile", "diagnose", "extra"),
        ):
            self.assertEqual(self.cli(*arguments).returncode, 2)
        result = self.cli("profile", "diagnose")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rejected", json.loads(result.stdout))

    def test_install_select_reexecute_and_clear_with_host_only_identity_stub(
        self,
    ) -> None:
        if sys.platform != "linux":
            self.skipTest("native loader selection is Linux-specific")
        directory = self.bundle()
        (directory / "libgenerativeqc.so").write_bytes(PROFILE_PROBE.read_bytes())
        profile = json.loads((directory / "profile.json").read_text())
        device = profile["identity"]["device"]
        for field in (
            "warp_size",
            "maximum_threads_per_block",
            "maximum_threads_per_sm",
            "maximum_blocks_per_sm",
            "registers_per_sm",
            "maximum_registers_per_thread",
            "sm_count",
            "shared_memory_per_block",
            "shared_memory_per_block_optin",
            "shared_memory_per_sm",
        ):
            device[field] = 0
        device["toolkit_version"] = 12090
        version = self.cli("--version").stdout
        profile["identity"]["native_abi"] = int(version.split("ABI ")[1].split(")")[0])
        profile["identity"]["host"] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        }
        self.write_profile(directory, profile)
        self.env["LD_PRELOAD"] = str(PROFILE_PROBE)
        self.env.pop("GENERATIVEQC_PROFILE")
        archive = self.root / "mock-profile.zip"
        self.assertEqual(
            self.cli("profile", "export", str(directory), str(archive)).returncode, 0
        )
        installed = self.cli("profile", "install", str(archive))
        self.assertEqual(installed.returncode, 0, installed.stderr)
        destination = Path(installed.stdout.strip())
        self.assertEqual(destination.name, digest(profile))
        active = self.root / "cache/active.json"
        self.assertEqual(
            json.loads(active.read_text()),
            {digest(profile["identity"]): digest(profile)},
        )
        diagnosis = self.cli("profile", "diagnose")
        self.assertEqual(
            json.loads(diagnosis.stdout)["source"], "local", diagnosis.stderr
        )
        replay = self.cli(
            "run",
            str(self.root / "missing.xyz"),
            "--method",
            "rhf",
            "--backend",
            "cuda",
            "--units",
            "bohr",
            "--json",
        )
        # The stub selects an actual binary but cannot execute CUDA in this CPU
        # fixture. Successful loader selection must not recurse or fake results.
        self.assertNotEqual(replay.returncode, 0)
        self.assertNotIn("after native re-execution", replay.stderr)
        self.assertIn("cannot open XYZ", replay.stderr)
        self.assertEqual(replay.stdout, "")
        repeated = self.cli("profile", "install", str(archive))
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(self.cli("profile", "clear").returncode, 0)
        self.assertEqual(json.loads(active.read_text()), {})
        self.assertTrue((destination / "libgenerativeqc.so").exists())

    def test_canonical_local_basis_checksum_representation_and_decimal_expansion(
        self,
    ) -> None:
        basis = {
            "schema": "generativeqc.basis",
            "schema_version": 1,
            "name": "test-STO3G",
            "representation": "spherical",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic test",
                "version": "1",
                "license": "CC0",
                "checksum": "c" * 64,
            },
            "elements": [
                {
                    "atomic_number": 1,
                    "nuclear_charge": 1,
                    "ecp_core_electrons": 0,
                    "ecp_data": None,
                    "shells": [
                        {
                            "angular_momentum": 0,
                            "source_group": 0,
                            "exponents": ["3.425250914", "0.6239137298", "0.168855404"],
                            "coefficients": [
                                ["0.1543289673", "0.5353281423", "0.4446345422"]
                            ],
                        }
                    ],
                }
            ],
        }
        basis["checksum"] = digest(basis)
        path = self.root / 'Mixed Case_"local".json'
        path.write_text(json.dumps(basis))
        run = self.cli(
            "run",
            str(SOURCE / "tests/data/h2_bohr.xyz"),
            "--method",
            "rhf",
            "--basis",
            str(path),
            "--units",
            "bohr",
            "--json",
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["basis"], str(path))
        self.assertEqual(result["basis_identity"], basis["checksum"])
        self.assertEqual(result["representation"], "spherical")
        self.assertAlmostEqual(result["energy_hartree"], -1.116714325063, places=9)
        conflicting = self.resources(
            "--basis", str(path), "--representation", "cartesian"
        )
        self.assertEqual(conflicting.returncode, 2)
        auxiliary = self.resources(
            "--representation",
            "cartesian",
            "--density-fitting",
            "cpu",
            "--auxiliary-basis",
            str(path),
        )
        self.assertEqual(auxiliary.returncode, 0, auxiliary.stderr)
        native_auxiliary = self.cli(
            "run",
            str(SOURCE / "tests/data/h2_bohr.xyz"),
            "--units",
            "bohr",
            "--method",
            "rhf",
            "--representation",
            "cartesian",
            "--density-fitting",
            "cpu",
            "--auxiliary-basis",
            str(path),
            "--json",
        )
        self.assertEqual(native_auxiliary.returncode, 0, native_auxiliary.stderr)
        basis["elements"][0]["shells"][0]["exponents"][0] = "8"
        path.write_text(json.dumps(basis))
        self.assertNotEqual(self.resources("--basis", str(path)).returncode, 0)

    def test_retired_autotune_aliases_are_not_runtime_commands(self) -> None:
        for arguments in (
            ("autotune", "unused.xyz"),
            ("autotune", "--show-profile"),
            ("autotune", "--clear-profile"),
        ):
            result = self.cli(*arguments)
            self.assertEqual(result.returncode, 2)
            self.assertIn("generativeqc-tune", result.stderr)
            self.assertEqual(result.stdout, "")
        # Profile administration remains natively available without Python.
        result = self.cli("profile", "show")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_resources_reject_unqualified_hybrid_and_correlated_domains(
        self,
    ) -> None:
        for method in ("pbe0-rks", "pbe0-uks", "rccsd"):
            result = self.resources("--method", method)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "unsupported")

    def test_scalar_ecp_analytic_energy_forces_and_metadata_validation(self) -> None:
        # Keep this native-only contract runnable without Python packages.
        potential = [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [2],
                "gaussian_exponents": ["1"],
                "coefficients": [["0"]],
            }
        ]
        record = {
            "schema": "generativeqc.basis",
            "schema_version": 1,
            "name": "analytic-He-ECP",
            "representation": "cartesian",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic",
                "version": "1",
                "license": "CC0",
                "checksum": "c" * 64,
            },
            "elements": [
                {
                    "atomic_number": 2,
                    "nuclear_charge": 2,
                    "ecp_core_electrons": 1,
                    "ecp_data": canonical(potential),
                    "shells": [
                        {
                            "angular_momentum": 0,
                            "source_group": 0,
                            "exponents": ["1E0"],
                            "coefficients": [["1E0"]],
                        }
                    ],
                }
            ],
        }
        record["checksum"] = digest(record)
        path = self.root / "ecp.json"
        path.write_text(canonical(record))
        molecule = self.root / "he.xyz"
        molecule.write_text("1\nanalytic ECP\nHe 0 0 0\n")
        flags = (
            str(molecule),
            "--method",
            "uhf",
            "--multiplicity",
            "2",
            "--basis",
            str(path),
        )
        run = self.cli("run", *flags, "--forces", "--json")
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        # Normalized alpha=1 Gaussian, effective charge one, zero local potential.
        self.assertAlmostEqual(
            payload["energy_hartree"], 1.5 - math.sqrt(8 / math.pi), places=10
        )
        self.assertEqual(payload["forces_hartree_per_bohr"], [[0.0, 0.0, 0.0]])
        planned = self.cli("resources", *flags)
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertGreater(json.loads(planned.stdout)["ecp_workspace"], 0)
        corrupted = copy.deepcopy(record)
        corrupted["elements"][0]["ecp_data"] = (
            '[{"ecp_type":"scalar_ecp","ecp_type":"scalar_ecp"}]'
        )
        corrupted.pop("checksum")
        corrupted["checksum"] = digest(corrupted)
        path.write_text(canonical(corrupted))
        self.assertIn("duplicate JSON key", self.cli("resources", *flags).stderr)


if __name__ == "__main__":
    unittest.main()
