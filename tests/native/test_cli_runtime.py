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
            **os.environ,
            "PATH": str(self.root / "no-interpreters"),
            "GENERATIVEQC_PROFILE_CACHE": str(self.root / "cache"),
            "GENERATIVEQC_PROFILE": "off",
        }

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

    def test_scalar_ecp_analytic_energy_forces_and_metadata_validation(self) -> None:
        sys.path.insert(0, str(SOURCE / "python"))
        from generativeqc.basis import (
            BasisProvenance,
            BasisSet,
            BasisShell,
            ElementBasis,
        )

        potential = [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [2],
                "gaussian_exponents": ["1"],
                "coefficients": [["0"]],
            }
        ]
        element = ElementBasis(
            atomic_number=2,
            shells=(BasisShell(0, ("1",), (("1",),)),),
            ecp_core_electrons=1,
            ecp_data=canonical(potential),
        )
        record = BasisSet(
            name="analytic-He-ECP",
            elements=(element,),
            provenance=BasisProvenance("synthetic", "1", "CC0", "c" * 64),
        )
        path = self.root / "ecp.json"
        path.write_text(canonical(record.to_payload()))
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
        corrupted = record.to_payload()
        corrupted["elements"][0]["ecp_data"] = (
            '[{"ecp_type":"scalar_ecp","ecp_type":"scalar_ecp"}]'
        )
        corrupted.pop("checksum")
        corrupted["checksum"] = digest(corrupted)
        path.write_text(canonical(corrupted))
        self.assertIn("duplicate JSON key", self.cli("resources", *flags).stderr)


if __name__ == "__main__":
    unittest.main()
