"""Python-free native molecular task admission, results and failure receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
HOST = os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN", "1") == "1"


class SaddleCli(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="gqc-native-saddle-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.xyz = self.root / "Input With Spaces.xyz"
        self.xyz.write_text("2\nBohr\nH 0 0 0\nH 0 0 1.5\n")
        self.masses = self.root / "Masses With Spaces.json"
        self.masses.write_text("[1.00782503223, 1.00782503223]")

    def call(
        self,
        *arguments: str,
        masses: bool = True,
        timeout: int = 60,
        journal: Path | None = None,
    ) -> subprocess.CompletedProcess:
        command = [str(CLI), "saddle", str(self.xyz), "--units", "bohr", "--json"]
        if masses:
            command += ["--masses", str(self.masses)]
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=dict(
                os.environ,
                PATH="/nonexistent",
                PYTHONHOME="/nonexistent",
                PYTHONPATH="/nonexistent",
                CXX="/nonexistent",
                CC="/nonexistent",
                **({"GENERATIVEQC_DF_PROGRESS_TRACE": str(journal)} if journal else {}),
            ),
        )

    def scf_journal(self, path: Path, spin: str) -> list[dict]:
        """Check completed scopes without prescribing internal evaluation counts."""
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        starts = {row["id"]: row for row in rows if row["event"] == "BEGIN"}
        ends = {row["id"]: row for row in rows if row["event"] == "END"}
        self.assertTrue(starts)
        self.assertEqual(starts.keys(), ends.keys())
        for identity, start in starts.items():
            self.assertEqual(ends[identity]["name"], start["name"])
            self.assertGreaterEqual(ends[identity]["time_ns"], start["time_ns"])
        scopes = [row for row in starts.values() if row["name"] == f"{spin}_scf"]
        self.assertTrue(scopes)
        self.assertTrue(
            any(row["name"] == f"{spin}_physical_evaluation" for row in starts.values())
        )
        observations = []
        for scope in scopes:
            values = {
                row["key"]: row["value"]
                for row in rows
                if row["id"] == scope["id"] and row["event"] == "VALUE"
            }
            self.assertGreater(values["basis_dimension"], 0)
            self.assertGreater(values["grid_points"], 0)
            self.assertGreaterEqual(
                values["terminal_fock_builds"], values["terminal_iterations"]
            )
            self.assertEqual(ends[scope["id"]]["status"], "returned")
            observations.append(values)
        return observations

    def rejected(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_help(self) -> None:
        result = subprocess.run(
            [str(CLI), "saddle", "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in (
            "--masses",
            "--max-attempts",
            "--occupied-state-budget-bytes",
            "--accepted-frame-warm-start",
            "--cold-fock-damping",
            "--host-bytes",
            "unscaled",
            "no Python/JIT or preoptimization",
        ):
            self.assertIn(text, result.stdout)

    def test_explicit_masses_and_attempt_bound(self) -> None:
        self.rejected(self.call(masses=False), "requires --masses")
        for value in ("0", "101", "-1", "1.5", "18446744073709551616"):
            with self.subTest(value=value):
                self.rejected(
                    self.call("--max-attempts", value), "positive bounded integer"
                )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_invalid_masses(self) -> None:
        for value in ("[true, 1]", "[1]", "[1, -1]", "[1, 0]", '[1, "2"]'):
            with self.subTest(value=value):
                self.masses.write_text(value)
                self.rejected(self.call(), "one positive finite dalton mass")

    def test_no_implicit_backend_or_method_switch(self) -> None:
        for arguments in (
            ("--backend", "cuda"),
            ("--method", "wb97m-v-rks"),
            ("--method", "rhf"),
        ):
            with self.subTest(arguments=arguments):
                self.rejected(
                    self.call(*arguments), "requires a linked native CPU B3LYP"
                )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_unqualified_ecp_is_refused_before_any_molecular_state(self) -> None:
        """An admission-only potential must never reach an all-electron frame."""
        potential = [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [2],
                "gaussian_exponents": ["1"],
                "coefficients": [["0.3"]],
            }
        ]
        record = {
            "schema": "generativeqc.basis",
            "schema_version": 1,
            "name": "synthetic-He-ECP-saddle-refusal",
            "representation": "spherical",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic admission control, not numerical qualification",
                "version": "1",
                "license": "CC0",
                "checksum": "c" * 64,
            },
            "elements": [
                {
                    "atomic_number": 2,
                    "nuclear_charge": 2,
                    "ecp_core_electrons": 1,
                    "ecp_data": json.dumps(
                        potential, sort_keys=True, separators=(",", ":")
                    ),
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
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["checksum"] = hashlib.sha256(payload.encode()).hexdigest()
        basis = self.root / "Unqualified ECP With Spaces.json"
        basis.write_text(json.dumps(record))
        self.xyz.write_text("2\nBohr\nHe 0 0 0\nHe 0 0 1.5\n")
        self.masses.write_text("[4.00260325413, 4.00260325413]")
        for method, charge, multiplicity in (
            ("b3lyp-rks", "0", "1"),
            ("b3lyp-d3(bj)-uks", "1", "2"),
        ):
            with self.subTest(method=method):
                self.rejected(
                    self.call(
                        "--basis",
                        str(basis),
                        "--method",
                        method,
                        "--charge",
                        charge,
                        "--multiplicity",
                        multiplicity,
                        "--max-iterations",
                        "1",
                        "--progress-json",
                    ),
                    "ECP requires separate real-element qualification",
                )
        self.rejected(
            self.call("--basis", str(basis), "--host-bytes", "536870912"),
            "ECP requires separate real-element qualification",
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_def2_ecp_saddle_admission(self) -> None:
        """Check the qualified potentials and controls without a long saddle search."""
        data = Path(__file__).resolve().parents[1] / "data" / "global_hybrid_ecp"
        manifest = json.loads((data / "manifest.json").read_text())
        upstream = (data / manifest["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(upstream).hexdigest(), manifest["sha256"])
        elements = json.loads(upstream)["elements"]
        basis = self.root / "def2-SVP with ECP.json"

        def write_basis(
            representation: str = "spherical", *, altered: bool = False, heavy: int = 55
        ) -> None:
            record = {
                "schema": "generativeqc.basis",
                "schema_version": 1,
                "name": "def2-SVP with ECP",
                "representation": representation,
                "exponent_units": "bohr^-2",
                "normalization": "normalized-primitives; native-normalized-contractions",
                "ordering": "CCA-cartesian; libcint-real-spherical",
                "provenance": {
                    "source": manifest["source_repository"],
                    "version": manifest["source_commit"],
                    "license": manifest["license"],
                    "checksum": manifest["sha256"],
                },
                "elements": [],
            }
            for number in (1, heavy):
                source = elements[str(number)]
                potentials = source.get("ecp_potentials")
                if altered and potentials:
                    potentials = json.loads(json.dumps(potentials))
                    potentials[0]["coefficients"][0][0] = "-28.88"
                record["elements"].append(
                    {
                        "atomic_number": number,
                        "nuclear_charge": number,
                        "ecp_core_electrons": source.get("ecp_electrons", 0),
                        "ecp_data": json.dumps(potentials) if potentials else None,
                        "shells": [
                            {
                                "angular_momentum": shell["angular_momentum"][0],
                                "source_group": group,
                                "exponents": shell["exponents"],
                                "coefficients": shell["coefficients"],
                            }
                            for group, shell in enumerate(source["electron_shells"])
                        ],
                    }
                )
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record["checksum"] = hashlib.sha256(payload.encode()).hexdigest()
            basis.write_text(json.dumps(record))

        self.xyz.write_text("2\nBohr\nCs 0 0 0\nH 0.1 0.2 5.5\n")
        self.masses.write_text("[132.90545196, 1.00782503223]")
        base = ("--basis", str(basis), "--max-iterations", "3")
        warm = ("--accepted-frame-warm-start",)
        damping = ("--cold-fock-damping", "2", "0.5", "1048576")
        d3_rks = ("--method", "b3lyp-d3(bj)-rks")
        write_basis()
        journal = self.root / "admitted.jsonl"
        admitted = self.call(*base, *d3_rks, *warm, *damping, journal=journal)
        self.assertEqual(admitted.returncode, 1, admitted.stderr)
        panel = json.loads(admitted.stdout)
        self.assertFalse(panel["receipt"]["has_valid_state"])
        self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
        self.assertIn("SCF did not converge", panel["derivative_failure"])
        self.assertTrue(self.scf_journal(journal, "rks"))

        def budget_boundary(arguments: tuple[str, ...]) -> None:
            # A small iteration cap yields a preparation receipt without
            # making CI run the real-element Hessian.
            admitted = json.loads(
                self.call(*base, *arguments, "--host-bytes", "1073741824").stdout
            )
            cap = admitted["host_admission"]["minimum_host_peak_bytes"]
            self.assertIn("SCF did not converge", admitted["derivative_failure"])
            exact = json.loads(
                self.call(*base, *arguments, "--host-bytes", str(cap)).stdout
            )
            self.assertEqual(exact["host_admission"]["selected_host_peak_bytes"], cap)
            self.assertIn("SCF did not converge", exact["derivative_failure"])
            short = json.loads(
                self.call(*base, *arguments, "--host-bytes", str(cap - 1)).stdout
            )
            self.assertIn(
                "--host-bytes preparation refused", short["derivative_failure"]
            )
            self.assertEqual(short["work"]["scf_fock_builds"], 0)
            self.assertFalse(
                any(row["stage"] == "scf_forces" for row in short["evaluation_records"])
            )

        budget_boundary((*d3_rks, *warm, *damping))

        def refused_before_scf(
            name: str, arguments: tuple[str, ...], message: str
        ) -> None:
            journal = self.root / f"refused-{name}.jsonl"
            self.rejected(self.call(*base, *arguments, journal=journal), message)
            if journal.exists():
                self.assertFalse(
                    any(
                        event["event"] == "BEGIN"
                        and event["name"] in ("rks_scf", "uks_scf")
                        for event in map(json.loads, journal.read_text().splitlines())
                    )
                )

        for name, representation, altered, arguments in (
            ("no-warm", "spherical", False, d3_rks),
            ("no-damping", "spherical", False, (*d3_rks, *warm)),
            ("no-d3", "spherical", False, ("--method", "b3lyp-rks", *warm)),
            ("uks", "spherical", False, ("--method", "b3lyp-d3(bj)-uks", *warm)),
            ("cartesian", "cartesian", False, (*d3_rks, *warm, *damping)),
            ("changed-potential", "spherical", True, (*d3_rks, *warm, *damping)),
        ):
            with self.subTest(name=name):
                write_basis(representation, altered=altered)
                refused_before_scf(name, arguments, "Cs46 def2-ECP")
                refused_before_scf(
                    f"{name}-budgeted",
                    (*arguments, "--host-bytes", "1073741824"),
                    "Cs46 def2-ECP",
                )

        self.xyz.write_text("2\nBohr\nI 0 0 0\nH 0.1 0.2 3.1\n")
        self.masses.write_text("[126.9044719, 1.00782503223]")
        write_basis(heavy=53)
        bounds = ("--initial-radius", "0.015", "--maximum-radius", "0.03")
        continuity = ("--minimum-occupied-overlap", "0.99")
        budget_boundary((*d3_rks, *warm, *bounds, *continuity))
        journal = self.root / "admitted-iodine.jsonl"
        admitted = self.call(
            *base, *d3_rks, *warm, *bounds, *continuity, journal=journal
        )
        self.assertEqual(admitted.returncode, 1, admitted.stderr)
        panel = json.loads(admitted.stdout)
        self.assertFalse(panel["receipt"]["has_valid_state"])
        self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
        self.assertIn("SCF did not converge", panel["derivative_failure"])
        self.assertTrue(self.scf_journal(journal, "rks"))

        for name, arguments in (
            ("default-radii", (*d3_rks, *warm, *continuity)),
            ("default-overlap", (*d3_rks, *warm, *bounds)),
            ("damped", (*d3_rks, *warm, *bounds, *continuity, *damping)),
        ):
            with self.subTest(iodine=name):
                refused_before_scf(name, arguments, "I28 def2-ECP")
        write_basis(heavy=53, altered=True)
        refused_before_scf(
            "changed-iodine", (*d3_rks, *warm, *bounds, *continuity), "I28 def2-ECP"
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_direct_fp64_and_resource_admission(self) -> None:
        self.rejected(self.call("--precision", "fp32"), "--precision must be")
        self.rejected(self.call("--precision", "auto"), "requires direct FP64")
        self.rejected(self.call("--output-budget-bytes", "1"), "result panels exceed")
        self.rejected(
            self.call("--saddle-budget-bytes", "1"), "argument/resource refusal"
        )
        self.rejected(self.call("--max-coordinates", "3"), "exceeds --max-coordinates")
        self.rejected(self.call("--device-budget-bytes", "1"), "does not use CUDA")
        self.rejected(self.call("--host-reserve-bytes", "1"), "require --host-bytes")
        self.rejected(
            self.call(
                "--host-bytes",
                "536870912",
                "--method",
                "b3lyp-d3(bj)-rks",
                "--cold-fock-damping",
                "8",
                "0.5",
                "4096",
            ),
            "--accepted-frame-warm-start",
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_aggregate_host_admission_boundary(self) -> None:
        controls = ("--grid-level", "0", "--occupied-state-budget-bytes", "256")
        baseline = json.loads(self.call(*controls).stdout)
        fitted = json.loads(self.call(*controls, "--host-bytes", "536870912").stdout)
        cap = fitted["host_admission"]["minimum_host_peak_bytes"]
        reserve = 123
        exact = self.call(
            *controls,
            "--host-bytes",
            str(cap + reserve),
            "--host-reserve-bytes",
            str(reserve),
        )
        self.assertEqual(exact.returncode, 1, exact.stderr)  # Nonnegative target.
        panel = json.loads(exact.stdout)
        capacity = panel["host_admission"]
        self.assertEqual(capacity["effective_host_limit_bytes"], cap)
        self.assertEqual(capacity["minimum_host_peak_bytes"], cap)
        self.assertEqual(capacity["selected_host_peak_bytes"], cap)
        self.assertEqual(
            cap,
            max(
                capacity["harmonic_host_peak_bytes"],
                capacity["outside_electronic_lease_bytes"]
                + capacity["electronic_host_peak_bytes"],
            ),
        )
        import numpy as np

        for key in (
            "energy_hartree",
            "gradient_hartree_per_bohr",
            "raw_hessian_hartree_per_bohr2",
        ):
            np.testing.assert_allclose(panel[key], baseline[key], rtol=0, atol=1e-10)
            np.testing.assert_allclose(fitted[key], baseline[key], rtol=0, atol=1e-10)
        self.assertEqual(panel["occupied_state"], baseline["occupied_state"])
        self.assertEqual(fitted["work"], baseline["work"])
        short = self.call(*controls, "--host-bytes", str(cap - 1))
        self.assertEqual(short.returncode, 1, short.stderr)
        refused = json.loads(short.stdout)
        self.assertFalse(refused["receipt"]["has_valid_state"])
        self.assertIn("--host-bytes", refused["derivative_failure"])
        self.assertEqual(refused["work"]["scf_fock_builds"], 0)
        self.assertIsNone(refused["occupied_state"])
        self.assertFalse(
            any(row["stage"] == "scf_forces" for row in refused["evaluation_records"])
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_bounded_cold_fock_damping(self) -> None:
        model = ("--method", "b3lyp-d3(bj)-rks")
        damping = ("--cold-fock-damping", "8", "0.5", "4096")
        self.rejected(self.call(*model, *damping), "--accepted-frame-warm-start")
        result = self.call(*model, *damping, "--accepted-frame-warm-start")
        panel = json.loads(result.stdout)
        self.assertTrue(panel["receipt"]["has_valid_state"], result.stderr)
        initial = next(
            event
            for event in panel["evaluation_records"]
            if event["stage"] == "energy_gradient" and event["status"] == "complete"
        )
        self.assertFalse(initial["accepted_frame_warm_start"])
        self.assertEqual(initial["cold_fock_prefix_iterations"], 8)

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_damped_proposal_is_charged_to_host_lease(self) -> None:
        controls = (
            "--basis",
            "sto-3g",
            "--grid-level",
            "0",
            "--method",
            "b3lyp-d3(bj)-rks",
            "--accepted-frame-warm-start",
            "--host-bytes",
            "536870912",
        )
        base = json.loads(self.call(*controls).stdout)
        damping = ("--cold-fock-damping", "8", "0.5")
        # Two H/STO-3G AOs give one 2x2 FP64 proposal matrix.
        proposal_bytes = 2 * 2 * 8
        fitted = json.loads(self.call(*controls, *damping, str(proposal_bytes)).stdout)
        cap = fitted["host_admission"]["minimum_host_peak_bytes"]
        self.assertEqual(
            cap - base["host_admission"]["minimum_host_peak_bytes"], proposal_bytes
        )
        scopes = [
            row
            for row in fitted["evaluation_records"]
            if row["stage"] == "energy_gradient" and row["status"] == "complete"
        ]
        self.assertTrue(scopes)
        self.assertEqual(scopes[0]["cold_fock_prefix_iterations"], 8)
        exact_controls = (*controls[:-1], str(cap))
        exact = json.loads(
            self.call(*exact_controls, *damping, str(proposal_bytes)).stdout
        )
        self.assertTrue(exact["receipt"]["has_valid_state"])
        for limit, proposal, expected in (
            (cap - 1, proposal_bytes, "--host-bytes preparation refused"),
            (cap, proposal_bytes - 1, "cold RKS Fock proposal exceeds"),
        ):
            with self.subTest(limit=limit, proposal=proposal):
                result = self.call(*controls[:-1], str(limit), *damping, str(proposal))
                self.assertEqual(result.returncode, 1, result.stderr)
                failure = json.loads(result.stdout)
                self.assertFalse(failure["receipt"]["has_valid_state"])
                self.assertIn(expected, failure["derivative_failure"])
                self.assertEqual(failure["work"]["scf_fock_builds"], 0)

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_packed_capacity_is_forwarded_with_cold_damping(self) -> None:
        controls = (
            "--method",
            "b3lyp-d3(bj)-rks",
            "--basis",
            "sto-3g",
            "--representation",
            "spherical",
            "--accepted-frame-warm-start",
            "--cold-fock-damping",
            "8",
            "0.5",
            "4096",
            "--cpu-packed-eri-budget-bytes",
        )
        # Two spherical s orbitals give three unordered AO pairs, hence six values.
        refused = self.call(*controls, "47")
        self.assertEqual(refused.returncode, 1, refused.stderr)
        panel = json.loads(refused.stdout)
        self.assertFalse(panel["receipt"]["has_valid_state"])
        self.assertIn(
            "packed spherical ERI capacity exceeded", panel["derivative_failure"]
        )
        self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])

        admitted = self.call(*controls, "48")
        panel = json.loads(admitted.stdout)
        self.assertTrue(panel["receipt"]["has_valid_state"], admitted.stderr)
        initial = next(
            event
            for event in panel["evaluation_records"]
            if event["stage"] == "energy_gradient" and event["status"] == "complete"
        )
        self.assertEqual(initial["cold_fock_prefix_iterations"], 8)

    def test_no_preoptimization_or_worker_controls(self) -> None:
        for option in ("--stages", "--preopt-basis", "--compiler-python", "--cache"):
            with self.subTest(option=option):
                self.rejected(self.call(option, "/nonexistent"), "option")
        self.rejected(
            self.call("--initial-radius", "0.01", "--minimum-radius", "0.1"),
            "invalid saddle trust",
        )
        self.rejected(
            self.call("--minimum-occupied-overlap", "1.01"), "overlap controls"
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_nonnegative_target_retains_native_state_without_frequencies(self) -> None:
        journal = self.root / "rks-scf.jsonl"
        result = self.call("--diis-history", "16", "--progress-json", journal=journal)
        self.assertEqual(result.returncode, 1, result.stderr)
        panel = json.loads(result.stdout)
        self.assertFalse(panel["converged"])
        self.assertEqual(panel["controls"]["scf_diis_history"], 16)
        self.assertNotIn("occupied_state", panel)
        receipt = panel["receipt"]
        self.assertEqual(receipt["status"], "target_not_negative")
        self.assertTrue(receipt["has_valid_state"])
        self.assertFalse(receipt["stationary_verified"])
        self.assertEqual(receipt["attempts"], 0)
        self.assertEqual(panel["coordinates_bohr"], [[0, 0, 0], [0, 0, 1.5]])
        self.assertEqual(len(panel["raw_hessian_hartree_per_bohr2"]), 6)
        self.assertIsNone(panel["harmonics"])
        completed = next(
            event
            for event in panel["evaluation_records"]
            if event["stage"] == "energy_gradient" and event["status"] == "complete"
        )
        self.assertEqual(len(completed["occupied_counts"]), 1)
        self.assertEqual(len(completed["minimum_occupied_overlaps_by_spin"]), 1)
        self.assertEqual(
            completed["occupied_overlap_coordinate_frame"],
            "mass-centroid-translation-registered",
        )
        self.assertEqual(completed["root_registration_translation_bohr"], [0, 0, 0])
        self.assertGreater(panel["work"]["scf_iterations"], 0)
        self.assertGreaterEqual(
            panel["work"]["scf_fock_builds"], panel["work"]["scf_iterations"]
        )
        observations = self.scf_journal(journal, "rks")
        self.assertTrue(all(row["terminal_converged"] for row in observations))
        self.assertEqual(
            sum(row["terminal_iterations"] for row in observations),
            panel["work"]["scf_iterations"],
        )
        self.assertEqual(
            sum(row["terminal_fock_builds"] for row in observations),
            panel["work"]["scf_fock_builds"],
        )
        diagnostics = panel["diagnostics"]
        self.assertEqual(diagnostics["numerical_runtime"], "native-cpp-cpu")
        self.assertFalse(diagnostics["runtime_compilation"])
        self.assertEqual(diagnostics["preoptimization_stages"], 0)
        events = [json.loads(line) for line in result.stderr.splitlines()]
        observations = [event for event in events if event["stage"] == "optimizer"]
        self.assertTrue(observations)
        self.assertTrue(all(event["status"] == "in_progress" for event in observations))
        for stage in ("system_prepare", "batch_prepare", "scf_forces"):
            started = next(
                event
                for event in events
                if event["stage"] == stage and event["status"] == "start"
            )
            completed = next(
                event
                for event in events
                if event["stage"] == stage and event["status"] == "complete"
            )
            self.assertGreaterEqual(
                completed["elapsed_seconds"], started["elapsed_seconds"]
            )
            self.assertGreaterEqual(completed["phase_seconds"], 0)
        self.assertTrue(any(event["stage"] == "optimizer" for event in events))
        self.assertTrue(
            any(
                event["stage"] == "hessian" and event["status"] == "complete"
                for event in events
            )
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_bounded_occupied_export(self) -> None:
        doublet = ("--method", "b3lyp-uks", "--charge", "1", "--multiplicity", "2")
        for arguments, populations in (((), [1]), (doublet, [2, 1])):
            with self.subTest(populations=populations):
                if arguments == doublet:
                    self.xyz.write_text("2\nBohr\nLi 0 0 0\nH 0 0 3.0\n")
                    self.masses.write_text("[7.0160034366, 1.00782503223]")
                result = self.call(*arguments, "--occupied-state-budget-bytes", "4096")
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertTrue(panel["receipt"]["has_valid_state"])
                self.assertIsNone(panel["occupied_state_export_failure"])
                state = panel["occupied_state"]
                self.assertEqual(
                    state["numeric_bytes"],
                    8 * state["basis_dimension"] * sum(populations),
                )
                self.assertEqual(len(state["channels"]), len(populations))
                for channel, occupied in zip(
                    state["channels"], populations, strict=True
                ):
                    self.assertEqual(channel["occupied_orbitals"], occupied)
                    self.assertEqual(
                        len(channel["coefficients"]), state["basis_dimension"]
                    )
                    for row in channel["coefficients"]:
                        self.assertEqual(len(row), occupied)
                        self.assertTrue(all(map(math.isfinite, row)))
        self.xyz.write_text("2\nBohr\nH 0 0 0\nH 0 0 1.5\n")
        refused = json.loads(self.call("--occupied-state-budget-bytes", "1").stdout)
        self.assertTrue(refused["receipt"]["has_valid_state"])
        self.assertIsNone(refused["occupied_state"])
        self.assertIn(
            "--occupied-state-budget-bytes", refused["occupied_state_export_failure"]
        )
        no_state = json.loads(
            self.call(
                "--occupied-state-budget-bytes", "4096", "--max-map-terms", "1"
            ).stdout
        )
        self.assertFalse(no_state["receipt"]["has_valid_state"])
        self.assertIsNone(no_state["occupied_state"])
        self.assertIsNone(no_state["occupied_state_export_failure"])

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_derivative_refusal_is_not_a_successful_state(self) -> None:
        """A valid SCF cannot substitute for an admitted curvature result."""
        for option, limit in (
            ("--integral-budget-bytes", "1"),
            ("--max-map-terms", "1"),
            ("--numeric-budget-bytes", str(65 << 20)),
            ("--response-budget-bytes", "1"),
        ):
            with self.subTest(option=option):
                result = self.call(option, limit)
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertEqual(panel["receipt"]["status"], "hessian_failed")
                self.assertFalse(panel["receipt"]["has_valid_state"])
                self.assertFalse(panel["converged"])
                self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
                self.assertEqual(panel["coordinates_bohr"], [[0, 0, 0], [0, 0, 1.5]])
                self.assertTrue(panel["derivative_failure"])
                self.assertTrue(
                    any(
                        event["stage"] == "energy_gradient"
                        and event["status"] == "complete"
                        for event in panel["evaluation_records"]
                    )
                )
                self.assertEqual(panel["evaluation_records"][-1]["status"], "failed")
                if option == "--max-map-terms":
                    self.assertIn("integral-map work cap", panel["derivative_failure"])

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_root_and_unconverged_scf_refusals_retain_no_fake_hessian(self) -> None:
        for arguments in (("--root-budget-bytes", "1"), ("--max-iterations", "1")):
            with self.subTest(arguments=arguments):
                journal = self.root / f"failure-{arguments[0]}.jsonl"
                result = self.call(*arguments, journal=journal)
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertEqual(panel["receipt"]["status"], "evaluation_failed")
                self.assertFalse(panel["receipt"]["has_valid_state"])
                self.assertFalse(panel["converged"])
                self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
                self.assertTrue(panel["derivative_failure"])
                if arguments[0] == "--max-iterations":
                    self.assertIn("SCF did not converge", panel["derivative_failure"])
                    observations = self.scf_journal(journal, "rks")
                    self.assertFalse(
                        any(row["terminal_converged"] for row in observations)
                    )
                    self.assertTrue(
                        all(row["terminal_iterations"] <= 1 for row in observations)
                    )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_uks_native_corrected_total(self) -> None:
        for method in ("b3lyp-uks", "b3lyp-d3(bj)-uks"):
            with self.subTest(method=method):
                journal = self.root / f"{method}.jsonl"
                result = self.call("--method", method, journal=journal)
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertEqual(panel["receipt"]["status"], "target_not_negative")
                self.assertEqual(panel["method"], method)
                self.assertEqual(panel["multiplicity"], 1)
                observations = self.scf_journal(journal, "uks")
                self.assertTrue(all(row["terminal_converged"] for row in observations))
                self.assertEqual(
                    sum(row["terminal_iterations"] for row in observations),
                    panel["work"]["scf_iterations"],
                )
                self.assertEqual(
                    sum(row["terminal_fock_builds"] for row in observations),
                    panel["work"]["scf_fock_builds"],
                )
                event = next(
                    event
                    for event in panel["evaluation_records"]
                    if event["stage"] == "energy_gradient"
                    and event["status"] == "complete"
                )
                self.assertEqual(event["occupied_counts"], [1, 1])
                self.assertEqual(event["minimum_occupied_overlaps_by_spin"], [1, 1])
                if "d3" in method:
                    self.assertGreater(panel["work"]["d3_pair_visits"], 0)
                else:
                    self.assertEqual(panel["work"]["d3_pair_visits"], 0)

    @unittest.skipUnless(
        HOST and os.environ.get("GENERATIVEQC_TEST_SADDLE_ENDPOINT") == "1",
        "opt-in complete NH3 endpoint, not a fast CLI contract test",
    )
    def test_complete_nh3_endpoint_and_independent_mass_weighting(self) -> None:
        for warm in (False, True):
            with self.subTest(accepted_frame_warm_start=warm):
                self.check_complete_nh3_endpoint_and_independent_mass_weighting(warm)

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_host_lease_covers_moved_frames_and_preserves_accepted_root(self) -> None:
        for warm in (False, True):
            with self.subTest(warm=warm):
                self.check_complete_nh3_endpoint_and_independent_mass_weighting(
                    warm, host=True
                )

    def check_complete_nh3_endpoint_and_independent_mass_weighting(
        self, warm: bool, *, host: bool = False
    ) -> None:
        import numpy as np
        from generativeqc.vibrations import analyze_harmonic_modes

        specification = json.loads(
            (
                Path(__file__).resolve().parents[1] / "data/native_saddle/manifest.json"
            ).read_text()
        )
        radius = specification["cases"]["saddle"]["hydrogen_radius_bohr"]
        height = specification["cases"]["saddle"]["nitrogen_z_bohr"]
        off_axis = radius * math.sqrt(3) / 2
        self.xyz.write_text(
            f"4\nBohr\nN 0 0 {height}\nH {radius} 0 0\n"
            f"H {-radius / 2} {off_axis} 0\nH {-radius / 2} {-off_axis} 0\n"
        )
        self.masses.write_text(json.dumps(specification["masses_dalton"]))
        controls = (
            "--representation",
            "spherical",
            "--occupied-state-budget-bytes",
            "4096",
            "--response-rtol",
            "1e-11",
            "--raw-symmetry-tolerance",
            str(specification["controls"]["raw_symmetry_tolerance"]),
            *(("--accepted-frame-warm-start",) if warm else ()),
        )
        if host:
            controls += ("--grid-level", "0", "--host-bytes", "536870912")
        result = self.call(*controls, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr)
        panel = json.loads(result.stdout)
        self.assertTrue(panel["converged"])
        self.assertIsNone(panel["occupied_state_export_failure"])
        self.assertEqual(panel["occupied_state"]["channels"][0]["occupied_orbitals"], 5)
        energy = [
            entry
            for entry in panel["evaluation_records"]
            if entry["stage"] == "energy_gradient" and entry["status"] == "complete"
        ]
        hessians = [
            entry
            for entry in panel["evaluation_records"]
            if entry["stage"] == "hessian" and entry["status"] == "complete"
        ]
        self.assertGreater(len(energy), 2)
        if host:
            admission = panel["host_admission"]
            if warm:
                self.assertGreater(admission["warm_seed_snapshot_bytes_per_frame"], 0)
                self.assertEqual(
                    {row["root_seed_snapshot_bytes"] for row in hessians},
                    {admission["warm_seed_snapshot_bytes_per_frame"]},
                )
            else:
                self.assertEqual(admission["warm_seed_snapshot_bytes_per_frame"], 0)
            self.assertLessEqual(
                admission["selected_host_peak_bytes"],
                admission["effective_host_limit_bytes"],
            )
            self.assertNotEqual(
                energy[0]["coordinates_bohr"], energy[1]["coordinates_bohr"]
            )
            self.assertGreater(energy[1]["root_cross_overlap_workspace_bytes"], 0)
            # The independent root cap can still reject a moved trial before
            # SCF; neither its batch nor its orbitals may be published.
            exact_root = energy[0]["root_numeric_reservation_bytes"]
            rollback = self.call(*controls, "--root-budget-bytes", str(exact_root))
            self.assertEqual(rollback.returncode, 1, rollback.stderr)
            failed = json.loads(rollback.stdout)
            self.assertTrue(failed["receipt"]["has_valid_state"])
            self.assertEqual(failed["coordinates_bohr"], energy[0]["coordinates_bohr"])
            self.assertEqual(failed["energy_hartree"], energy[0]["energy_hartree"])
            self.assertIsNotNone(failed["occupied_state"])
            self.assertIsNone(failed["harmonics"])
            self.assertIn(
                "occupied-root tracking exceeds", failed["derivative_failure"]
            )
            self.assertEqual(
                failed["work"]["scf_fock_builds"], energy[0]["scf_fock_builds"]
            )
        self.assertEqual(
            [entry["accepted_frame_warm_start"] for entry in energy],
            [False] + [warm] * (len(energy) - 1),
        )
        self.assertTrue(hessians)
        self.assertTrue(
            all((entry["root_seed_snapshot_bytes"] > 0) == warm for entry in hessians)
        )
        if warm and not host:
            # Admit the initial root, but not its simultaneous retention with a trial.
            # Derive this capacity from the actual endpoint's numeric reservations.
            budget = energy[0]["root_numeric_reservation_bytes"]
            self.assertLess(budget, energy[1]["root_numeric_reservation_bytes"])
            refused = self.call(
                "--representation",
                "spherical",
                "--accepted-frame-warm-start",
                "--root-budget-bytes",
                str(budget),
                "--occupied-state-budget-bytes",
                "4096",
                "--response-rtol",
                "1e-11",
                timeout=900,
            )
            self.assertEqual(refused.returncode, 1, refused.stderr)
            failure = json.loads(refused.stdout)
            self.assertFalse(failure["converged"])
            self.assertTrue(failure["receipt"]["has_valid_state"])
            self.assertIsNone(failure["harmonics"])
            self.assertIsNotNone(failure["occupied_state"])
            self.assertIn(
                "occupied-root tracking exceeds", failure["derivative_failure"]
            )
            self.assertEqual(failure["coordinates_bohr"], energy[0]["coordinates_bohr"])
        receipt = panel["receipt"]
        self.assertTrue(receipt["stationary_verified"])
        self.assertEqual(receipt["negative_curvatures"], 1)
        self.assertLessEqual(receipt["attempts"], 100)
        self.assertLessEqual(receipt["accepted_steps"], receipt["attempts"])
        for field in (
            "gradient_rms",
            "gradient_max",
            "energy_change",
            "displacement_rms_angstrom",
            "displacement_max_angstrom",
        ):
            self.assertLessEqual(receipt[field], specification["controls"][field])
        matrix = np.asarray(panel["raw_hessian_hartree_per_bohr2"])
        modes = analyze_harmonic_modes(
            matrix,
            coordinates=panel["coordinates_bohr"],
            masses=panel["masses_dalton"],
            hessian_units="hartree/bohr^2",
            coordinate_units="bohr",
            mass_units="dalton",
            symmetrize=True,
            raw_symmetry_tolerance=specification["controls"]["raw_symmetry_tolerance"],
        )
        self.assertEqual(modes.negative_mode_count, 1)
        self.assertFalse(panel["harmonics"]["scaled"])
        np.testing.assert_allclose(
            panel["harmonics"]["frequencies_cm1"],
            modes.frequencies_cm1,
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(panel["harmonics"]["rigid_mode_count"], modes.rigid_mode_count)
        for name in ("scf_iterations", "scf_fock_builds"):
            completed = (
                event
                for event in panel["evaluation_records"]
                if event["stage"] == "energy_gradient" and event["status"] == "complete"
            )
            self.assertEqual(
                panel["work"][name], sum(event[name] for event in completed)
            )


if __name__ == "__main__":
    unittest.main()
