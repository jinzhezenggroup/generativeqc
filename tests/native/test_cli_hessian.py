"""Host-only native CLI admission: never start a worker or initialize a GPU.

Independent molecular and real CUDA publication/lifetime gates live in the
Slurm-only stationary probe and compiled CLI tests, not fake worker replies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive

CLI = Path(sys.argv.pop(1)).resolve()


class HessianCli(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="gqc-native-hessian-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.xyz = self.root / "Input With Spaces.xyz"
        self.xyz.write_text("2\nBohr\nH 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
        self.direction = self.root / "Direction With Spaces.json"
        self.direction.write_text(json.dumps([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]]))

    def call(
        self, *arguments: str, operation: str = "hvp", trace: Path | None = None
    ) -> subprocess.CompletedProcess:
        command = [str(CLI), operation, str(self.xyz), "--units", "bohr", "--json"]
        if operation == "hvp":
            command += ["--direction", str(self.direction)]
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=dict(
                os.environ,
                PATH="/nonexistent",
                PYTHONHOME="/nonexistent",
                PYTHONPATH="/nonexistent",
                GENERATIVEQC_DF_PROGRESS_TRACE=str(trace) if trace is not None else "",
            ),
        )

    def rejected(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def range_payload(self) -> dict:
        """Use the live compiler encoder rather than pinning a catalog hash."""
        pbe = resolve_method("PBE", spin="unpolarized")
        omega = Fraction(33, 100)
        return replace(
            pbe,
            primitives=(
                *pbe.primitives,
                RangeSeparatedExchangePrimitive(Fraction(1, 5), omega, "short-range"),
                RangeSeparatedExchangePrimitive(Fraction(13, 20), omega, "long-range"),
            ),
        ).to_payload()

    def test_help_documents_linked_native_execution(self) -> None:
        result = subprocess.run(
            [str(CLI), "hessian", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no Python/JIT", result.stdout)
        self.assertIn("--device-budget-bytes", result.stdout)
        self.assertIn("--method-ir", result.stdout)
        self.assertNotIn("--compiler-python", result.stdout)

    def test_unlinked_cpu_method_does_not_fall_back(self) -> None:
        self.rejected(
            self.call("--method", "pbe-uks"),
            "native CPU Hessian has no exact linked total MethodIR",
        )

    def test_obsolete_worker_controls_cannot_execute_any_interpreter(self) -> None:
        worker = self.root / "Fake Worker"
        marker = self.root / "was-run"
        worker.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        worker.chmod(0o700)
        for option in (
            "--compiler-python",
            "--compiler-library",
            "--cache",
            "--timeout-seconds",
        ):
            with self.subTest(option=option):
                self.rejected(
                    self.call(option, str(worker)),
                    "no Python worker or runtime compilation",
                )
                self.assertFalse(marker.exists())

    def test_invalid_response_and_resource_controls(self) -> None:
        for option, value in (
            ("--response-rtol", "1"),
            ("--response-atol", "nan"),
            ("--response-restart", "0"),
            ("--response-max-iterations", "-1"),
            ("--device-budget-bytes", "0"),
            ("--max-nonlocal-pairs", "18446744073709551616"),
            ("--max-map-terms", "0"),
        ):
            with self.subTest(option=option):
                self.rejected(self.call(option, value), option)

    def test_full_output_budget_fails_before_device_initialization(self) -> None:
        self.rejected(
            self.call(
                "--backend", "cuda", "--output-budget-bytes", "1", operation="hessian"
            ),
            "--output-budget-bytes",
        )

    def test_direction_validation_fails_before_device_initialization(self) -> None:
        for value in ([[1, 2, 3]], [[1, 2], [3, 4]], [["nan", 0, 0], [0, 0, 0]]):
            self.direction.write_text(json.dumps(value))
            self.rejected(self.call("--backend", "cuda"), "finite natoms-by-3")

    def test_only_hvp_accepts_direction(self) -> None:
        self.rejected(
            self.call("--direction", str(self.direction), operation="hessian"),
            "only hvp",
        )

    def test_recycle_groups_require_dense_cpu(self) -> None:
        flags = ("--response-recycle-block-size", "3")
        self.rejected(
            self.call(*flags, "--backend", "cpu"), "requires a dense CPU hessian"
        )
        self.rejected(
            self.call(*flags, "--backend", "cuda", operation="hessian"),
            "requires a dense CPU hessian",
        )

    def test_wrong_method_precision_and_provider_never_touch_cuda(self) -> None:
        self.rejected(
            self.call("--backend", "cuda", "--method", "rhf"),
            "no qualified native CUDA hessian/hvp",
        )
        self.rejected(
            self.call("--backend", "cuda", "--precision", "mixed"),
            "--precision must be fp64 or auto",
        )
        self.rejected(
            self.call("--backend", "cuda", "--density-fitting", "cpu"),
            "CPU density-fitting",
        )
        self.rejected(
            self.call("--backend", "cuda", "--max-map-terms", "1"),
            "does not use CPU",
        )

    def test_cpu_full_matrix_hvp_and_total_composition_without_python(self) -> None:
        """Run the actual binary with no interpreter path, not a worker fixture."""
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        self.xyz.write_text("3\nBohr\nH 0 .1 -1.6\nH .3 -.2 0\nH -.1 .4 1.6\n")
        self.direction.write_text(
            json.dumps([[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]])
        )
        for suffix, multiplicity, charge in (("rks", "1", "1"), ("uks", "2", "0")):
            with self.subTest(spin=suffix):
                controls = (
                    "--backend",
                    "cpu",
                    "--representation",
                    "spherical",
                    "--charge",
                    charge,
                    "--multiplicity",
                    multiplicity,
                    "--grid-radial-points",
                    "8",
                    "--grid-polar-points",
                    "4",
                    "--grid-azimuth-points",
                    "8",
                )
                method = f"b3lyp-d3(bj)-{suffix}"
                matrix_call = self.call(
                    "--method", method, *controls, operation="hessian"
                )
                self.assertEqual(matrix_call.returncode, 0, matrix_call.stderr)
                matrix = json.loads(matrix_call.stdout)
                action_call = self.call("--method", method, *controls)
                self.assertEqual(action_call.returncode, 0, action_call.stderr)
                action = json.loads(action_call.stdout)
                vector = [
                    component
                    for atom in json.loads(self.direction.read_text())
                    for component in atom
                ]
                flattened = [
                    component for atom in action["value"] for component in atom
                ]
                for row, actual in zip(matrix["matrix"], flattened, strict=True):
                    expected = sum(
                        first * second
                        for first, second in zip(row, vector, strict=True)
                    )
                    self.assertTrue(math.isfinite(actual))
                    self.assertAlmostEqual(actual, expected, delta=2e-8)
                self.assertEqual(matrix["backend"], "cpu")
                diagnostic = matrix["diagnostics"]
                self.assertFalse(diagnostic["runtime_compilation"])
                self.assertFalse(diagnostic["posthoc_symmetrization"])
                self.assertLess(diagnostic["raw_symmetry_error"], 1e-7)
                self.assertGreater(diagnostic["d3_pair_visits"], 0)
                self.assertGreater(diagnostic["first_one_electron_primitives"], 0)
                self.assertGreater(diagnostic["curvature_one_electron_primitives"], 0)
                self.assertGreater(diagnostic["source_xc_point_visits"], 0)
                self.assertGreater(diagnostic["curvature_xc_point_visits"], 0)
                self.assertGreater(diagnostic["response_xc_point_visits"], 0)
                self.assertEqual(diagnostic["ecp_kernel_blocks"], 0)
                self.assertEqual(diagnostic["max_eri_primitives"], 100_000_000)
                self.assertEqual(
                    diagnostic["d3_pair_visits"],
                    9 * action["diagnostics"]["d3_pair_visits"],
                )
                bare_call = self.call(
                    "--method", f"b3lyp-{suffix}", *controls, operation="hessian"
                )
                self.assertEqual(bare_call.returncode, 0, bare_call.stderr)
                bare = json.loads(bare_call.stdout)
                self.assertEqual(bare["diagnostics"]["d3_pair_visits"], 0)
                self.assertAlmostEqual(
                    matrix["energy_hartree"] - bare["energy_hartree"],
                    diagnostic["d3_energy"],
                    delta=2e-12,
                )
                self.assertLess(diagnostic["d3_energy"], 0)

                for selected, corrected in ((f"b3lyp-{suffix}", False), (method, True)):
                    flags = ("--method", selected, *controls)
                    for product in ("hvp", "hessian"):
                        with self.subTest(selected=selected, product=product):
                            planned = self.call(
                                *flags, "--property", product, operation="resources"
                            )
                            self.assertEqual(
                                planned.returncode, 0, planned.stderr or planned.stdout
                            )
                            inventory = json.loads(planned.stdout)
                            host_peak = inventory["peak_bytes"]["host"]
                            derivative = inventory["derivative_inventory"]
                            self.assertEqual(
                                "correction_inventory" in inventory, corrected
                            )
                            if corrected:
                                self.assertEqual(
                                    inventory["correction_inventory"][
                                        "method_identity"
                                    ],
                                    derivative["method_identity"],
                                )
                            exact = self.call(
                                *flags,
                                "--host-bytes",
                                str(host_peak),
                                operation=product,
                            )
                            self.assertEqual(exact.returncode, 0, exact.stderr)
                            bounded = json.loads(exact.stdout)
                            receipt = bounded["diagnostics"]
                            self.assertEqual(
                                bounded["method_ir_identity"],
                                derivative["method_identity"],
                            )
                            self.assertEqual(
                                receipt["numeric_reservation_bytes"],
                                derivative["minimum_numeric_bytes"],
                            )
                            self.assertEqual(
                                receipt["resource_admission"][
                                    "selected_host_peak_bytes"
                                ],
                                host_peak,
                            )
                            self.assertEqual(receipt["d3_pair_visits"] > 0, corrected)
                            reference = matrix if corrected else bare
                            if product == "hessian":
                                for actual_row, expected_row in zip(
                                    bounded["matrix"], reference["matrix"], strict=True
                                ):
                                    for actual, expected in zip(
                                        actual_row, expected_row, strict=True
                                    ):
                                        self.assertAlmostEqual(
                                            actual, expected, delta=2e-8
                                        )
                            else:
                                for row, actual in zip(
                                    reference["matrix"],
                                    (
                                        value
                                        for atom in bounded["value"]
                                        for value in atom
                                    ),
                                    strict=True,
                                ):
                                    self.assertAlmostEqual(
                                        actual,
                                        sum(
                                            entry * step
                                            for entry, step in zip(
                                                row, vector, strict=True
                                            )
                                        ),
                                        delta=2e-8,
                                    )
                            self.rejected(
                                self.call(
                                    *flags,
                                    "--host-bytes",
                                    str(host_peak - 1),
                                    operation=product,
                                ),
                                "allocation inventory exceeds host budget",
                            )

    def test_cpu_resource_floor_matches_execution_before_scf(self) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        controls = (
            "--backend",
            "cpu",
            "--basis",
            "sto-3g",
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
            "--grid-budget-bytes",
            str(4 << 20),
            "--response-budget-bytes",
            str(1 << 20),
        )
        for method in ("pbe-rks", "pbe0-d3(bj)-rks"):
            for product in ("hessian", "hvp"):
                with self.subTest(method=method, product=product):
                    flags = ("--method", method, *controls)
                    inventory = ("--property", product, *flags)
                    planned = self.call(*inventory, operation="resources")
                    self.assertEqual(
                        planned.returncode, 0, planned.stderr or planned.stdout
                    )
                    plan = json.loads(planned.stdout)
                    floor = plan["derivative_inventory"]["minimum_numeric_bytes"]
                    host_peak = plan["peak_bytes"]["host"]
                    self.assertGreater(host_peak, floor)
                    self.assertEqual("correction_inventory" in plan, "d3(bj)" in method)

                    for delta, expected in ((0, "feasible"), (-1, "infeasible")):
                        result = self.call(
                            *inventory,
                            "--host-bytes",
                            str(host_peak + 1024 + delta),
                            "--host-reserve-bytes",
                            "1024",
                            "--numeric-budget-bytes",
                            str(floor),
                            operation="resources",
                        )
                        self.assertEqual(
                            result.returncode, 0 if delta == 0 else 2, result.stderr
                        )
                        self.assertEqual(json.loads(result.stdout)["status"], expected)
                    short_inventory = self.call(
                        *inventory,
                        "--numeric-budget-bytes",
                        str(floor - 1),
                        operation="resources",
                    )
                    self.assertEqual(
                        short_inventory.returncode, 2, short_inventory.stderr
                    )
                    self.assertEqual(
                        json.loads(short_inventory.stdout)["status"], "infeasible"
                    )

                    capped = self.call(
                        *flags, "--numeric-budget-bytes", str(floor), operation=product
                    )
                    self.assertEqual(capped.returncode, 0, capped.stderr)
                    diagnostic = json.loads(capped.stdout)["diagnostics"]
                    self.assertEqual(diagnostic["minimum_numeric_bytes"], floor)
                    self.assertEqual(diagnostic["numeric_reservation_bytes"], floor)
                    for slack in (0, 4096):
                        effective = host_peak + slack
                        host_flags = (
                            "--host-bytes",
                            str(2 * (effective + 1024)),
                            "--host-reserve-bytes",
                            "1024",
                            "--headroom-fraction",
                            "0.5",
                        )
                        planned_host = self.call(
                            *inventory, *host_flags, operation="resources"
                        )
                        self.assertEqual(
                            planned_host.returncode, 0, planned_host.stderr
                        )
                        selected = json.loads(planned_host.stdout)[
                            "derivative_inventory"
                        ]
                        bounded = self.call(*flags, *host_flags, operation=product)
                        self.assertEqual(bounded.returncode, 0, bounded.stderr)
                        result = json.loads(bounded.stdout)
                        actual = result["diagnostics"]["resource_admission"]
                        self.assertEqual(actual["minimum_host_peak_bytes"], host_peak)
                        self.assertEqual(
                            actual["effective_host_limit_bytes"], effective
                        )
                        self.assertEqual(actual["selected_host_peak_bytes"], effective)
                        self.assertEqual(
                            actual["derivative_numeric_cap_bytes"], floor + slack
                        )
                        self.assertEqual(
                            selected["selected_host_peak_bytes"], effective
                        )
                        self.assertEqual(
                            selected["selected_numeric_cap_bytes"], floor + slack
                        )
                        self.assertLessEqual(
                            result["diagnostics"]["numeric_reservation_bytes"],
                            floor + slack,
                        )
                        panel = "matrix" if product == "hessian" else "value"
                        reference = json.loads(capped.stdout)[panel]
                        for candidate_row, reference_row in zip(
                            result[panel], reference, strict=True
                        ):
                            for candidate, expected in zip(
                                candidate_row, reference_row, strict=True
                            ):
                                self.assertAlmostEqual(candidate, expected, delta=2e-8)

                    one_step = ("--max-iterations", "1")
                    one_step_plan = self.call(
                        *inventory, *one_step, operation="resources"
                    )
                    self.assertEqual(one_step_plan.returncode, 0, one_step_plan.stderr)
                    one_step_peak = json.loads(one_step_plan.stdout)["peak_bytes"][
                        "host"
                    ]
                    self.rejected(
                        self.call(
                            *flags,
                            *one_step,
                            "--host-bytes",
                            str(one_step_peak - 1),
                            operation=product,
                        ),
                        "allocation inventory exceeds host budget",
                    )
                    # An admitted one-iteration solve cannot converge. A one-byte-short
                    # request must instead fail sizing before entering that same SCF.
                    self.rejected(
                        self.call(*flags, "--max-iterations", "1", operation=product),
                        "SCF did not converge",
                    )
                    self.rejected(
                        self.call(
                            *flags,
                            "--max-iterations",
                            "1",
                            "--numeric-budget-bytes",
                            str(floor - 1),
                            operation=product,
                        ),
                        "CPU Hessian aggregate numerical reservation exceeded",
                    )

    def test_unqualified_derivative_inventory_does_not_probe_cuda(self) -> None:
        result = self.call(
            "--property",
            "hessian",
            "--method",
            "pbe-rks",
            "--backend",
            "cuda",
            operation="resources",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "unsupported")
        self.assertIn("Direct FP64 CPU", plan["diagnostic"])
        self.assertNotIn("derivative_inventory", plan)
        self.rejected(
            self.call(
                "--backend", "cuda", "--method", "pbe-rks", "--host-bytes", "1024"
            ),
            "--host-bytes for Hessian/HVP requires --backend cpu",
        )

    def test_packed_eri_and_host_budgets_admit_exact_fit_before_scf(self) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        self.xyz.write_text("2\nBohr\nHe 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
        common = (
            "--backend",
            "cpu",
            "--basis",
            "sto-3g",
            "--representation",
            "spherical",
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
        )
        # Two AOs have three distinct pairs and six FP64 pair-pair orbits.
        packed_size = 6 * 8
        saving = 2**4 * 8 - packed_size
        for method, spin in (
            ("pbe0-rks", ("--charge", "1")),
            ("b3lyp-d3(bj)-uks", ("--multiplicity", "2")),
        ):
            for product, panel in (("hessian", "matrix"), ("hvp", "value")):
                with self.subTest(method=method, product=product):
                    args = (*common, "--method", method, *spin)
                    query = (*args, "--property", product)
                    packed = ("--cpu-packed-eri-budget-bytes", str(packed_size))
                    dense = self.call(*query, operation="resources")
                    plan = self.call(*query, *packed, operation="resources")
                    self.assertEqual(dense.returncode, 0, dense.stderr or dense.stdout)
                    self.assertEqual(plan.returncode, 0, plan.stderr or plan.stdout)
                    inventory, baseline = (
                        json.loads(plan.stdout),
                        json.loads(dense.stdout),
                    )
                    self.assertEqual(
                        inventory["packed_eri_inventory"],
                        {"retained_bytes": packed_size, "cap_bytes": packed_size},
                    )
                    peak = inventory["peak_bytes"]["host"]
                    self.assertEqual(peak + saving, baseline["peak_bytes"]["host"])
                    self.assertEqual(
                        inventory["item_inventory"]["setup_workspace"],
                        baseline["item_inventory"]["setup_workspace"],
                    )
                    loose = self.call(
                        *query,
                        "--cpu-packed-eri-budget-bytes",
                        str(1 << 20),
                        operation="resources",
                    )
                    self.assertEqual(loose.returncode, 0, loose.stderr)
                    self.assertEqual(
                        json.loads(loose.stdout)["peak_bytes"]["host"], peak
                    )

                    trace = self.root / "packed-refused.jsonl"
                    for provider_cap, host_cap, detail in (
                        (
                            packed_size - 1,
                            peak,
                            "packed spherical ERI capacity exceeded",
                        ),
                        (
                            packed_size,
                            peak - 1,
                            "allocation inventory exceeds host budget",
                        ),
                    ):
                        caps = (
                            "--cpu-packed-eri-budget-bytes",
                            str(provider_cap),
                            "--host-bytes",
                            str(host_cap),
                        )
                        refused = self.call(*query, *caps, operation="resources")
                        self.assertEqual(refused.returncode, 2, refused.stderr)
                        refusal = json.loads(refused.stdout)
                        self.assertEqual(refusal["status"], "infeasible")
                        self.assertIn(detail, refusal["diagnostic"])
                        self.rejected(
                            self.call(*args, *caps, operation=product, trace=trace),
                            detail,
                        )
                        self.assertFalse(
                            trace.exists(), "SCF ran before budget admission"
                        )

                    trace = self.root / f"{method}-{product}.jsonl"
                    bounded = self.call(
                        *args,
                        *packed,
                        "--host-bytes",
                        str(peak),
                        operation=product,
                        trace=trace,
                    )
                    reference = self.call(*args, operation=product)
                    self.assertEqual(bounded.returncode, 0, bounded.stderr)
                    self.assertEqual(reference.returncode, 0, reference.stderr)
                    actual, expected = (
                        json.loads(bounded.stdout),
                        json.loads(reference.stdout),
                    )
                    self.assertEqual(
                        actual["method_ir_identity"], expected["method_ir_identity"]
                    )
                    self.assertEqual(
                        actual["diagnostics"]["resource_admission"][
                            "selected_host_peak_bytes"
                        ],
                        peak,
                    )
                    events = [
                        json.loads(line) for line in trace.read_text().splitlines()
                    ]
                    self.assertEqual(
                        [
                            event["value"]
                            for event in events
                            if event.get("key") == "packed_eri_retained_bytes"
                        ],
                        [packed_size],
                    )
                    for actual_row, expected_row in zip(
                        actual[panel], expected[panel], strict=True
                    ):
                        for value, ref in zip(actual_row, expected_row, strict=True):
                            self.assertAlmostEqual(value, ref, delta=2e-8)

    def test_packed_eri_host_budget_rejects_cartesian_representation(self) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        self.rejected(
            self.call(
                "--backend",
                "cpu",
                "--method",
                "pbe0-rks",
                "--representation",
                "cartesian",
                "--host-bytes",
                str(1 << 30),
                "--cpu-packed-eri-budget-bytes",
                str(1 << 20),
            ),
            "packed Coulomb ERIs require a real spherical s/p/d/f basis",
        )

    def test_cpu_work_refusal_publishes_no_prefix_or_electronic_only_result(
        self,
    ) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        result = self.call(
            "--backend",
            "cpu",
            "--method",
            "b3lyp-d3(bj)-rks",
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
            "--max-d3-pairs",
            "1",
        )
        self.rejected(result, "D3 pair-work cap")

    def test_cpu_map_work_cap_refusal_and_recovery(self) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        options = ("--backend", "cpu", "--method", "b3lyp-rks")
        for option, detail in (
            ("--max-map-terms", "integral-map work cap"),
            ("--max-eri-primitives", "nuclear primitive-work cap"),
        ):
            with self.subTest(option=option):
                refused = self.call(*options, option, "1")
                self.rejected(refused, detail)
                self.assertRegex(refused.stderr, r"planned=[1-9][0-9]+, cap=1\)")
        map_cap = 1 << 39
        trace = self.root / "hvp.jsonl"
        action = self.call(*options, "--max-map-terms", str(map_cap), trace=trace)
        self.assertEqual(action.returncode, 0, action.stderr)
        diagnostic = json.loads(action.stdout)["diagnostics"]
        self.assertEqual(diagnostic["max_map_terms"], map_cap)
        events = [json.loads(line) for line in trace.read_text().splitlines()]

        def observed(stage: str, key: str) -> int:
            """Sum completed work reported by the direction's diagnostic scopes."""
            return sum(
                event["value"]
                for event in events
                if event["name"] == stage and event.get("key") == key
            )

        for phase in ("first", "curvature"):
            self.assertGreater(observed(f"cpu_hessian_{phase}_xc", "map_terms"), 0)
            self.assertEqual(
                observed(f"cpu_hessian_{phase}_integrals", "eri_primitives"),
                diagnostic[f"{phase}_eri_primitives"],
            )
        self.assertEqual(
            observed("cpu_hessian_first_one_electron", "one_electron_primitives"),
            diagnostic["first_one_electron_primitives"],
        )
        self.assertEqual(
            observed("cpu_hessian_first_eri", "eri_primitives"),
            diagnostic["first_eri_primitives"],
        )
        self.assertTrue(
            any(
                event["name"] == "cpu_hessian_cpks"
                and event["event"] == "END"
                and event["status"] == "converged"
                for event in events
            )
        )
        self.assertEqual(
            observed("cpu_hessian_cpks", "response_iterations"),
            diagnostic["response_iterations"],
        )

    def test_cpu_ecp_reaches_bounded_native_owner_without_a_python_worker(self) -> None:
        """Synthetic admission/resource control, not real-element qualification."""
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
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
            "name": "synthetic-He-ECP-admission",
            "representation": "spherical",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic admission control",
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
        basis = self.root / "ECP With Spaces.json"
        basis.write_text(json.dumps(record))
        self.xyz.write_text("1\nBohr\nHe 0 0 0\n")
        self.direction.write_text("[[0.02,-0.03,0.1]]")
        for endpoint in ("hessian", "hvp"):
            with self.subTest(endpoint=endpoint):
                self.rejected(
                    self.call(
                        "--backend",
                        "cpu",
                        "--method",
                        "b3lyp-uks",
                        "--multiplicity",
                        "2",
                        "--basis",
                        str(basis),
                        "--grid-radial-points",
                        "8",
                        "--grid-polar-points",
                        "4",
                        "--grid-azimuth-points",
                        "8",
                        "--numeric-budget-bytes",
                        "1",
                        operation=endpoint,
                    ),
                    "CPU Hessian aggregate numerical reservation exceeded",
                )

    def test_method_ir_is_validated_before_device_initialization(self) -> None:
        path = self.root / "Method IR.json"
        path.write_text(json.dumps(self.range_payload()))
        flags = ("--method-ir", str(path))
        for operation in ("hessian", "hvp"):
            with self.subTest(operation=operation):
                self.rejected(
                    self.call(*flags, "--backend", "cpu", operation=operation),
                    "range-PBE MethodIR is not supported by native CPU hessian/hvp",
                )
        self.rejected(self.call(*flags, "--method", "pbe-rks"), "mutually exclusive")
        self.rejected(self.call(*flags, *flags), "can only be specified once")
        self.rejected(
            self.call(*flags, "--backend", "cuda", "--precision", "auto"),
            "range-PBE MethodIR requires FP64",
        )
        self.rejected(
            self.call(*flags, "--backend", "cuda", "--density-fitting", "auto"),
            "range-PBE MethodIR requires FP64",
        )

    def test_method_ir_semantic_changes_fail_closed(self) -> None:
        path = self.root / "Method IR.json"
        for index, key, value, error in (
            (0, "functional", {}, "does not match the linked unscaled PBE"),
            (1, "omega", "1/2", "one shared omega"),
            (2, "coefficient", "1/5", "distinct FP64"),
            (2, "coefficient", "3/2", "must not exceed one"),
            (2, "coefficient", "2/4", "canonical positive MethodIR fraction"),
            (2, "operator", "short-range", "canonical short-range and long-range"),
            (2, "omega_units", "angstrom^-1", "canonical short-range and long-range"),
        ):
            with self.subTest(index=index, key=key, value=value):
                payload = self.range_payload()
                payload["primitives"][index][key] = value
                path.write_text(json.dumps(payload))
                self.rejected(
                    self.call("--method-ir", str(path), "--backend", "cuda"), error
                )


if __name__ == "__main__":
    unittest.main()
