"""C++ CLI transport/failure gates and complete compiler-AD CPU endpoints.

Fake interpreters test transport only, including synthetic CUDA metadata; they
never execute a GPU. Real CUDA qualification is separately opt-in under Slurm.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
MONOLITHIC = os.environ.get("GENERATIVEQC_TEST_MONOLITHIC") == "1"


class HessianCli(unittest.TestCase):
    """Exercise the actual C++ parser, worker process and publication boundary."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="gqc-hessian-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.xyz = self.root / "Input With Spaces.xyz"
        self.xyz.write_text("2\nBohr\nH 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
        self.direction = [[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]]
        self.direction_file = self.root / "Direction With Spaces.json"
        self.direction_file.write_text(json.dumps(self.direction))
        self.interpreter = self.root / "Interpreter With Spaces"
        self.interpreter.write_text(
            f"#!{sys.executable}\n"
            "import json,sys\n"
            "request=json.load(sys.stdin)\n"
            "operation=request['operation']\n"
            "count=len(request['atoms_bohr'])\n"
            "diagnostics={'derivative_method':'compiler-automatic-differentiation',"
            "'displaced_scf_evaluations':0,'execution_residency':'device-resident',"
            "'numerical_host_fallbacks':0,'intermediate_panel_downloads':0}\n"
            "diagnostics.update({key:'cuda' for key in ('first_integral_backend',"
            "'second_integral_backend','grid_ad_backend','response_backend')})\n"
            "result={key:request[key] for key in "
            "('operation','native_source_identity','charge','multiplicity')}\n"
            "result.update(schema='generativeqc.compiler-hessian-result/v1',"
            "method=request['calculator']['method'],backend=request['calculator']['device'],"
            "atom_count=count,units='hartree/bohr^2',identity='a'*64,native_library_path=sys.argv[-1],"
            "diagnostics=diagnostics,request_echo=request)\n"
            "result['value' if operation=='hvp' else 'matrix']="
            "[[0.]*(3 if operation=='hvp' else 3*count) "
            "for row in range(count if operation=='hvp' else 3*count)]\n"
            "json.dump(result,sys.stdout)\n"
        )
        self.interpreter.chmod(0o700)

    def call(
        self,
        *arguments: str,
        interpreter: Path | None = None,
        operation: str = "hvp",
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        command = [
            str(CLI),
            operation,
            str(self.xyz),
            "--units",
            "bohr",
            "--json",
            "--compiler-python",
            str(interpreter or self.interpreter),
        ]
        if operation == "hvp":
            command += ["--direction", str(self.direction_file)]
        if MONOLITHIC:
            command += ["--compiler-library", str(self.interpreter)]
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            **kwargs,
        )

    def test_native_help_documents_explicit_optional_runtime(self) -> None:
        result = subprocess.run(
            [str(CLI), "hessian", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env={"PATH": ""},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--compiler-python", result.stdout)
        self.assertIn("requires Python", result.stdout)

    def test_synthetic_cuda_transport_preserves_every_control(self) -> None:
        result = self.call(
            "--backend",
            "cuda",
            "--method",
            "wb97m-v-uks",
            "--charge",
            "1",
            "--multiplicity",
            "2",
            "--representation",
            "spherical",
            "--density-fitting",
            "cuda",
            "--auxiliary-basis",
            "def2-svp",
            "--max-iterations",
            "240",
            "--energy-tolerance",
            "1e-13",
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
            "--response-atol",
            "1e-12",
            "--response-rtol",
            "1e-11",
            "--response-restart",
            "17",
            "--response-max-iterations",
            "130",
            "--integral-budget-bytes",
            "12345",
            "--grid-budget-bytes",
            "67890",
            "--response-budget-bytes",
            "54321",
            "--max-nonlocal-pairs",
            "65536",
            "--cache",
            str(self.root / "Cache With Spaces"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        request = json.loads(result.stdout)["request_echo"]
        self.assertEqual(request["atoms_bohr"][0], [1, [0.03, -0.02, 0.01]])
        self.assertEqual(request["direction"], self.direction)
        self.assertEqual(request["calculator"]["basis_representation"], "spherical")
        self.assertEqual(request["calculator"]["density_fitting"], "cuda")
        self.assertEqual(request["calculator"]["auxiliary_basis"], "def2-svp")
        self.assertIsNone(request["calculator"]["initial_guess"])
        self.assertEqual(request["calculator"]["max_iterations"], 240)
        self.assertEqual(
            request["grid"],
            {"radial_points": 8, "angular_polar": 4, "angular_azimuth": 8},
        )
        self.assertEqual(
            request["solver"],
            {
                "atol": 1e-12,
                "rtol": 1e-11,
                "restart": 17,
                "max_iterations": 130,
                "max_workspace_bytes": 54321,
            },
        )
        self.assertEqual(request["controls"]["integral_budget_bytes"], 12345)
        self.assertEqual(request["controls"]["grid_budget_bytes"], 67890)
        self.assertEqual(request["controls"]["max_nonlocal_pairs"], 65536)
        self.assertEqual(
            request["controls"]["cache"], str(self.root / "Cache With Spaces")
        )

    def test_native_only_invocation_never_selects_implicit_python(self) -> None:
        result = subprocess.run(
            [str(CLI), "hessian", str(self.xyz)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("--compiler-python", result.stderr)

    @unittest.skipIf(
        MONOLITHIC, "monolithic builds explicitly select a separate shared library"
    )
    def test_explicit_library_cannot_change_the_linked_native_owner(self) -> None:
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        library = json.loads(result.stdout)["native_library_path"]
        matching = self.call("--compiler-library", library)
        self.assertEqual(matching.returncode, 0, matching.stderr)
        different = self.call("--compiler-library", str(self.xyz))
        self.assertNotEqual(different.returncode, 0)
        self.assertEqual(different.stdout, "")
        self.assertIn("already-linked native library", different.stderr)

    @unittest.skipUnless(
        MONOLITHIC, "only monolithic builds need an explicit shared library"
    )
    def test_monolithic_extension_requires_explicit_shared_library(self) -> None:
        result = subprocess.run(
            [
                str(CLI),
                "hessian",
                str(self.xyz),
                "--compiler-python",
                str(self.interpreter),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires --compiler-library", result.stderr)

    def test_invalid_options_never_start_worker(self) -> None:
        for arguments in (
            ("--method", "pbe-rks"),
            ("--precision", "auto"),
            ("--backend", "cuda", "--density-fitting", "cpu"),
            ("--response-rtol", "nan"),
            ("--response-rtol", "1"),
            ("--response-atol", "-1"),
            ("--response-budget-bytes", "0"),
            ("--timeout-seconds", "0"),
            ("--forces", "yes"),
            ("--compiler-python", "python"),
            ("--compiler-python", "/no/such/python"),
            ("--compiler-library", "relative-library.so"),
        ):
            with self.subTest(arguments=arguments):
                result = self.call(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_direction_and_full_publication_budget_gates(self) -> None:
        for direction in ([1, 2, 3], [[0, 0, 0]], [[True, 0, 0], [0, 0, 0]]):
            with self.subTest(direction=direction):
                self.direction_file.write_text(json.dumps(direction))
                result = self.call()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        result = self.call("--output-budget-bytes", "575", operation="hessian")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        result = self.call("--direction", str(self.direction_file), operation="hessian")
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_worker_failure_and_timeout_publish_nothing(self) -> None:
        for program in (
            "import sys; print('{}'); sys.exit(7)",
            "import time; print('{}',flush=True); time.sleep(30)",
        ):
            with self.subTest(program=program):
                self.interpreter.write_text(f"#!{sys.executable}\n{program}\n")
                result = self.call("--timeout-seconds", "1")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("no result published", result.stderr)

    def test_malformed_or_unqualified_result_is_not_published(self) -> None:
        source = self.interpreter.read_text()
        for mutation in (
            "result['matrix']=[[0.]]",
            "result['native_source_identity']='wrong'",
            "result['units']='angstrom'",
            "result['diagnostics']['displaced_scf_evaluations']=2",
            "result['diagnostics']['numerical_host_fallbacks']=1",
            "result['diagnostics']['execution_residency']='host'",
            "result['diagnostics']['first_integral_backend']='cpu'",
        ):
            with self.subTest(mutation=mutation):
                self.interpreter.write_text(
                    source.replace(
                        "json.dump(result,sys.stdout)",
                        mutation + "\njson.dump(result,sys.stdout)",
                    )
                )
                result = self.call("--backend", "cuda", operation="hessian")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    @unittest.skipIf(
        MONOLITHIC, "requires a separately built source-matched shared library"
    )
    def test_complete_cpu_matrix_hvp_and_independent_native_force_oracle(self) -> None:
        common = [
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
            "--energy-tolerance",
            "1e-13",
            "--density-tolerance",
            "1e-11",
            "--max-iterations",
            "240",
        ]
        result = self.call(
            *common,
            "--cache",
            str(self.root / "artifacts"),
            interpreter=Path(sys.executable),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        hvp = json.loads(result.stdout)
        matrix = self.call(
            *common,
            "--cache",
            str(self.root / "artifacts"),
            interpreter=Path(sys.executable),
            operation="hessian",
        )
        self.assertEqual(matrix.returncode, 0, matrix.stderr)
        matrix = json.loads(matrix.stdout)
        self.assertFalse(matrix["diagnostics"]["posthoc_symmetrization"])
        motion = [item for row in self.direction for item in row]
        values = [item for row in hvp["value"] for item in row]
        for row, expected in zip(matrix["matrix"], values, strict=True):
            self.assertAlmostEqual(
                sum(left * right for left, right in zip(row, motion, strict=True)),
                expected,
                delta=2e-8,
            )
        forces = []
        os.environ["GENERATIVEQC_LIBRARY"] = hvp["native_library_path"]
        from generativeqc import Calculator, GridSpec, KsOptions

        calculator = Calculator(
            method="wb97m-v",
            device="cpu",
            basis_representation="spherical",
            energy_tolerance=1e-13,
            density_tolerance=1e-11,
            max_iterations=240,
            initial_guess=None,
            ks_options=KsOptions(
                grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
            ),
        )
        for sign in (-1, 1):
            positions = [[0.03, -0.02, 0.01], [0.15, 0.13, 1.5]]
            atoms = [
                (
                    "H",
                    [
                        value + sign * 3e-4 * delta
                        for value, delta in zip(row, direction, strict=True)
                    ],
                )
                for row, direction in zip(positions, self.direction, strict=True)
            ]
            singlepoint = calculator.singlepoint(atoms, properties=("energy", "forces"))
            self.assertTrue(singlepoint.converged)
            forces.append(singlepoint.forces.tolist())
        calculator.clear_cache()
        flat_forces = [[item for row in force for item in row] for force in forces]
        for left, right, expected in zip(*flat_forces, values, strict=True):
            self.assertAlmostEqual(-(right - left) / 6e-4, expected, delta=3e-6)


if __name__ == "__main__":
    unittest.main()
