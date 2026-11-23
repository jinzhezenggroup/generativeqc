"""Check the same prepared CPU total model through the SDK, CLI and Python."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._native_cpu_hessian import linked_method
from generativeqc_compiler.method.host_hessian_aot import parse_inventory


def compare(sdk: str, cli: str, method: str, *, expected_linked: bool) -> None:
    """Compare both molecular products, accepted energy and the D3 work receipts."""
    unrestricted = method.endswith("-uks")
    element = "He" if unrestricted else "H"
    multiplicity = 2 if unrestricted else 1
    atoms = [(element, (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = [[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]]
    representation = "spherical"
    calculator = Calculator(
        method=method,
        device="cpu",
        basis="sto-3g",
        basis_representation=representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    assert linked_method(calculator._library, calculator.method_ir.identity) == (
        expected_linked
    ), f"linked total MethodIR for {method} disagrees with the selected inventory"
    runtime = dict(
        os.environ,
        PATH="/nonexistent",
        PYTHONHOME="/nonexistent",
        PYTHONPATH="/nonexistent",
    )

    def run(*arguments: str) -> dict:
        return json.loads(subprocess.check_output(arguments, env=runtime, timeout=60))

    with tempfile.TemporaryDirectory(prefix="gqc-hvp-sdk-parity-") as directory:
        xyz = Path(directory) / "input.xyz"
        vector = Path(directory) / "direction.json"
        xyz.write_text(f"2\nBohr\n{element} 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
        vector.write_text(json.dumps(direction))
        command = (
            cli,
            str(xyz),
            "--json",
            "--units",
            "bohr",
            "--method",
            method,
            "--multiplicity",
            str(multiplicity),
            "--basis",
            "sto-3g",
            "--backend",
            "cpu",
            "--representation",
            representation,
            "--grid-radial-points",
            "12",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
            "--energy-tolerance",
            "1e-12",
            "--density-tolerance",
            "1e-10",
            "--screening-tolerance",
            "1e-14",
        )
        hvp_command = (cli, "hvp", *command[1:], "--direction", str(vector))
        refused_method = "pbe-d3(0)-rks" if expected_linked else method
        unsupported = subprocess.run(
            tuple(
                refused_method if argument == method else argument
                for argument in hvp_command
            ),
            env=runtime,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert unsupported.returncode != 0
        assert not unsupported.stdout
        assert (
            "native CPU Hessian has no exact linked total MethodIR"
            in unsupported.stderr
        )
        if not expected_linked:
            return

        cpp = run(sdk, "--cpu", method)
        cli_action = run(*hvp_command)
        panel_file = Path(directory) / "directions.json"
        panel_command = (cli, "hvp", *command[1:], "--directions", str(panel_file))
        seed = np.asarray(direction)
        panel = np.stack((seed, -0.5 * seed))
        panel_file.write_text(json.dumps(panel.tolist()))
        cli_panel = run(*panel_command)
        python_panel = calculator.hessian_vector_products(
            atoms, panel, multiplicity=multiplicity
        )
        for values, diagnostic in (
            (cli_panel["value"], cli_panel["diagnostics"]),
            (python_panel.value, python_panel.diagnostics),
        ):
            assert np.shape(values) == panel.shape
            np.testing.assert_allclose(
                np.asarray(values).ravel(), cpp["hvp_panel"], atol=2e-8, rtol=0
            )
            assert diagnostic["response_strategy"] == "independent"
            assert (
                diagnostic["direction_count"] == diagnostic["completed_directions"] == 2
            )
            assert diagnostic["d3_pair_visits"] == cpp["panel_d3_pair_visits"]
        if method == "pbe-d3(bj)-rks":
            panel = np.stack(
                (seed, -0.5 * seed, np.zeros_like(seed), seed[::-1], 2 * seed)
            )
            panel_file.write_text(json.dumps(panel.tolist()))
            unbounded = run(*panel_command)
            output_bytes = str(unbounded["diagnostics"]["output_bytes"])
            output_flag = ("--output-budget-bytes", output_bytes)
            resource = run(
                cli,
                "resources",
                *command[1:],
                "--property",
                "hvp",
                "--hvp-directions",
                str(len(panel)),
                *output_flag,
            )
            inventory = resource["derivative_inventory"]
            assert inventory["direction_count"] == len(panel)
            assert (
                inventory["minimum_numeric_bytes"]
                == unbounded["diagnostics"]["minimum_numeric_bytes"]
            )
            peak = resource["peak_bytes"]["host"]
            exact = run(*panel_command, *output_flag, "--host-bytes", str(peak))
            assert (
                exact["diagnostics"]["resource_admission"]["minimum_host_peak_bytes"]
                == peak
            )
            np.testing.assert_allclose(
                exact["value"], unbounded["value"], atol=2e-8, rtol=0
            )
            for limit, short in (
                ("--host-bytes", peak - 1),
                ("--output-budget-bytes", int(output_bytes) - 1),
            ):
                trace = Path(directory) / f"panel-{limit[2:]}.jsonl"
                rejected = subprocess.run(
                    (*panel_command, limit, str(short)),
                    env=dict(runtime, GENERATIVEQC_DF_PROGRESS_TRACE=str(trace)),
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                assert (
                    rejected.returncode != 0
                    and not rejected.stdout
                    and not trace.exists()
                )
        cli_matrix = run(cli, "hessian", *command[1:])
        if method in ("lda-rks", "pbe-rks", "pbe-d3(bj)-rks", "pbe-d3(bj)-uks"):
            packed = ("--cpu-packed-eri-budget-bytes", "48")
            resource = run(cli, "resources", *command[1:], "--property", "hvp", *packed)
            assert resource["packed_eri_inventory"]["retained_bytes"] == 48
            if "d3(bj)" in method:
                assert resource["correction_inventory"]["plan_host_bytes"] > 0
                trace = Path(directory) / "dispersion-short.jsonl"
                refused = subprocess.run(
                    (*hvp_command, "--dispersion-budget-bytes", "1"),
                    env=dict(runtime, GENERATIVEQC_DF_PROGRESS_TRACE=str(trace)),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                assert refused.returncode != 0 and not refused.stdout
                assert "CPU Hessian D3 storage cap exceeded" in refused.stderr
                assert not trace.exists()
            peak = resource["peak_bytes"]["host"]
            packed_action = run(*hvp_command, *packed, "--host-bytes", str(peak))
            admission = packed_action["diagnostics"]["resource_admission"]
            assert admission["minimum_host_peak_bytes"] == peak
            assert admission["selected_host_peak_bytes"] == peak
            assert admission["effective_host_limit_bytes"] == peak
            np.testing.assert_allclose(
                packed_action["value"], cli_action["value"], atol=2e-8, rtol=0
            )
            short_trace = Path(directory) / "short.jsonl"
            shortage = subprocess.run(
                (*hvp_command, *packed, "--host-bytes", str(peak - 1)),
                env=dict(runtime, GENERATIVEQC_DF_PROGRESS_TRACE=str(short_trace)),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert shortage.returncode != 0 and not shortage.stdout
            assert "allocation inventory exceeds host budget" in shortage.stderr
            assert not short_trace.exists()
        if method in ("lda-rks", "pbe-rks", "pbe-d3(bj)-rks"):
            sequential_command = (cli, "hessian", *command[1:], "--response-sequential")
            sequential_budget = cpp["sequential"]["output_bytes"]
            cli_sequential = run(
                *sequential_command, "--output-budget-bytes", str(sequential_budget)
            )
            python_sequential = calculator.hessian(atoms, strategy="sequential")
            for diagnostic in (
                cli_sequential["diagnostics"],
                python_sequential.diagnostics,
            ):
                assert diagnostic["response_strategy"] == "sequential"
                assert diagnostic["response_concurrent_rhs"] == 1
                assert diagnostic["completed_directions"] == 6
                for work in (
                    "response_iterations",
                    "induced_fock_actions",
                    "d3_pair_visits",
                ):
                    assert diagnostic[work] == cpp["sequential"][work]
            for values in (cli_sequential["matrix"], python_sequential.matrix):
                np.testing.assert_allclose(
                    np.asarray(values).ravel(), cpp["matrix"], atol=2e-8, rtol=0
                )
            rejected = subprocess.run(
                (
                    *sequential_command,
                    "--output-budget-bytes",
                    str(sequential_budget - 1),
                ),
                env=runtime,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            assert rejected.returncode != 0 and not rejected.stdout
            assert "output" in rejected.stderr.lower()
            cli_recycled = run(
                cli, "hessian", *command[1:], "--response-recycle-block-size", "4"
            )
            python_recycled = calculator.hessian(
                atoms, strategy="recycled", block_size=4
            )
            for diagnostic in (
                cli_recycled["diagnostics"],
                python_recycled.diagnostics,
            ):
                assert diagnostic["response_strategy"] == "recycled"
                assert diagnostic["block_size"] == 4
                assert diagnostic["completed_directions"] == 6
            for values in (cli_recycled["matrix"], python_recycled.matrix):
                np.testing.assert_allclose(
                    np.asarray(values).ravel(), cpp["matrix"], atol=2e-8, rtol=0
                )

    python_action = calculator.hessian_vector_product(
        atoms, direction, multiplicity=multiplicity
    )
    python_matrix = calculator.hessian(atoms, multiplicity=multiplicity)
    assert (cpp["d3_pair_visits"] > 0) == ("d3(bj)" in method)
    for cli_result, python_result, values, key, pair_key, count in (
        (cli_action, python_action, python_action.value, "value", "d3_pair_visits", 1),
        (
            cli_matrix,
            python_matrix,
            python_matrix.matrix,
            "matrix",
            "dense_d3_pair_visits",
            6,
        ),
    ):
        assert cli_result["backend"] == "cpu"
        assert python_result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        for diagnostic in (cli_result["diagnostics"], python_result.diagnostics):
            assert not diagnostic["runtime_compilation"]
            assert diagnostic["completed_directions"] == count
            assert diagnostic["d3_pair_visits"] == cpp[pair_key]
        np.testing.assert_allclose(
            [cli_result["energy_hartree"], python_result.diagnostics["total_energy"]],
            cpp["energy"],
            atol=2e-10,
            rtol=0,
        )
        for product in (cli_result[key], values):
            np.testing.assert_allclose(
                np.asarray(product).ravel(), cpp[key], atol=2e-8, rtol=0
            )


if __name__ == "__main__":
    sdk, cli, inventory_path = sys.argv[1:]
    inventory = parse_inventory(json.loads(Path(inventory_path).read_text()))
    compare(sdk, cli, "b3lyp-d3(bj)-rks", expected_linked=True)
    for selector in ("pbe0-rks", "pbe0-d3(bj)-rks", "pbe0-uks", "pbe0-d3(bj)-uks"):
        compare(
            sdk,
            cli,
            selector,
            expected_linked="PBE0" in inventory.additional_methods
            and (not selector.endswith("-uks") or "polarized" in inventory.spins),
        )
    for selector, functional in (
        ("lda-rks", "LDA_XC_PW"),
        ("lda-uks", "LDA_XC_PW"),
        ("pbe-rks", "PBE"),
        ("pbe-uks", "PBE"),
        ("pbe-d3(bj)-rks", "PBE"),
        ("pbe-d3(bj)-uks", "PBE"),
    ):
        compare(
            sdk,
            cli,
            selector,
            expected_linked=functional in inventory.additional_methods
            and (not selector.endswith("-uks") or "polarized" in inventory.spins),
        )
