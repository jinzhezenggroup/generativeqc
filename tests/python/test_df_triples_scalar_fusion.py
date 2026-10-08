"""Compiler-owned tile fusion against the original individual AD outputs."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import energy_scalar_program, inverse
from generativeqc_compiler.cc.occupied_triples_response import (
    energy_scalar_vjp,
    fused_tile_program,
)
from generativeqc_compiler.cc.triples import _LABELS, VP
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp
from test_df_occupied_triples import case
from test_df_occupied_triples_fock import (
    native_combined_probe,  # noqa: F401 -- pytest fixture
    run_combined,
)


def original_feed(feed: dict, coordinates: tuple[int, ...]) -> dict:
    """Rebind one legacy epilogue using independent coordinate composition."""
    labels = {VP[label]: label for label in _LABELS}
    label = labels[coordinates]
    result = {
        "bar_energy": feed["bar_energy"],
        "denominator": feed[f"denominator_{label}"],
    }
    for occupied in _LABELS:
        result[f"v_{occupied}"] = feed[f"v_{occupied}_{label}"]
        for virtual in _LABELS:
            mapped = tuple(coordinates[axis] for axis in VP[virtual])
            result[f"w_{occupied}_{virtual}"] = feed[f"w_{occupied}_{labels[mapped]}"]
    return result


def unfused(feed: dict) -> dict:
    """Preserve the original six individual derivative calls and gather order."""
    reverse = energy_scalar_vjp()
    direct = original_feed(feed, (0, 1, 2))
    result = {"energy": execute(energy_scalar_program(), direct).outputs["energy"]}
    derivatives = execute(reverse, direct).outputs
    for occupied in _LABELS:
        result[f"bar_v_{occupied}"] = derivatives[f"bar_v_{occupied}"]
        total = 0.0
        for virtual in _LABELS:
            values = execute(reverse, original_feed(feed, inverse(VP[virtual]))).outputs
            total += values[f"bar_w_{occupied}_{virtual}"]
        result[f"bar_w_{occupied}"] = np.array(total)
    return result


def scalar_case(seed: int) -> dict:
    """Distinct ordered denominators catch accidental symmetry-based merging."""
    rng = np.random.default_rng(seed)
    program = fused_tile_program()
    result = {
        node.attrs["name"]: np.array(rng.normal())
        for node in program.live_nodes
        if node.op == "input"
    }
    result["bar_energy"] = np.array(1.0)
    for label in _LABELS:
        result[f"denominator_{label}"] = np.array(-rng.uniform(1.0, 10.0))
    return result


@pytest.mark.parametrize("seed", [0, 19, 101])
def test_fused_outputs_match_original_gathers(seed: int) -> None:
    feed = scalar_case(seed)
    actual = execute(fused_tile_program(), feed).outputs
    for name, expected in unfused(feed).items():
        np.testing.assert_allclose(actual[name], expected, atol=2e-14, rtol=2e-14)


def test_fusion_identity_and_bounded_inventory() -> None:
    from tools.generate_df_occupied_triples import response_scalar_header

    program = fused_tile_program()
    assert program.logical_hash == fused_tile_program().logical_hash
    assert len(program.outputs) == 13
    assert all(not node.spec.shape for node in program.live_nodes)
    inputs = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    assert {f"denominator_{label}" for label in _LABELS} <= inputs
    assert all("eps" not in name for name in program.outputs)
    source = response_scalar_header()
    counts = {}
    for schedule in ("fused", "unfused"):
        for metric in ("value_reads", "arithmetic_ops"):
            key = f"scalar_{schedule}_{metric}"
            counts[key] = int(source.split(key + "=", 1)[1].split(";", 1)[0])
    assert counts["scalar_fused_value_reads"] < counts["scalar_unfused_value_reads"] / 2
    assert (
        counts["scalar_fused_arithmetic_ops"]
        < counts["scalar_unfused_arithmetic_ops"] / 2
    )


@pytest.fixture(scope="module")
def scalar_probe(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Execute the production scalar lowering with finite checks and strict FP64."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    program = fused_tile_program()
    inputs = tuple(
        sorted(
            {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
        )
    )
    outputs = tuple(sorted(program.outputs))
    directory = tmp_path_factory.mktemp("df-triples-fused-scalar")
    source, obj, library = (
        directory / name for name in ("probe.cpp", "probe.o", "probe.so")
    )
    arguments = ",".join(
        [
            *(f"inputs[{index}]" for index in range(len(inputs))),
            *(f"outputs[{index}]" for index in range(len(outputs))),
        ]
    )
    source.write_text(
        "#include <cmath>\n"
        + emit_scalar_cpp(
            program,
            function_name="fused",
            ordered_native_sums=True,
            output_dependency_order=True,
        )
        + '\nextern "C" bool run(const double* inputs,double* outputs){return fused('
        + arguments
        + ");}\n"
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-ffp-contract=off",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    dll = ct.CDLL(str(library))
    call = dll.run
    call.argtypes = [ct.POINTER(ct.c_double)] * 2
    call.restype = ct.c_bool
    return call, inputs, outputs


@pytest.mark.parametrize("seed", [0, 19, 101])
def test_native_scalar_fusion_preserves_fp64_gathers(
    scalar_probe: tuple, seed: int
) -> None:
    call, inputs, outputs = scalar_probe
    feed = scalar_case(seed)
    values = np.array([feed[name].item() for name in inputs])
    result = np.full(len(outputs), np.nan)
    assert call(
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        result.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    expected = unfused(feed)
    for name, actual in zip(outputs, result, strict=True):
        np.testing.assert_allclose(actual, expected[name], atol=2e-14, rtol=2e-14)


def test_fused_native_energy_tree_spans_multiple_ctas(
    native_combined_probe: object,  # noqa: F811 -- imported pytest fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise partial-block energy reduction and all occupied degeneracies."""
    from generativeqc_compiler.cc.triples import triples_energy

    inputs, _ = case(3, 9, 4)
    monkeypatch.setenv("GENERATIVEQC_TEST_FUSED_TRIPLES_SCALARS", "0")
    status, legacy, values, legacy_counts, error = run_combined(
        native_combined_probe, inputs
    )
    assert status == 0, error
    monkeypatch.setenv("GENERATIVEQC_TEST_FUSED_TRIPLES_SCALARS", "1")
    status, fused, fused_values, counts, error = run_combined(
        native_combined_probe, inputs
    )
    assert status == 0, error
    assert counts[15] == 1
    points = int(counts[17]) * 9**3
    partials = int(counts[22]) - 12 * points
    assert partials > counts[17]
    assert counts[1] == legacy_counts[1]
    for actual, expected in zip(fused, legacy, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    conventional = np.einsum("Qia,Qfb->iafb", inputs[0], inputs[1])
    energy = triples_energy(3, 9, conventional, *inputs[2:])
    np.testing.assert_allclose(fused_values[0], energy, atol=2e-12, rtol=0)
    np.testing.assert_allclose(fused_values[0], values[0], atol=2e-12, rtol=0)
