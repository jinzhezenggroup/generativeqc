"""Execute build-only CPU KS maps against independent displaced generalized SCF."""

from __future__ import annotations

import ctypes as ct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import test_ks_cpks as reference
from generativeqc_compiler.method.ks_cpks import build_ks_cpks_maps
from generativeqc_compiler.tensor.host_affine import emit_affine_host

from tools.generate_host_ks_cpks import generate

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)


class Array(ct.Structure):
    """Borrow one exact contiguous FP64 panel through the generated view ABI."""

    _fields_ = [("data", POINTER), ("elements", ct.c_size_t)]


def _view(values: np.ndarray) -> Array:
    return Array(values.ctypes.data_as(POINTER), values.size)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    directory = tmp_path_factory.mktemp("host-ks-maps")
    (directory / "maps.hpp").write_text(generate())
    probe = directory / "probe.cpp"
    probe.write_text(r"""
#include "maps.hpp"
namespace maps = generativeqc_host_ks_maps;
const maps::Program* entry(const char* name) {
  for (const auto& program : maps::programs()) if (program.name == name) return &program;
  return nullptr;
}
extern "C" int query(const char* name, std::size_t basis, std::size_t occupied,
                      std::size_t virtuals, std::size_t* bytes, std::size_t* terms) {
  const auto* program = entry(name);
  if (!program) return -1;
  const maps::Extents shape{basis, occupied, virtuals};
  return program->workspace(shape, *bytes) && program->work(shape, *terms) ? 0 : -1;
}
extern "C" const char* identity(const char* name) {
  const auto* program = entry(name);
  return program ? program->equation.data() : nullptr;
}
extern "C" int evaluate(const char* name, std::size_t basis, std::size_t occupied,
                         std::size_t virtuals, const maps::Input* inputs,
                         std::size_t input_count, const maps::Output* outputs,
                         std::size_t output_count, double* arena, std::size_t arena_count) {
  const auto* program = entry(name);
  if (!program) return -1;
  return static_cast<int>(program->evaluate({basis, occupied, virtuals},
      {inputs, input_count}, {outputs, output_count}, {arena, arena_count}));
}
""")
    library = required_native_cxx.build_shared(
        [probe],
        directory / "maps.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
        ],
    )
    loaded = ct.CDLL(str(library))
    loaded.query.argtypes = (
        [ct.c_char_p] + [ct.c_size_t] * 3 + [ct.POINTER(ct.c_size_t)] * 2
    )
    loaded.identity.argtypes = [ct.c_char_p]
    loaded.identity.restype = ct.c_char_p
    loaded.evaluate.argtypes = (
        [ct.c_char_p]
        + [ct.c_size_t] * 3
        + [ct.POINTER(Array), ct.c_size_t] * 2
        + [POINTER, ct.c_size_t]
    )
    return loaded


def _call(
    native: Any,
    prefix: str,
    stage: str,
    program: Any,
    feeds: dict,
    basis: int,
    occupied: int,
    failure: str = "",
) -> SimpleNamespace:
    inputs = [node for node in program.nodes if node.op == "input"]
    inputs.sort(key=lambda node: node.attrs["name"])
    arrays = [
        np.array(feeds[node.attrs["name"]], dtype=np.float64, order="C", copy=True)
        for node in inputs
    ]
    outputs = {
        name: np.full(node.spec.shape, -9182.5)
        for name, node in program.outputs.items()
    }
    views = (Array * len(arrays))(*map(_view, arrays))
    output_views = (Array * len(outputs))(*map(_view, outputs.values()))
    name = (prefix + "_" + stage).encode()
    size, work = ct.c_size_t(), ct.c_size_t()
    assert (
        native.query(
            name, basis, occupied, basis - occupied, ct.byref(size), ct.byref(work)
        )
        == 0
    )
    arena = np.empty(size.value // 8)
    if failure == "nonfinite":
        arrays[0].flat[0] = np.nan
    if failure == "shape":
        views[0].elements += 1
    if failure == "alias":
        output_views[0] = views[0]
    status = native.evaluate(
        name,
        basis,
        occupied,
        basis - occupied,
        views,
        len(views),
        output_views,
        len(output_views),
        arena.ctypes.data_as(POINTER),
        arena.size - (1 if failure == "workspace" else 0),
    )
    return SimpleNamespace(status=status, outputs=outputs, work=work.value)


@pytest.mark.parametrize("nao,counts", reference.CASES)
@pytest.mark.parametrize("degenerate", [False, True])
def test_native_metric_rhs_density_weights_against_independent_scf(
    native: Any,
    monkeypatch: pytest.MonkeyPatch,
    nao: int,
    counts: tuple[int, ...],
    degenerate: bool,
) -> None:
    model = reference._model(nao, counts, degenerate=degenerate)
    prefix = "rks" if len(counts) == 1 else "uks"
    lookups = {
        program.logical_hash: (stage, count, program)
        for count in counts
        for stage, program in build_ks_cpks_maps(
            nao, count, occupation=model["occupation"]
        ).items()
    }

    def execute(program: Any, feeds: dict) -> Any:
        stage, count, actual = lookups[program.logical_hash]
        result = _call(native, prefix, stage, actual, feeds, nao, count)
        assert result.status == 0
        return result

    monkeypatch.setattr(reference, "execute", execute)
    actual = reference._compiler_response(model)
    for step in (1e-4, 3e-4):
        for computed, expected in zip(
            actual, reference._oracle(model, step), strict=True
        ):
            np.testing.assert_allclose(computed, expected, atol=3e-8, rtol=3e-7)
    metric_trace = np.einsum("sij,ji->s", actual[0], model["overlap"])
    metric_trace += np.einsum("sij,ji->s", model["density"], model["overlap_direction"])
    np.testing.assert_allclose(metric_trace, 0, atol=3e-13)


@pytest.mark.parametrize(
    "failure,status", [("nonfinite", 3), ("shape", 1), ("alias", 1), ("workspace", 2)]
)
def test_checked_arena_failures_publish_no_outputs(
    native: Any, failure: str, status: int
) -> None:
    basis, occupied = 4, 2
    program = build_ks_cpks_maps(basis, occupied)["metric"]
    feeds = {
        "coefficients": np.eye(basis),
        "occupied": np.eye(basis)[:, :occupied],
        "occupied_energies": np.array([-2.0, -1.0]),
        "virtual_energies": np.array([1.0, 2.0]),
        "overlap_direction": np.eye(basis) * 0.01,
    }
    result = _call(native, "uks", "metric", program, feeds, basis, occupied, failure)
    assert result.status == status
    for output in result.outputs.values():
        np.testing.assert_array_equal(output, -9182.5)
    recovered = _call(native, "uks", "metric", program, feeds, basis, occupied)
    assert recovered.status == 0 and recovered.work > 0


def test_overflow_and_exact_scientific_identity(native: Any) -> None:
    for prefix, occupation in (("rks", 2), ("uks", 1)):
        for name, program in build_ks_cpks_maps(occupation=occupation).items():
            assert (
                native.identity((prefix + "_" + name).encode()).decode()
                == program.logical_hash
            )
    assert native.identity(b"absent") is None
    size, work = ct.c_size_t(), ct.c_size_t()
    assert (
        native.query(
            b"uks_metric", ct.c_size_t(-1), 2, 2, ct.byref(size), ct.byref(work)
        )
        == -1
    )


def test_host_generation_has_no_runtime_imports_and_is_deterministic(
    tmp_path: Path,
) -> None:
    script = r"""
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generativeqc', 'cupy', 'torch', 'pyscf'):
            raise AssertionError('runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    outputs = [tmp_path / "first.hpp", tmp_path / "second.hpp"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(ROOT / "tools/generate_host_ks_cpks.py"),
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with pytest.raises(ValueError, match="unbound"):
        emit_affine_host(
            {"metric": build_ks_cpks_maps()["metric"]}, namespace="test", extents={}
        )
