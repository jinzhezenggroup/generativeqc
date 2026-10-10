"""Separately generated pair storage must not change existing DF owner semantics."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _cc_owner_test_support import write_df_cpu_headers
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.df_spectator_pairs import LADDER_OUTPUT
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    execute,
    input_tensor,
)
from test_df_cc_native_solver import _case
from test_df_cc_spectator_pairs import _summands

from tools import generate_df_ccsd_spectator_pairs as generator
from tools.generate_df_ccsd_core import cuda_header as core_cuda_header
from tools.generate_df_ccsd_hoisted import cuda_header as hoisted_cuda_header
from tools.generate_rccsd_native import (
    _dim,
    _uses_occupied_pairs,
    ordered_batch_accumulation,
)

ROOT = Path(__file__).resolve().parents[2]


def test_generation_is_independent_of_runtime_and_reference_packages() -> None:
    script = r"""
import importlib.abc
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / 'python')]
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'pyscf', 'torch', 'cupy', 'generativeqc'}:
            raise ImportError('compiler imported runtime dependency: ' + fullname)
        return None
sys.meta_path.insert(0, BlockRuntime())
from tools.generate_df_ccsd_spectator_pairs import cpu_header, cuda_header, cuda_source
assert 'run_ladder_norm_cpu' in cpu_header()
assert 'paired_batched_contractions' in cuda_header()
assert 'occupied_ladder_offset' in cuda_source()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_runtime_pair_dimension_and_storage_map_fail_closed() -> None:
    occupied_pairs = IndexSpace("occupied_pairs", "pair", 3)
    assert _dim(Index("pair", occupied_pairs)) == "occupied_pairs"
    with pytest.raises(ValueError, match="unsupported runtime-shape"):
        _dim(Index("pair", IndexSpace("composite_pairs", "pair", 3)))
    opaque = input_tensor(
        "opaque",
        TensorSpec(
            (Index("pair", IndexSpace("composite_pairs", "pair", 3)),), role="input"
        ),
    )
    assert not _uses_occupied_pairs(Program({"borrowed": opaque}))
    with pytest.raises(ValueError, match="storage-coordinate map"):
        ordered_batch_accumulation(
            generator.hoisted.batched_auxiliary_program(),
            "invalid_pairs",
            "CudaState",
            "AuxiliaryOutputs",
            "s.q",
            source_program=generator.programs()["auxiliary_batched"],
        )


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def pair_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, bool]:
    cuda = request.param == "cuda"
    if sys.platform == "win32":
        pytest.skip("requires a POSIX native compiler")
    if cuda and os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("explicit scheduled real-GPU qualification is required")
    compiler = shutil.which("nvcc" if cuda else "c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a native compiler and verified compiler cache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    directory = tmp_path_factory.mktemp("df-cc-occupied-pairs-" + request.param)
    write_df_cpu_headers(directory)
    for name, producer in (
        ("generated_df_ccsd_spectator_pairs_cpu.hpp", generator.cpu_header),
        ("generated_df_ccsd_spectator_pairs_cuda.cuh", generator.cuda_header),
        ("generated_df_ccsd_spectator_pairs_cuda.cu", generator.cuda_source),
        ("generated_df_ccsd_hoisted_cuda.cuh", hoisted_cuda_header),
        ("generated_df_ccsd_core_cuda.cuh", core_cuda_header),
    ):
        (directory / name).write_text(producer())
    sources = [ROOT / "tests/native/df_cc_spectator_pairs_probe.cpp"]
    command = [
        cache,
        compiler,
        "-std=c++20",
        "-O2",
        "-I" + str(directory),
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
    ]
    if cuda:
        command += ["-x", "cu", "-arch=sm_120", "-DDF_PROBE_CUDA=1"]
        sources.append(directory / "generated_df_ccsd_spectator_pairs_cuda.cu")
    objects = []
    for source in sources:
        obj = directory / (source.name + ".o")
        build = subprocess.run(
            [*command, "-c", str(source), "-o", str(obj)],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        assert build.returncode == 0, build.stdout + build.stderr
        objects.append(str(obj))
    executable = directory / "probe"
    subprocess.run(
        [compiler, *objects, *(["-lcublas"] if cuda else []), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable, cuda


@pytest.mark.parametrize(
    "nocc,nvir,naux,batch",
    [(1, 3, 2, 1), (2, 3, 4, 1), (2, 3, 4, 3), (3, 2, 5, 3), (2, 4, 5, 2)],
)
def test_native_pair_actions_reconstruct_every_cut_and_uneven_q_tail(
    pair_probe: tuple[Path, bool], nocc: int, nvir: int, naux: int, batch: int
) -> None:
    executable, cuda = pair_probe
    _, _, feeds = _case(nocc, nvir, naux)
    random = np.random.default_rng(2202 + nocc + nvir)
    feeds["t1"] = random.normal(scale=0.06, size=feeds["t1"].shape)
    doubles = random.normal(scale=0.08, size=feeds["t2"].shape)
    feeds["t2"] = (doubles + doubles.transpose(1, 0, 3, 2)) / 2
    pipeline = build_df_auxiliary_reduction_programs(nocc, nvir)
    tau = execute(pipeline.prepare, feeds).outputs["df_tau"]
    tau = (tau + tau.transpose(1, 0, 3, 2)) / 2
    paired = np.array(
        [tau[first, second] for first in range(nocc) for second in range(first, nocc)]
    )
    supplied = [feeds["t1"], feeds["t2"], tau, paired, feeds["bov"], feeds["bvv"]]
    seed = 173 if cuda and batch > 1 else 0
    data = f"{nocc} {nvir} {naux} {batch} {seed}\n" + "\n".join(
        " ".join(f"{value:.17g}" for value in array.flat) for array in supplied
    )
    if os.environ.get("GENERATIVEQC_DF_CC_SAVE_INPUTS") == "1":
        executable.with_name(f"case-{nocc}-{nvir}-{naux}-{batch}.txt").write_text(data)
    result = subprocess.run(
        [str(executable)],
        input=data,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("work ")
    expected_work = 0
    for auxiliary in range(0, naux, batch):
        tile = min(batch, naux - auxiliary)
        name = (
            "auxiliary_batched"
            if tile > 1
            else "auxiliary_packed"
            if cuda
            else "auxiliary"
        )
        expected_work += _summands(generator.programs()[name], nocc, nvir, tile)
    assert int(lines[0].split()[1]) == expected_work
    assert int(lines[1].split()[1]) == seed
    actual = np.fromstring(" ".join(lines[2:]), sep=" ")
    original = pipeline.auxiliary
    expected = {
        name: np.zeros(node.spec.shape) for name, node in original.outputs.items()
    }
    for auxiliary in range(naux):
        row = execute(
            original,
            {
                **feeds,
                "df_tau": tau,
                "bov": feeds["bov"][auxiliary],
                "bvv": feeds["bvv"][auxiliary],
            },
        ).outputs
        for name, values in row.items():
            expected[name] += values
    reference = np.concatenate([expected[name].ravel() for name in generator.FIELDS])
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=0)
    assert expected[LADDER_OUTPUT].shape == (nocc, nocc, nvir, nvir)
