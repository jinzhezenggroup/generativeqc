"""Runtime shape lowering agrees with independently exercised RHF matrix IR."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.rhf_orbital_response import build_rhf_frame_response
from generativeqc_compiler.tensor import execute

from tools.generate_rhf_frame_response import (
    STAGES,
    cpu_header,
    cuda_source,
    inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_shapes_match_all_matrix_maps(tmp_path: Path) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ and ccache")
    (tmp_path / "maps.hpp").write_text(cpu_header())
    cases = []
    for o, v in ((1, 1), (1, 3), (2, 2), (3, 2)):
        rng = np.random.default_rng(1807 + o * 10 + v)
        maps = build_rhf_frame_response(o, v)
        for stage in STAGES:
            program = getattr(maps, stage)
            feeds = {
                node.attrs["name"]: np.asarray(rng.normal(size=node.spec.shape))
                for node in program.live_nodes
                if node.op == "input"
            }
            expected = execute(program, feeds).outputs
            block = ["{", f"const std::size_t o={o},v={v};", "Inputs input;"]
            for name in inputs(program):
                values = feeds[name].ravel()
                block += [
                    f"const double {name}[]={{"
                    + ",".join(float(x).hex() for x in values)
                    + "};",
                    f"input.{name}={name};",
                ]
            block += [
                f"std::vector<double> arena({stage}_arena_elements(o,v));",
                f"const auto result=run_{stage}_cpu(o,v,input,arena.data(),arena.size());",
            ]
            for name, array in expected.items():
                block += [
                    f"const double want_{name}[]={{"
                    + ",".join(float(x).hex() for x in array.ravel())
                    + "};",
                    f"for(std::size_t i=0;i<{array.size};++i) if(std::abs(result.{name}[i]-want_{name}[i])>3e-11) return 1;",
                ]
            cases.append("\n".join([*block, "}"]))
    source = tmp_path / "main.cpp"
    source.write_text(
        '#include "maps.hpp"\n#include <vector>\n'
        "using namespace generativeqc::scf::generated::rhf_frame;\n"
        "int main(){\n" + "\n".join(cases) + "\n}"
    )
    executable = tmp_path / "check"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_packed_blas_lowering_is_opt_in_with_scalar_fallback() -> None:
    source = cuda_source()
    assert "s.gemm('" in source
    for name in STAGES:
        assert f"run_{name}_blas(s) : run_{name}_scalar(s)" in source
    # Multiple maps compose under one sticky arithmetic audit. Only their owner
    # may clear it; a later map must not erase an earlier failed intermediate.
    assert "cudaMemsetAsync" not in source
    assert source == cuda_source()
