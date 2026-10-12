"""Slurm-only qualification of the actual generic shell-pair derivative owner.

Linked probe resources and CUDA-event intervals are diagnostics, not molecular
endpoint speed claims. The independent Gaussian quadrature does not use the
production Hermite/Coulomb recurrence or forward differentiation.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from test_hermite_convolution import independent_jet, normalization

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    7: ((2, 0, 0), (0, 1, 1), (1, 0, 1), (0, 0, 1)),
    8: ((2, 0, 0), (0, 1, 1), (1, 1, 0), (0, 0, 2)),
    12: ((2, 1, 0), (0, 1, 2), (1, 1, 1), (1, 1, 1)),
}


def test_generic_shell_pair_spatial_response_on_gpu(tmp_path: Path) -> None:
    """Check every atom, repeated atoms, arbitrary directions and all radials."""
    if os.environ.get("GENERATIVEQC_DIRECT_SPATIAL_RESPONSE_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU qualification")
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    compiler = shutil.which("nvcc")
    cache = shutil.which("sccache") or shutil.which("ccache")
    assert compiler and cache
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    # This fixture uses ordinary cacheable object compilation. Runtime
    # JIT has a separate foreground-sccache lifetime contract and is not used.
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
    }
    shell = emit_direct_recurrence_headers()["generated_direct_shell_class.cuh"]
    begin = shell.index("/** Two independent radial values")
    end = shell.index("/**\n * Evaluate one Cartesian primitive quartet", begin)
    headers["probe_shell_pair.cuh"] = (
        '#include "generated_direct_cartesian.cuh"\n'
        '#include "generated_direct_shell_pair_hermite.cuh"\n'
        "namespace generativeqc::scf::cuda_execution {\n" + shell[begin:end] + "}\n"
    )
    for name, source in headers.items():
        (tmp_path / name).write_text(source)
    executable, object_file = (
        tmp_path / "spatial_response",
        tmp_path / "spatial_response.o",
    )
    environment = {**os.environ, "CCACHE_BASEDIR": str(ROOT)}
    compile_command = [
        cache,
        compiler,
        "-std=c++20",
        "-O3",
        "--fmad=false",
        "-arch=sm_120",
        f"-I{tmp_path}",
        f"-I{ROOT / 'src'}",
        f"-I{ROOT / 'include'}",
        "-c",
        str(ROOT / "tests/native/test_direct_spatial_response.cu"),
        "-o",
        str(object_file),
    ]
    (tmp_path / "compile-command.json").write_text(
        json.dumps(compile_command, indent=2)
    )
    (tmp_path / "cache-before.txt").write_bytes(
        subprocess.check_output([cache, "--show-stats"], env=environment)
    )
    compiled = subprocess.run(
        compile_command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )
    (tmp_path / "compile.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run(
        [compiler, "-arch=sm_120", str(object_file), "-o", str(executable)],
        env=environment,
        check=True,
        timeout=120,
    )
    (tmp_path / "cache-after.txt").write_bytes(
        subprocess.check_output([cache, "--show-stats"], env=environment)
    )
    expected, norms, rows = [], [], []
    for order, component in COMPONENTS.items():
        powers = np.asarray(component, dtype=np.uint32)
        for radial in range(3):
            for mask in (1, 2, 4, 8, 3, 5, 15, 17):
                for selected in range(4):
                    positions = np.asarray(
                        (
                            (0.1, -0.2, -0.8),
                            (0.3, 0.1, 0.7),
                            (-0.5, 0.6, 0.2),
                            (0.8, -0.4, 0.3),
                        )
                    )
                    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
                    if mask == 3:
                        positions[1] = positions[0]
                    if mask == 8:
                        exponents *= 0.025
                        positions *= 4
                    norm = normalization(powers, exponents)
                    reference = independent_jet(
                        powers, positions, exponents, radial, 0.3, mask
                    )
                    if mask == 17:
                        reference[1:] = reference[1:] @ np.asarray(
                            (
                                (1.0, 0.25, -0.5),
                                (-0.75, 1.0, 0.125),
                                (0.375, -0.25, 1.0),
                            )
                        )
                    expected.append(reference * norm)
                    norms.append(norm)
                    rows.append(
                        [
                            order,
                            radial,
                            mask,
                            selected,
                            *powers.ravel(),
                            *positions.ravel(),
                            *exponents,
                        ]
                    )
    requests = tmp_path / "requests.txt"
    np.savetxt(requests, np.asarray(rows), fmt="%.17g")
    executed = subprocess.run(
        [str(executable), str(requests)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (tmp_path / "gpu.log").write_text(executed.stdout + executed.stderr)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    receipt = json.loads(executed.stdout)
    actual = np.asarray(receipt["jets"]) * np.asarray(norms)[:, None]
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=2e-11)
    receipt["max_normalized_absolute_error"] = float(np.max(np.abs(actual - expected)))
    receipt["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    receipt["requests"] = len(rows)
    (tmp_path / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
