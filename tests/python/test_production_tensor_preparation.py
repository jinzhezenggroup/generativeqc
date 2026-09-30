"""Production TensorIR lowering must pass through the shared compiler preparation boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import Program, add, constant, multiply
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.optimize import prepare_for_backend

ROOT = Path(__file__).resolve().parents[2]


def test_prepare_for_backend_folds_constants_without_numpy_interpreter() -> None:
    program = Program({"value": add(constant(2), constant(3))})
    prepared = prepare_for_backend(program, "cpu")
    assert prepared.outputs["value"].op == "constant"
    assert prepared.outputs["value"].attrs["values"] == ((5, 1),)


def test_prepare_for_backend_is_site_package_independent() -> None:
    script = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / "python")!r}]
from generativeqc_compiler.tensor import Program, add, constant
from generativeqc_compiler.tensor.optimize import prepare_for_backend
program = Program({{"value": add(constant(2), constant(3))}})
prepared = prepare_for_backend(program, "cpu")
assert prepared.outputs["value"].op == "constant"
assert "numpy" not in sys.modules
"""
    run = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_cuda_planner_runs_exact_shared_preparation_before_lowering() -> None:
    live = multiply(constant(2), constant(3))
    dead = multiply(constant(4), constant(5))
    program = Program({"value": live}, definitions=(dead,))
    plan = plan_cuda(
        program, cuda_target_info("sm_80"), provider_bytes=0, library_bytes=0
    )
    assert plan.program.outputs["value"].op == "constant"
    assert len(plan.program.nodes) == 1


def test_production_generators_cannot_restore_private_optimizer_bypasses() -> None:
    rccsd = (ROOT / "tools/generate_rccsd_native.py").read_text()
    mp2 = (ROOT / "tools/generate_mp2_native.py").read_text()

    assert '"optimize": lambda program: program' not in rccsd
    assert "prepare_for_backend" in rccsd
    assert rccsd.count("_prepare_production(") >= 12

    assert "prepare_for_backend" in mp2
    assert mp2.count("prepare_for_backend(") >= 2

    scf = (ROOT / "tools/generate_scf_array_native.py").read_text()
    assert "prepare_for_backend" in scf
    assert "preserve_reduction_order=True" in scf

    stationary = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    wb97mv = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()
    assert "plan_cuda(" in stationary
    assert "plan_cuda(" in wb97mv


def test_gfn2_generators_use_shared_production_preparation() -> None:
    cpu = (ROOT / "tools/generate_gfn2_electronic_native.py").read_text()
    cuda = (ROOT / "tools/generate_gfn2_electronic_cuda.py").read_text()
    pair = (ROOT / "tools/generate_gfn2_pair_native.py").read_text()
    es2 = (ROOT / "tools/generate_gfn2_es2_native.py").read_text()

    assert cpu.count("prepare_for_backend(") >= 6
    assert 'backend="cpu"' in cpu
    assert cuda.count("prepare_for_backend(") >= 3
    assert 'backend="cuda"' in cuda

    # Pair/ES2 emit one source shared by host and device, so use the
    # backend-neutral scalar preparation domain rather than a CUDA-only route.
    assert pair.count("prepare_for_backend(") >= 2
    assert 'backend="scalar"' in pair
    assert es2.count("prepare_for_backend(") >= 4
    assert 'backend="scalar"' in es2
