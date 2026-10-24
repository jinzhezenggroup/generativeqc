"""Executable canonical layout-copy projection; GPU gates are explicitly opt-in."""

from __future__ import annotations

import shutil
import subprocess
import typing
from dataclasses import replace

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LayoutTransfer,
    LoweringConstraints,
    LoweringCost,
    LoweringPrecision,
    OperandLayout,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
)
from generativeqc_compiler.common.precision import (
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)
from generativeqc_compiler.tensor.layout_contraction import (
    FAMILY,
    emit_layout_contraction_portfolio,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


def portfolio(
    mask: int = 7, beta: float = 0.25
) -> tuple[
    LoweringRequest,
    tuple[LoweringCandidate, ...],
    TargetCapabilities,
    CompilationIdentity,
]:
    m, n, k = 3, 5, 7
    precision = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (("operation", PrecisionDirective("float64", "float64", "float64")),)
        ),
        "operation",
        ("float64",) * (3 if beta else 2),
        "float64",
    )
    dense = (
        OperandLayout("a", (0, 2), (m, k), (k, 1)),
        OperandLayout("b", (2, 1), (k, n), (n, 1)),
        OperandLayout(
            "out", (0, 1), (m, n), (n, 1), access="read-write" if beta else "write"
        ),
    )
    source = tuple(
        replace(row, strides=(row.shape[1] * 3 + 2, 2)) if mask & (1 << i) else row
        for i, row in enumerate(dense)
    )
    science = canonical_hash({"einsum": "ik,kj->ij", "alpha": -0.75, "beta": beta})
    request = LoweringRequest(
        "tensor",
        "einsum",
        "cuda",
        "float64",
        "float64",
        (m, n),
        semantics=(("alpha", -0.75), ("beta", beta)),
        scientific_identity=science,
        operands=source,
        precisions=(precision,),
        constraints=LoweringConstraints(
            additional_device_bytes=128 << 20, host_bytes=32768
        ),
    )
    target = TargetCapabilities(cuda_target_info("sm_90").target_info)
    compilation = CompilationIdentity(science, canonical_hash({"executor": FAMILY}))
    transfers = tuple(
        LayoutTransfer(
            source[i] if stage == "before" else dense[i],
            dense[i] if stage == "before" else source[i],
            "float64",
            stage,
            FAMILY,
        )
        for i in range(3)
        if mask & (1 << i)
        for stage in ("before", "after")
        if (stage == "before" and (i != 2 or beta)) or (stage == "after" and i == 2)
    )
    buffers = sum(
        row.shape[0] * row.shape[1] * 8
        for i, row in enumerate(dense)
        if mask & (1 << i)
    )
    candidates = tuple(
        LoweringCandidate(
            request,
            f"layout-{name}",
            (ProviderDescriptor(name, kind, "matmul", version="qualification-source"),),
            "ready",
            precision.directive.math_mode,
            provider_bytes=(96 << 20) if name == "cublas" else 0,
            execution=CandidateExecution(
                precision,
                algorithm,
                dense,
                layout_transfers=transfers,
                temporary_bytes=buffers,
                host_bytes=32768,
            ),
            cost=LoweringCost(
                "qualification-only-estimate",
                "estimated",
                prepare_ns=1,
                kernel_ns=10 if name == "cublas" else 100,
                cast_ns=0,
                pack_ns=10,
                refinement_ns=0,
                audit_ns=0,
                fallback_ns=0,
                pack_bytes=sum(t.read_bytes + t.write_bytes for t in transfers),
                launches=len(transfers) + 2,
            ),
            target=target,
        )
        for name, kind, algorithm in (
            ("generated.cuda", "generated", "affine-copy-v1-generated"),
            ("cublas", "library", "affine-copy-v1-blas"),
        )
    )
    return request, candidates, target, compilation


def generated_cases() -> str:
    return '#include "tensor/layout_contraction.hpp"\n' + "\n".join(
        emit_layout_contraction_portfolio(*portfolio(mask, beta), name=f"case{i}")
        for i, (mask, beta) in enumerate(
            (mask, beta) for mask in (1, 2, 4, 7) for beta in (0.0, 0.25)
        )
    )


def test_generated_layout_plan_and_native_selector_compile_and_enforce_budgets(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "plan.cpp"
    checks = []
    for i in range(8):
        checks.append(f"""
{{
 auto p=case{i}_plan(0);
 p.validate_binding(case{i}_request,case{i}_candidates[0],case{i}_target,case{i}_compilation,
                    p.scratch_bytes(),32768);
 if(p.copy_launches()==0 || p.transfer_bytes()!=p.transfer_elements()*16) return 1;
 auto bad=case{i}_candidates[0]; bad.temporary_bytes=p.scratch_bytes()-1;
 if(!reject([&]{{p.validate_binding(case{i}_request,bad,case{i}_target,case{i}_compilation,
                                   p.scratch_bytes(),32768);}})) return 2;
 if(!reject([&]{{p.validate_binding(case{i}_request,case{i}_candidates[0],"stale",case{i}_compilation,
                                   p.scratch_bytes(),32768);}})) return 3;
 auto changed=p.physical; changed.operands[0].strides[0]++;
 if(!reject([&]{{generativeqc::tensor::LayoutContractionPlan q(
     p.request_identity,p.candidate_identity,p.original,changed);}})) return 4;
}}""")
    source.write_text(
        '#include "tensor/layout_contraction.hpp"\n'
        + generated_cases()
        + "\nint main(){auto reject=[](auto f){try{f();}catch(const std::exception&){return true;}return false;};"
        + "\n".join(checks)
        + "\n}\n"
    )
    binary = tmp_path / "plan"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "change", ["copy", "arithmetic", "capture", "effects", "layout", "order"]
)
def test_codegen_refuses_unsupported_executor_obligations(change: str) -> None:
    request, candidates, target, compilation = portfolio()
    candidate = candidates[1] if change == "order" else candidates[0]
    execution = candidate.execution
    assert execution is not None
    if change == "copy":
        execution = replace(
            execution,
            layout_transfers=tuple(
                replace(t, implementation="unknown-copy")
                for t in execution.layout_transfers
            ),
        )
    elif change == "capture":
        execution = replace(execution, capture_safe=True)
    elif change == "order":
        execution = replace(execution, determinism="exact-order")
    elif change == "layout":
        execution = replace(
            execution,
            layouts=tuple(replace(row, alignment=16) for row in execution.layouts),
        )
    elif change == "arithmetic":
        execution = replace(execution, algorithm="implicit-tf32")
    else:
        request = replace(request, effects=(("state", "donated"),))
    candidate = replace(candidate, request=request, execution=execution)
    with pytest.raises(ValueError):
        emit_layout_contraction_portfolio(
            request, (candidate,), target, compilation, name="unsupported"
        )


def test_layout_contraction_cuda_execution(tmp_path: Path) -> None:
    """Requested GPU qualification fails rather than silently skipping a prerequisite."""
    import os
    import re

    if os.environ.get("GENERATIVEQC_LAYOUT_CUDA_TEST") != "1":
        pytest.skip("requires explicit real NVIDIA layout endpoint qualification")
    cache = shutil.which("sccache") or shutil.which("ccache")
    compiler = shutil.which("nvcc")
    if cache is None or compiler is None:
        pytest.fail(
            "requested CUDA qualification requires a verified compiler cache and nvcc"
        )
    architecture = os.environ.get("GENERATIVEQC_LAYOUT_CUDA_ARCH", "")
    if not re.fullmatch(r"sm_[0-9]+", architecture):
        pytest.fail("qualification must bind the observed CUDA architecture")
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    (tmp_path / "generated_layout_portfolios.hpp").write_text(generated_cases())
    obj, binary = tmp_path / "layout.o", tmp_path / "layout"
    command = [
        cache,
        compiler,
        "-std=c++20",
        "-arch=" + architecture,
        "-DGENERATIVEQC_HAS_CUDA=1",
        "-I" + str(root / "src"),
        "-I" + str(root / "include"),
        "-I" + str(tmp_path),
        "-c",
        str(root / "tests/native/test_layout_contraction_cuda.cu"),
        "-o",
        str(obj),
    ]
    compiled = subprocess.run(
        command, capture_output=True, text=True, timeout=240, check=False
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    linked = subprocess.run(
        [cache, compiler, str(obj), "-lcublas", "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert linked.returncode == 0, linked.stdout + linked.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=120, check=False
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LAYOUT_GPU_PASS" in result.stdout


if __name__ == "__main__":
    import sys
    from pathlib import Path

    Path(sys.argv[1]).write_text(generated_cases())
