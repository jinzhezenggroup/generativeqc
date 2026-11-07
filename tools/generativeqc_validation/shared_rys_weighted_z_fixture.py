"""Reuse frozen Cartesian arrays for exact weighted-Z companion qualification.

No reference evaluator is called. Guard observations extract the actual emitted
task prefix and stay outside the exact uninstrumented timed specialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from generativeqc_compiler.integral import FusedShellPlan, emit_shell_class_fused_cuda
from generativeqc_compiler.integral.production_emission import _PRODUCTION_PRELUDE
from generativeqc_compiler.integral.tuning.policy import schedule_payload

from tools.generativeqc_validation.shared_rys_xy_fixture import (
    _RUNTIME,
    CLASSES,
    MASKS,
    TASK_MASKS,
    _case_runner,
    fixture_plan,
)


def companion_plan(name: str, *, weighted: bool) -> FusedShellPlan:
    """Keep the qualified SM90 mapping and vary only the typed capability."""
    plan = fixture_plan(name, shared=True)
    return replace(
        plan,
        kernel=replace(
            plan.kernel,
            schedule=replace(plan.schedule, fock_shared_rys_weighted_z=weighted),
        ),
    )


def admission_source(body: str, prefix: str, class_name: str) -> tuple[str, dict]:
    """Observe the real emitted pre-primitive guard, without evaluating ERIs."""
    symbol = prefix + "_shell_class_fock_shared_xy_task"
    start = body.index("__device__ __forceinline__ void " + symbol)
    end = body.index("  const unsigned ax =", start)
    original = body[start:end]
    observed = original.replace(symbol, prefix + "_admission_task", 1)
    observed = observed.replace(
        "  if (retained_count == 0U) return;",
        f"  if (retained_count == 0U) {{ if(lane==0U) atomicAdd({prefix}_admission_counts,1ULL); return; }}",
        1,
    )
    observed += f"""
  if(lane==0U) atomicAdd({prefix}_admission_counts +
      (share_weighted_z ? 1U : (share_xy ? 2U : 3U)),1ULL);
}}
__global__ void {prefix}_admission_kernel(
    const {class_name}ShellTask* tasks,
    const {class_name}PrimitivePairData* pairs,const std::int64_t* offsets,
    const double* coefficients,const {class_name}Vec3* positions,
    double threshold,const double* schwarz,const double* density,double* output,
    std::size_t count) {{
  for(std::size_t index=blockIdx.x;index<count;index+=gridDim.x) {{
    {prefix}_admission_task(tasks,pairs,offsets,coefficients,positions,
        threshold,schwarz,density,generativeqc::runtime::CompensatedOutput(output),index);
  }}
}}
"""
    code = f"\n__device__ unsigned long long {prefix}_admission_counts[4];\n" + observed
    return code, {
        "original_emitted_guard_prefix_sha256": hashlib.sha256(
            original.encode()
        ).hexdigest(),
        "observer_changes": "rename task; count zero return and uniform final decision; omit primitive loop and publication",
        "counter_order": ["zero", "admitted", "mismatch_fallback", "small_live"],
    }


_GUARD_RUNTIME = r"""
template<class Launch> void guard_check(const char* label,Launch launch,void* counters,
    const unsigned long long* expected,cudaStream_t stream,bool graph) {
  auto enqueue=[&]() {CUDA(cudaMemsetAsync(counters,0,4*sizeof(unsigned long long),stream)); launch(); CUDA(cudaPeekAtLastError());};
  cudaGraph_t captured{};cudaGraphExec_t executable{};
  if(graph) {
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));enqueue();
    CUDA(cudaStreamEndCapture(stream,&captured));
    CUDA(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0));
  }
  for(unsigned repeat=0;repeat<2;++repeat) {
    if(graph) CUDA(cudaGraphLaunch(executable,stream));else enqueue();
    CUDA(cudaStreamSynchronize(stream));unsigned long long observed[4]{};
    CUDA(cudaMemcpy(observed,counters,sizeof(observed),cudaMemcpyDeviceToHost));
    for(unsigned k=0;k<4;++k) if(observed[k]!=expected[k]) {
      std::fprintf(stderr,"GUARD_FAIL %s graph=%d replay=%u category=%u actual=%llu expected=%llu\n",label,graph,repeat,k,observed[k],expected[k]);std::exit(6);
    }
    std::printf("GUARD_PASS %s graph=%d replay=%u zero=%llu admitted=%llu mismatch_fallback=%llu small_live=%llu\n",label,graph,repeat,observed[0],observed[1],observed[2],observed[3]);
  }
  if(graph) {CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(captured));}
}
"""


def _runner(metadata: dict, mode: str = "original") -> str:
    name, variant = metadata["name"], metadata["variant"]
    tag = name + "_" + variant
    source = _case_runner(metadata)
    source = source.replace("private", "xy").replace("Private", "Xy")
    source = source.replace("shared", "weighted").replace("Shared", "Weighted")
    label = tag if mode == "original" else tag + "_" + mode
    source = source.replace("run_" + tag + "(", "run_" + label + "(")
    source = source.replace(f'execute("{tag}_', f'execute("{label}_')
    n = metadata["n"]
    setup = f"  std::vector<double> control_coefficients({tag}_coefficients,{tag}_coefficients+{n});\n"
    offset = f"{tag}_ao_offsets[0]"
    if mode == "ulp_mismatch":
        setup += f"  control_coefficients[{offset}+1]=std::nextafter(control_coefficients[{offset}+1],INFINITY);\n"
    elif mode in ("signed_zero_match", "signed_zero_mismatch"):
        dim = 6 if name == "dppp" else 3
        setup += f"  for(unsigned a=0;a<{dim};++a) control_coefficients[{offset}+a]={('-0.0' if mode == 'signed_zero_match' else '0.0')};\n"
        if mode == "signed_zero_mismatch":
            setup += f"  control_coefficients[{offset}+1]=-0.0;\n"
    elif mode == "nonfinite_representative":
        setup += f"  control_coefficients[{offset}]=INFINITY;\n"
    elif mode != "original":
        raise ValueError("Unknown focused guard control")
    source = source.replace(
        f"  auto* coefficients=upload({tag}_coefficients,{n});",
        setup + f"  auto* coefficients=upload(control_coefficients.data(),{n});",
    )
    if mode.startswith("signed_zero"):
        source = source.replace(
            "  double* output=nullptr;",
            f"  std::vector<double> zero_expected({5 * n * n},0.0);\n  double* output=nullptr;",
        ).replace(f",head,{tag}_expected,", ",head,zero_expected.data(),")
    if mode == "nonfinite_representative":
        # This control checks refusal only: no nonfinite chemical matrix gets
        # accepted or compared with a relaxed tolerance.
        begin = source.index("  const auto xy_output=")
        end = source.index("  for(void* pointer:", begin)
        source = source[:begin] + source[end:]
    admitted, mismatch = 3, 0
    if mode in ("ulp_mismatch", "signed_zero_mismatch"):
        admitted, mismatch = (1, 2) if name == "dppp" else (0, 3)
    elif mode == "nonfinite_representative":
        admitted, mismatch = 0, 3
    observation = f"""
  void* guard_counters=nullptr;
  CUDA(cudaGetSymbolAddress(&guard_counters,weighted_{name}_admission_counts));
  const unsigned long long expected_guard[4]={{2,{admitted},{mismatch},2}};
  guard_check("{label}",[&]() {{
    weighted_{name}_admission_kernel<<<1,{192 if name == "dppp" else 96},0,stream>>>(
        reinterpret_cast<const Weighted{name.capitalize()}ShellTask*>(tasks),
        reinterpret_cast<const Weighted{name.capitalize()}PrimitivePairData*>(pairs),
        offsets,coefficients,reinterpret_cast<const Weighted{name.capitalize()}Vec3*>(positions),
        0.5,bounds,density,output,7);
  }},guard_counters,expected_guard,stream,graph);
"""
    return source.replace(
        "  for(void* pointer:", observation + "  for(void* pointer:", 1
    )


def prepare_fixture(directory: Path, frozen_inputs: Path) -> dict:
    """Emit two specializations and reuse all54 authenticated reference arrays."""
    directory.mkdir(parents=True, exist_ok=False)
    frozen = (frozen_inputs / "shared_rys_xy_fixture.cu").read_bytes()
    receipt = json.loads((frozen_inputs / "receipt.json").read_text())
    if hashlib.sha256(frozen).hexdigest() != receipt["source_sha256"]:
        raise ValueError("Frozen independent fixture changed")
    arrays = re.findall(r"^static const .*?;\n", frozen.decode(), re.MULTILINE)
    if len(arrays) != 54 or tuple(receipt["masks"]) != MASKS:
        raise ValueError("Frozen independent arrays/masks incomplete")
    source = [
        _PRODUCTION_PRELUDE,
        "\n#include <algorithm>\n#include <cstdio>\n#include <cstdlib>\n#include <utility>\n#include <vector>\n",
    ]
    schedules, observers = {}, {}
    for name in CLASSES:
        for mode in ("xy", "weighted"):
            plan = companion_plan(name, weighted=mode == "weighted")
            body = emit_shell_class_fused_cuda(plan.spec, plan)
            body = body.replace("generated_" + name, mode + "_" + name)
            body = body.replace(
                "Generated" + name.capitalize(), mode.capitalize() + name.capitalize()
            )
            source.append(body)
            schedules[name + "_" + mode] = schedule_payload(plan.schedule)
            if mode == "weighted":
                observed, proof = admission_source(
                    body, "weighted_" + name, "Weighted" + name.capitalize()
                )
                source.append(observed)
                observers[name] = proof
    source += arrays + [_RUNTIME, _GUARD_RUNTIME]
    controls = []
    for case in receipt["cases"]:
        source.append(_runner(case))
        if case["variant"] == "cartesian":
            for mode in (
                "ulp_mismatch",
                "signed_zero_match",
                "signed_zero_mismatch",
                "nonfinite_representative",
            ):
                source.append(_runner(case, mode))
                controls.append(
                    {
                        "name": case["name"],
                        "mode": mode,
                        "matrix_checked": mode != "nonfinite_representative",
                    }
                )
    source.append(
        "\nint main() {cudaStream_t stream;CUDA(cudaStreamCreate(&stream));\n"
    )
    for name in CLASSES:
        for mode in ("xy", "weighted"):
            symbol = mode + "_" + name + "_shell_class_fock_rhf_persistent_kernel"
            source.append(f'  resources("{symbol}",{symbol});\n')
    source.append("  for(bool graph:{false,true}) {\n")
    for case in receipt["cases"]:
        source.append(f"    run_{case['name']}_{case['variant']}(stream,graph);\n")
    for case in controls:
        source.append(
            f"    run_{case['name']}_cartesian_{case['mode']}(stream,graph);\n"
        )
    source.append(
        '  }\nCUDA(cudaStreamDestroy(stream));std::puts("ALL SYNTHETIC GATES PASS");\n}\n'
    )
    text = "".join(source)
    (directory / "shared_rys_xy_fixture.cu").write_text(text)
    result = {
        "scope": "existing independent arrays plus focused exact weighted-Z guard controls; no new references",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "frozen_independent_source_sha256": receipt["source_sha256"],
        "array_declarations": 54,
        "array_sha256": hashlib.sha256("".join(arrays).encode()).hexdigest(),
        "schedules": schedules,
        "cases": receipt["cases"],
        "masks": MASKS,
        "task_mask_sequence": TASK_MASKS,
        "guard_controls": controls,
        "guard_observers": observers,
        "total_matrix_predicates": 96,
        "independent_matrix_predicates": 48,
        "total_guard_controls": 56,
        "expected_logical_shared_bytes": {"pppp": 1152, "dppp": 1664},
        "fock_tolerances": {"atol": 2e-11, "rtol": 2e-10},
        "cuda_status": "CUDA compile and execution UNRUN",
    }
    (directory / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_fixture(args.output, args.frozen_inputs)))


if __name__ == "__main__":
    main()
