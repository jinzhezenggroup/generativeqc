"""Prepare a standalone, opt-in shared-Rys-XY CUDA correctness executable.

This is a synthetic independent fixture, never an HF96 task capture. Running
this module only writes source and its identity receipt; compilation and real
GPU execution belong to the separately scheduled qualification owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    FusedShellPlan,
    KernelConsumer,
    build_fused_shell_plan,
    cuda_target_info,
    emit_shell_class_fused_cuda,
)
from generativeqc_compiler.integral.production_emission import _PRODUCTION_PRELUDE
from generativeqc_compiler.integral.tuning.policy import schedule_payload

from tools.generativeqc_validation.f_shell_numerics import (
    _reference_integrals,
    contract_reference,
    eri_orbit,
    make_fixture,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


CLASSES = ("dppp", "pppp")
SUPPORTED_CLASSES = (*CLASSES, "dpds", "ddps", "dpdp", "ddpp")
VARIANTS = ("cartesian", "coincident", "reversed_pairs")
MASKS = ("full", "partial", "one", "two", "zero")
TASK_MASKS = (0, 4, 1, 2, 3, 0, 4)
CONSUMERS = ("rhf", "raw-k")


def raw_exchange_reference(
    integrals: ArrayLike, density: ArrayLike, offsets: tuple[int, ...]
) -> np.ndarray:
    """Contract positive K from independent ERIs, including unique symmetries.

    The combined RHF J-K/2 reference cannot independently qualify the raw-K
    consumer. Do not reuse the generated scatter or its multiplicity logic.
    """
    integrals = np.asarray(integrals)
    density = np.asarray(density)
    exchange = np.zeros_like(density)
    for component in np.ndindex(integrals.shape):
        indices = tuple(
            offset + index for offset, index in zip(offsets, component, strict=True)
        )
        for first, second, third, fourth in eri_orbit(indices):
            exchange[first, third] += density[second, fourth] * integrals[component]
    return exchange


def fixture_plan(name: str, *, shared: bool) -> FusedShellPlan:
    """Pin a component mapping without declaring production qualification."""
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK,),
        recurrence=f"rys{sum(spec.angular) // 2 + 1}",
        target=cuda_target_info("sm_90"),
    )
    return build_fused_shell_plan(
        plan.spec,
        integral=plan.kernel.integral,
        target=plan.kernel.target,
        schedule=replace(plan.schedule, fock_shared_rys_xy=shared),
    )


def _array(name: str, values: ArrayLike, kind: str = "double") -> str:
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite fixture array {name}")
    tokens = (
        format(float(v), ".17g") if kind == "double" else str(int(v))
        for v in values.ravel()
    )
    return f"static const {kind} {name}[] = {{{','.join(tokens)}}};\n"


def fixture_inputs(
    name: str, variant: str, *, consumer: str = "rhf"
) -> tuple[dict, str]:
    """Mask Libcint's full block before the independent symmetry contraction."""
    if consumer not in CONSUMERS:
        raise ValueError(f"unsupported shared Rys fixture consumer: {consumer}")
    fixture = make_fixture(name, variant)
    _, _, block, _ = _reference_integrals(fixture.inputs)
    if not np.all(np.isfinite(block)) or not np.all(np.isfinite(fixture.density)):
        raise ValueError("independent fixture must contain finite values and density")
    n = fixture.density.shape[0]
    bounds, expected, live_counts, retained_producers = [], [], [], []
    for mask in MASKS:
        schwarz = np.zeros((n, n))
        retained = np.zeros(block.shape, dtype=bool)
        for index in np.ndindex(block.shape):
            first_component = index[0]
            keep = (
                mask == "full"
                or (mask == "partial" and first_component == 1)
                or (
                    mask in ("one", "two")
                    and np.ravel_multi_index(index, block.shape)
                    < (1 if mask == "one" else 2)
                )
            )
            if keep:
                i, j, k, l = (
                    offset + value
                    for offset, value in zip(fixture.ao_offsets, index, strict=True)
                )
                schwarz[i, j] = schwarz[k, l] = 1.0
        # Screening is a product of pair bounds, not a hand-coded lane mask.
        for index in np.ndindex(block.shape):
            i, j, k, l = (
                offset + value
                for offset, value in zip(fixture.ao_offsets, index, strict=True)
            )
            retained[index] = schwarz[i, j] * schwarz[k, l] >= 0.5
        live_counts.append(int(np.count_nonzero(retained)))
        retained_producers.append(
            [
                int(lane)
                for lane in np.flatnonzero(retained.ravel())
                if lane
                < 2 * fixture_plan(name, shared=True).kernel.integral.required_rys_roots
            ]
        )
        screened = np.where(retained, block, 0.0)
        reference = (
            raw_exchange_reference(screened, fixture.density, fixture.ao_offsets)
            if consumer == "raw-k"
            else contract_reference(
                screened,
                np.zeros((4, 3, *block.shape)),
                fixture.density,
                fixture.spin_density,
                fixture.ao_offsets,
                fixture.atom_indices,
            )["rhf_fock"]
        )
        bounds.append(schwarz.ravel(order="F"))
        expected.append(reference.ravel(order="F") * TASK_MASKS.count(len(expected)))
    tag = f"{name}_{variant}"
    arrays = "".join(
        (
            _array(f"{tag}_pairs", fixture.pairs),
            _array(f"{tag}_coefficients", fixture.ao_coefficients),
            _array(f"{tag}_positions", fixture.positions),
            _array(f"{tag}_bounds", bounds),
            _array(
                f"{tag}_density", np.tile(fixture.density.ravel(order="F"), len(MASKS))
            ),
            _array(f"{tag}_expected", expected),
            _array(f"{tag}_ao_offsets", fixture.ao_offsets, "std::uint64_t"),
            _array(f"{tag}_atoms", fixture.atom_indices, "std::uint32_t"),
            _array(
                f"{tag}_primitive_counts",
                [len(shell["primitives"]) for shell in fixture.inputs["shells"]],
                "std::uint64_t",
            ),
        )
    )
    metadata = {
        "name": name,
        "variant": variant,
        "input_hash": fixture.inputs_hash,
        "n": n,
        "pair_count": len(fixture.pairs),
        "pair_split": fixture.pair_split,
        "reversed_mask": fixture.reversed_mask | (8 if consumer == "raw-k" else 0),
        "retained_components": live_counts,
        "retained_producer_lanes": retained_producers,
        "primitive_quartets_per_task": fixture.pair_split
        * (len(fixture.pairs) - fixture.pair_split),
    }
    return metadata, arrays


_RUNTIME = r"""
#define CUDA(call) do { cudaError_t status=(call); if(status!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %s: %s\n",#call,cudaGetErrorString(status)); std::exit(2); } } while(false)

template<class T> T* upload(const T* data,std::size_t count) {
  T* device=nullptr; CUDA(cudaMalloc(&device,count*sizeof(T)));
  CUDA(cudaMemcpy(device,data,count*sizeof(T),cudaMemcpyHostToDevice)); return device;
}

template<class Kernel> void resources(const char* name,Kernel kernel) {
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,kernel));
  std::printf("RESOURCE %s registers=%d local_bytes=%zu shared_bytes=%zu max_threads=%d\n",
              name,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,a.maxThreadsPerBlock);
}

template<class Launch> std::vector<double> execute(const char* label,Launch launch,
    double* output,std::uint32_t* head,const double* expected,std::size_t count,
    cudaStream_t stream,bool graph) {
  auto enqueue=[&]() {
    CUDA(cudaMemsetAsync(output,0,count*sizeof(double),stream));
    CUDA(cudaMemsetAsync(head,0,sizeof(std::uint32_t),stream));
    launch(); CUDA(cudaPeekAtLastError());
  };
  cudaGraph_t captured=nullptr; cudaGraphExec_t executable=nullptr;
  if(graph) {
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    enqueue(); CUDA(cudaStreamEndCapture(stream,&captured));
    CUDA(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0));
  }
  std::vector<double> last;
  // A second replay must reset output and the persistent queue head. Within
  // each launch one CTA traverses full -> zero -> partial -> one -> two -> full
  // -> zero tasks, reusing shared storage after four signed primitive quartets.
  for(unsigned repeat=0;repeat<2;++repeat) {
    if(graph) CUDA(cudaGraphLaunch(executable,stream)); else enqueue();
    CUDA(cudaStreamSynchronize(stream));
    std::vector<double> actual(count);
    CUDA(cudaMemcpy(actual.data(),output,count*sizeof(double),cudaMemcpyDeviceToHost));
    double maximum=0.0;
    for(std::size_t i=0;i<count;++i) {
      const double error=std::abs(actual[i]-expected[i]);
      maximum=std::max(maximum,error);
      if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
          error>2e-11+2e-10*std::abs(expected[i])) {
        std::fprintf(stderr,"FAIL %s graph=%d replay=%u entry=%zu actual=%.17g expected=%.17g\n",
                     label,graph,repeat,i,actual[i],expected[i]); std::exit(3);
      }
    }
    std::printf("PASS %s graph=%d replay=%u max_abs=%.17g\n",label,graph,repeat,maximum);
    last=std::move(actual);
  }
  if(graph) { CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(captured)); }
  return last;
}
"""


def _case_runner(metadata: dict) -> str:
    name, variant = metadata["name"], metadata["variant"]
    tag = f"{name}_{variant}"
    class_name = name.capitalize()
    n, pair_count, split = (metadata[key] for key in ("n", "pair_count", "pair_split"))
    launches = []
    for shared in (False, True):
        mode = "shared" if shared else "private"
        prefix = f"{mode}_{name}"
        ctype = f"{mode.capitalize()}{class_name}"
        launches.append(f'''
  const auto {mode}_output=execute("{tag}_{mode}",[&]() {{
    {prefix}_shell_class_fock_rhf_persistent_kernel<<<1,{fixture_plan(name, shared=shared).block_threads},0,stream>>>(
      reinterpret_cast<const {ctype}ShellTask*>(tasks),
      reinterpret_cast<const {ctype}PrimitivePairData*>(pairs),offsets,coefficients,
      reinterpret_cast<const {ctype}Vec3*>(positions),0.5,bounds,density,
      generativeqc::runtime::CompensatedOutput(output),task_offset,count,head);
  }},output,head,{tag}_expected,{5 * n * n},stream,graph);
''')
    return f"""
void run_{tag}(cudaStream_t stream,bool graph) {{
  using Task=Private{class_name}ShellTask;
  static_assert(sizeof(Task)==sizeof(Shared{class_name}ShellTask));
  const unsigned masks[]={{{",".join(map(str, TASK_MASKS))}}};
  Task host_tasks[7]{{}};
  for(unsigned t=0;t<7;++t) {{
    Task& task=host_tasks[t];
    for(unsigned c=0;c<4;++c) {{
      task.primitive_end[c]={tag}_primitive_counts[c];
      task.ao_begin[c]=task.ao_coefficient_begin[c]={tag}_ao_offsets[c];
      task.atom[c]={tag}_atoms[c]; task.shell[c]=c;
    }}
    task.matrix_order={n}; task.shell_pair[1]=1;
    task.density_offset=masks[t]*{n * n};
    task.reversed_shell_pair_mask={metadata["reversed_mask"]};
  }}
  auto* tasks=upload(host_tasks,7);
  const std::int64_t host_offsets[]={{0,{split},{pair_count}}};
  auto* offsets=upload(host_offsets,3);
  auto* pairs=upload({tag}_pairs,{pair_count * 8});
  auto* coefficients=upload({tag}_coefficients,{n});
  auto* positions=upload({tag}_positions,12);
  auto* bounds=upload({tag}_bounds,{5 * n * n});
  auto* density=upload({tag}_density,{5 * n * n});
  const std::uint32_t zero=0,seven=7;
  auto* task_offset=upload(&zero,1); auto* count=upload(&seven,1); auto* head=upload(&zero,1);
  double* output=nullptr; CUDA(cudaMalloc(&output,{5 * n * n}*sizeof(double)));
{"".join(launches)}
  for(std::size_t i=0;i<private_output.size();++i) {{
    if(std::abs(private_output[i]-shared_output[i])>
        2e-11+2e-10*std::abs(private_output[i])) {{
      std::fprintf(stderr,"FAIL {tag} candidate/private entry=%zu\\n",i); std::exit(4);
    }}
  }}
  for(void* pointer:std::vector<void*>{{tasks,offsets,pairs,coefficients,positions,
      bounds,density,task_offset,count,head,output}}) CUDA(cudaFree(pointer));
}}
"""


def prepare_fixture(
    directory: Path,
    *,
    classes: tuple[str, ...] = CLASSES,
    consumer: str = "rhf",
) -> dict:
    """Write explicit synthetic populations without changing production defaults."""
    if consumer not in CONSUMERS:
        raise ValueError(f"unsupported shared Rys fixture consumer: {consumer}")
    if (
        not classes
        or len(set(classes)) != len(classes)
        or set(classes) - set(SUPPORTED_CLASSES)
    ):
        raise ValueError("shared Rys fixture requires distinct supported classes")
    directory.mkdir(parents=True, exist_ok=True)
    source = [
        _PRODUCTION_PRELUDE,
        "\n#include <algorithm>\n#include <cstdio>\n#include <cstdlib>\n#include <utility>\n#include <vector>\n",
    ]
    schedules = {}
    for name in classes:
        for shared in (False, True):
            plan = fixture_plan(name, shared=shared)
            mode = "shared" if shared else "private"
            body = emit_shell_class_fused_cuda(plan.spec, plan)
            # Only generated identifier roots are scoped. Explicit fixture
            # classes never acquire a production capability through this path.
            body = body.replace(f"generated_{name}", f"{mode}_{name}")
            body = body.replace(
                f"Generated{name.capitalize()}",
                f"{mode.capitalize()}{name.capitalize()}",
            )
            source.append(body)
            schedules[f"{name}_{mode}"] = schedule_payload(plan.schedule)
    cases = []
    for name in classes:
        for variant in VARIANTS:
            metadata, arrays = fixture_inputs(name, variant, consumer=consumer)
            cases.append(metadata)
            source.append(arrays)
    source.append(_RUNTIME)
    source.extend(_case_runner(case) for case in cases)
    source.append(
        "\nint main() {\n  cudaStream_t stream; CUDA(cudaStreamCreate(&stream));\n"
    )
    for name in classes:
        for mode in ("private", "shared"):
            symbol = f"{mode}_{name}_shell_class_fock_rhf_persistent_kernel"
            source.append(f'  resources("{symbol}",{symbol});\n')
    source.append("  for(bool graph:{false,true}) {\n")
    source.extend(
        f"    run_{case['name']}_{case['variant']}(stream,graph);\n" for case in cases
    )
    source.append(
        '  }\n  CUDA(cudaStreamDestroy(stream));\n  std::puts("ALL SYNTHETIC GATES PASS");\n}\n'
    )
    contents = "".join(source)
    path = directory / "shared_rys_xy_fixture.cu"
    path.write_text(contents)
    receipt = {
        "scope": "synthetic independent Libcint class Fock correctness only; not HF96 replay",
        "consumer": consumer,
        "source_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "schedules": schedules,
        "cases": cases,
        "masks": MASKS,
        "task_mask_sequence": TASK_MASKS,
        "expected_extra_shared_bytes": {
            name: 2
            * fixture_plan(name, shared=True).kernel.integral.required_rys_roots
            * (sum(FUSED_SHELL_SPEC_BY_NAME[name].angular[:2]) + 1)
            * (sum(FUSED_SHELL_SPEC_BY_NAME[name].angular[2:]) + 1)
            * 8
            for name in classes
        },
        "fock_tolerances": {"atol": 2e-11, "rtol": 2e-10},
        "cuda_status": "not compiled or executed by source preparation",
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer", choices=CONSUMERS, default="rhf")
    parser.add_argument(
        "--classes", nargs="+", choices=SUPPORTED_CLASSES, default=CLASSES
    )
    args = parser.parse_args()
    receipt = prepare_fixture(
        args.output, classes=tuple(args.classes), consumer=args.consumer
    )
    print(
        json.dumps(
            {
                "source_sha256": receipt["source_sha256"],
                "cuda_status": receipt["cuda_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
