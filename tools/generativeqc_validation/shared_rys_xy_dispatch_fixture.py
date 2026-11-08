"""Reuse frozen mask/oracle arrays to test the built production dispatcher.

Preparation reads the already qualified TU and receipt; it never regenerates
Libcint data, loads a native library, or executes CUDA. The executable links the
new native core and tests ordinary plus repeated captured dispatch under frozen
host-policy choices. It is a correctness gate, never an endpoint timing vote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tools.generativeqc_validation.shared_rys_xy_fixture import _RUNTIME, _case_runner


def prepare_dispatch_fixture(frozen: Path, output: Path) -> dict:
    """Bind all copied input arrays to their qualified immutable source."""
    original = (frozen / "shared_rys_xy_fixture.cu").read_bytes()
    receipt_bytes = (frozen / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    source_hash = hashlib.sha256(original).hexdigest()
    if source_hash != receipt["source_sha256"]:
        raise ValueError("Frozen fixture source does not match its receipt")
    arrays = re.findall(r"^static const .*?;\n", original.decode(), re.MULTILINE)
    if len(arrays) != 9 * 6 or receipt["masks"] != [
        "full",
        "partial",
        "one",
        "two",
        "zero",
    ]:
        raise ValueError("Incomplete frozen mask/independent-oracle inventory")
    source = [
        r"""
#include "scf/aot_shell_registry.hpp"
#include "scf/generated_shell_task.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "runtime/cuda_device_facts.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cassert>
using PrivateDpppShellTask = generativeqc::scf::detail::GeneratedShellTask;
using SharedDpppShellTask = PrivateDpppShellTask;
using PrivatePpppShellTask = PrivateDpppShellTask;
using SharedPpppShellTask = PrivateDpppShellTask;
""",
        *arrays,
        _RUNTIME,
    ]
    for metadata in receipt["cases"]:
        name = metadata["name"]
        shell_class = 12 if name == "dppp" else 5
        runner = _case_runner(metadata).replace(
            "cudaStream_t stream,bool graph) {",
            "cudaStream_t stream,bool graph,const generativeqc::runtime::CudaTargetInfo& target) {",
        )
        for mode, value in (("private", "0"), ("shared", "1")):
            start = runner.index(
                f"    {mode}_{name}_shell_class_fock_rhf_persistent_kernel<<<"
            )
            end = runner.index("\n  },output", start)
            replacement = f'''    using namespace generativeqc::scf;
    auto request = cuda_policy::parse_shared_rys_xy_request("{value}");
    cuda_policy::SharedRysXyContext context{{}};
    context.quartet_direct=true;
    context.capability_mask=generated::enabled_shared_rys_xy_fock_shell_class_mask();
    const auto frozen=cuda_policy::resolve_shared_rys_xy_policy(request,context,target);
    assert(frozen.active()=={str(value == "1").lower()});
    auto changed=cuda_policy::resolve_shared_rys_xy_policy(
        cuda_policy::parse_shared_rys_xy_request("{int(value) ^ 1}"),context,target);
    assert(changed!=frozen); // a different choice cannot reuse this graph plan
    auto launch=frozen.active()?generated::launch_shell_class_shared_rys_xy_fock
                               :generated::launch_shell_class_fock;
    CUDA(launch({shell_class},stream,false,1,tasks,task_offset,offsets,pairs,
        coefficients,positions,0.5,bounds,density,output,count,head));'''
            runner = runner[:start] + replacement + runner[end:]
        source.append(runner)
    source.append("""
int main() {
  int device=0,major=0,minor=0;
  CUDA(cudaGetDevice(&device));
  generativeqc::runtime::CudaTargetInfo target{};char name[256]{};
  CUDA(generativeqc::runtime::cuda_device_facts(device,target,name));
  CUDA(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
  CUDA(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
  assert(major==9 && minor==0);
  generativeqc::scf::generated::select_profile_for_device(device,major,minor);
  assert(generativeqc::scf::generated::enabled_shared_rys_xy_fock_shell_class_mask()
         ==((1ULL<<12)|(1ULL<<5)));
  cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
  for(bool graph:{false,true}) {
""")
    source.extend(
        f"    run_{c['name']}_{c['variant']}(stream,graph,target);\n"
        for c in receipt["cases"]
    )
    source.append("""  }
  CUDA(cudaStreamDestroy(stream));
  std::puts("PRODUCTION DISPATCH ORDINARY+GRAPH+CONTROL-IDENTITY GATES PASS");
}
""")
    content = "".join(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "shared_rys_xy_dispatch_fixture.cu").write_text(content)
    binding = {
        "scope": "new production dispatcher correctness only; no endpoint timing",
        "qualified_fixture_source_sha256": source_hash,
        "qualified_fixture_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "cases": receipt["cases"],
        "masks": receipt["masks"],
        "task_mask_sequence": receipt["task_mask_sequence"],
        "fock_tolerances": receipt["fock_tolerances"],
        "expected_shared_bytes": {"dppp": 1064, "pppp": 920},
        "expected_registers": 166,
        "expected_local_bytes": {"dppp": 96, "pppp": 72},
        "expected_spill_loads_stores": 0,
        "cuda_status": "UNRUN_REQUIRES_REVIEW_AND_PARENT_SCHEDULING",
    }
    (output / "receipt.json").write_text(json.dumps(binding, indent=2) + "\n")
    return binding


def main() -> None:
    """Write the source-bound fixture without building or executing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_dispatch_fixture(args.frozen_fixture, args.output)))


if __name__ == "__main__":
    main()
