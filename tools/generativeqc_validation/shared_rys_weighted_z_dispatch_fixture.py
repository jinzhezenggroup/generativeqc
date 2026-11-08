"""Exercise new linked XY/weighted dispatch with the existing frozen arrays.

No references are regenerated and no native library is loaded by preparation.
Resource host-stub RVAs are supplied only after authenticating the newly built
core; old offsets or a new native resource-export ABI are not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tools.generativeqc_validation.shared_rys_xy_fixture import _RUNTIME, _case_runner


def _resource_source() -> str:
    return r"""
void linked_resources(const char* core_path,const char* const* values) {
  using namespace generativeqc::scf::generated;
  Dl_info anchor{};
  assert(dladdr(reinterpret_cast<const void*>(&launch_shell_class_fock),&anchor));
  assert(anchor.dli_fname && anchor.dli_fbase);
  assert(std::filesystem::canonical(anchor.dli_fname)==std::filesystem::canonical(core_path));
  std::printf("CORE_OWNER path=%s base=%p\n",anchor.dli_fname,anchor.dli_fbase);
  const char* names[]={"xy_dppp","xy_pppp","weighted_dppp","weighted_pppp"};
  int active[4]{};cudaFuncAttributes attributes[4]{};
  for(unsigned k=0;k<4;++k) {
    unsigned long long fields[4]{};
    for(unsigned j=0;j<4;++j) {
      const char* text=values[4*k+j];char* tail=nullptr;errno=0;
      assert(text && *text>='0' && *text<='9');
      fields[j]=std::strtoull(text,&tail,0);
      assert(!errno && tail && *tail=='\0');
    }
    const auto offset=fields[0],bytes=fields[1],begin=fields[2],end=fields[3];
    assert(offset>0 && bytes>0 && begin<end && offset>=begin);
    assert(offset<=std::numeric_limits<unsigned long long>::max()-bytes && offset+bytes<=end);
    const auto base=reinterpret_cast<std::uintptr_t>(anchor.dli_fbase);
    assert(end<=std::numeric_limits<std::uintptr_t>::max() && base<=std::numeric_limits<std::uintptr_t>::max()-end);
    const auto function=reinterpret_cast<const void*>(base+offset);
    Dl_info owner{};assert(dladdr(function,&owner) && owner.dli_fname);
    assert(owner.dli_fbase==anchor.dli_fbase &&
        std::filesystem::canonical(owner.dli_fname)==std::filesystem::canonical(core_path));
    // The authenticated executable PT_LOAD must also be executable in the
    // actual process map; never infer a load base from PLT/interposed symbols.
    bool mapped=false;std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line)) {
      unsigned long long low=0,high=0;char permissions[5]{};int consumed=0;
      if(std::sscanf(line.c_str(),"%llx-%llx %4s %*s %*s %*s %n",&low,&high,permissions,&consumed)!=3 || consumed<=0)continue;
      const auto path=line.substr(static_cast<std::size_t>(consumed));
      if(permissions[2]=='x' && low<=base+offset && base+offset+bytes<=high &&
          !path.empty() && path[0]=='/' && std::filesystem::canonical(path)==std::filesystem::canonical(core_path))mapped=true;
    }
    assert(mapped);
    CUDA(cudaFuncGetAttributes(&attributes[k],function));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active[k],function,k%2==0?192:96,0));
    std::printf("RESOURCE %s registers=%d local_bytes=%zu shared_bytes=%zu max_threads=%d active_blocks_per_sm=%d rva=%llu function_bytes=%llu exec_load_begin=%llu exec_load_end=%llu\n",
        names[k],attributes[k].numRegs,attributes[k].localSizeBytes,attributes[k].sharedSizeBytes,
        attributes[k].maxThreadsPerBlock,active[k],offset,bytes,begin,end);
    assert(attributes[k].maxThreadsPerBlock==(k%2==0?192:96) && active[k]>0);
  }
  for(unsigned k=0;k<2;++k) {
    assert(attributes[k+2].sharedSizeBytes<=2048);
    assert(attributes[k+2].localSizeBytes<=attributes[k].localSizeBytes);
    assert(active[k+2]>=active[k]);
  }
}
"""


def prepare_dispatch_fixture(frozen: Path, output: Path) -> dict:
    """Qualify actual linked dispatch, resource attributes and mode roundtrip."""
    original = (frozen / "shared_rys_xy_fixture.cu").read_bytes()
    old = json.loads((frozen / "receipt.json").read_text())
    if hashlib.sha256(original).hexdigest() != old["source_sha256"]:
        raise ValueError("Frozen independent source changed")
    arrays = re.findall(r"^static const .*?;\n", original.decode(), re.MULTILINE)
    if len(arrays) != 54:
        raise ValueError("Frozen array inventory incomplete")
    source = [
        r"""
#include "scf/aot_shell_registry.hpp"
#include "scf/shared_rys_weighted_z.hpp"
#include "scf/generated_shell_task.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <dlfcn.h>
#ifdef NDEBUG
#error "Linked dispatch qualification requires assertions enabled (-UNDEBUG)"
#endif
using PrivateDpppShellTask=generativeqc::scf::detail::GeneratedShellTask;
using SharedDpppShellTask=PrivateDpppShellTask;
using PrivatePpppShellTask=PrivateDpppShellTask;
using SharedPpppShellTask=PrivateDpppShellTask;
using Policy=generativeqc::scf::cuda_policy::SharedRysXyPolicy;
struct PreparedDispatch {
  cudaGraph_t graph{};cudaGraphExec_t executable{};Policy policy{};
  unsigned generations{},invalidations{};
};
""",
        *arrays,
    ]
    runtime = _RUNTIME
    runtime = runtime.replace(
        "cudaStream_t stream,bool graph) {",
        "cudaStream_t stream,bool graph,Policy policy,PreparedDispatch& prepared) {",
    )
    a = runtime.index("  cudaGraph_t captured=")
    b = runtime.index("  std::vector<double> last;", a)
    runtime = (
        runtime[:a]
        + r"""  if(graph) {
    // Same immutable native-policy equality used by RHF's prepared-plan guard.
    if(prepared.executable && prepared.policy!=policy) {
      CUDA(cudaGraphExecDestroy(prepared.executable));CUDA(cudaGraphDestroy(prepared.graph));
      prepared.executable=nullptr;prepared.graph=nullptr;++prepared.invalidations;
    }
    if(!prepared.executable) {
      CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));enqueue();
      CUDA(cudaStreamEndCapture(stream,&prepared.graph));
      CUDA(cudaGraphInstantiate(&prepared.executable,prepared.graph,nullptr,nullptr,0));
      prepared.policy=policy;++prepared.generations;
    }
    assert(prepared.policy==policy);
  }
"""
        + runtime[b:]
    )
    runtime = runtime.replace(
        "cudaGraphLaunch(executable,stream)",
        "cudaGraphLaunch(prepared.executable,stream)",
    )
    runtime = runtime.replace(
        "  if(graph) { CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(captured)); }",
        "",
    )
    source += [runtime, _resource_source()]
    for metadata in old["cases"]:
        name = metadata["name"]
        cls = 12 if name == "dppp" else 5
        case = _case_runner(metadata)
        case = case.replace(
            "  const unsigned masks[]=",
            "  PreparedDispatch prepared;\n  const unsigned masks[]=",
        )
        for mode, weighted in (("private", False), ("shared", True)):
            start = case.index(
                f"    {mode}_{name}_shell_class_fock_rhf_persistent_kernel<<<"
            )
            end = case.index("\n  },output", start)
            launch = f"""    using namespace generativeqc::scf;
    auto launch={mode}_policy.weighted_z()?generated::launch_shell_class_shared_rys_weighted_z_fock
                                        :generated::launch_shell_class_shared_rys_xy_fock;
    CUDA(launch({cls},stream,false,1,tasks,task_offset,offsets,pairs,coefficients,
                positions,0.5,bounds,density,output,count,head));"""
            case = case[:start] + launch + case[end:]
            context = f'''  generativeqc::scf::cuda_policy::SharedRysXyContext {mode}_context{{}};
  {mode}_context.profile_supported={mode}_context.quartet_direct=true;
  {mode}_context.capability_mask=generativeqc::scf::generated::enabled_shared_rys_{"weighted_z" if weighted else "xy"}_fock_shell_class_mask();
  const auto {mode}_policy=generativeqc::scf::cuda_policy::resolve_shared_rys_xy_policy(
      generativeqc::scf::cuda_policy::parse_shared_rys_xy_request("1","{int(weighted)}"),{mode}_context);
  assert({mode}_policy.active() && {mode}_policy.selects({cls}) && {mode}_policy.weighted_z()=={str(weighted).lower()});
'''
            marker = f"  const auto {mode}_output="
            case = case.replace(marker, context + marker, 1)
            case = case.replace(
                "stream,graph);", f"stream,graph,{mode}_policy,prepared);", 1
            )
        assert "private_policy,prepared" in case and "shared_policy,prepared" in case
        # A second XY invocation forces the reverse transition after weighted.
        a = case.index("  const auto private_output=")
        b = case.index("\n  },output", a)
        e = case.index(";", b) + 1
        back = (
            case[a:e]
            .replace("private_output=", "roundtrip_output=")
            .replace(
                '"' + name + "_" + metadata["variant"] + '_private"',
                '"' + name + "_" + metadata["variant"] + '_roundtrip_xy"',
            )
        )
        close = (
            r"""
  if(graph) {
"""
            + back
            + r"""
    assert(prepared.generations==3 && prepared.invalidations==2);
    CUDA(cudaGraphExecDestroy(prepared.executable));CUDA(cudaGraphDestroy(prepared.graph));
    std::printf("POLICY_ROUNDTRIP_PASS generations=%u invalidations=%u\n",prepared.generations,prepared.invalidations);
  }
"""
        )
        case = case.replace("  for(void* pointer:", close + "  for(void* pointer:", 1)
        source.append(case)
    source.append(r"""
int main(int argc,char** argv) {
  assert(argc==18); // core path plus4x(RVA,size,executable PT_LOAD begin/end)
  int device=0,major=0,minor=0;CUDA(cudaGetDevice(&device));
  CUDA(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
  CUDA(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
  assert(major==9 && minor==0);
  generativeqc::scf::generated::select_profile_for_device(device,major,minor);
  assert(generativeqc::scf::generated::enabled_shared_rys_xy_fock_shell_class_mask()==((1ULL<<12)|(1ULL<<5)));
  assert(generativeqc::scf::generated::enabled_shared_rys_weighted_z_fock_shell_class_mask()==((1ULL<<12)|(1ULL<<5)));
  linked_resources(argv[1],const_cast<const char* const*>(argv+2));
  cudaStream_t stream;CUDA(cudaStreamCreate(&stream));
  for(bool graph:{false,true}) {
""")
    source += [
        f"    run_{c['name']}_{c['variant']}(stream,graph);\n" for c in old["cases"]
    ]
    source.append(
        '  }\nCUDA(cudaStreamDestroy(stream));std::puts("WEIGHTED PRODUCTION DISPATCH RESOURCE+ORDINARY+GRAPH+MODE-ROUNDTRIP GATES PASS");\n}\n'
    )
    text = "".join(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "shared_rys_weighted_z_dispatch_fixture.cu").write_text(text)
    result = {
        "status": "SOURCE_PREPARED_CUDA_UNRUN_REQUIRES_EXACT_REVIEW",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "frozen_source_sha256": old["source_sha256"],
        "unchanged_array_declarations": 54,
        "cases": old["cases"],
        "matrix_predicates": 60,
        "policy_roundtrip_predicates": 6,
        "resource_rows": 4,
        "argv_contract": ["CORE_CANONICAL_PATH"]
        + [
            f"{name}_{field}"
            for name in ("XY_DPPP", "XY_PPPP", "WEIGHTED_DPPP", "WEIGHTED_PPPP")
            for field in (
                "HOSTSTUB_RVA",
                "FUNC_SIZE",
                "EXEC_PT_LOAD_BEGIN",
                "EXEC_PT_LOAD_END",
            )
        ],
        "RVA_authority": "only source-name/ELF-bound exact newly linked core, not old offsets; root/Astra admission required",
        "strict_gates": {
            "atol": 2e-11,
            "rtol": 2e-10,
            "finite": True,
            "weighted_static_shared_max": 2048,
            "local_stack_max": "actual XY",
            "spills": "zero",
            "occupancy_min": "actual XY",
        },
        "new_chemical_references": 0,
    }
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_dispatch_fixture(args.frozen_fixture, args.output)))


if __name__ == "__main__":
    main()
