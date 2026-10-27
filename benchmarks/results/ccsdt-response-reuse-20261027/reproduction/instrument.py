"""One-off NVTX-only overlay for the #2516 source-matched redundancy audit.

Every wrapper preserves the original expression and return value. No CUDA
synchronization, numeric operation, buffer, or scheduling option is added.
The overlay is diagnostic-only and is not used for clean endpoint timings.
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])
header = """#include <nvtx3/nvToolsExt.h>
#include <string>
namespace audit2516 {
struct Scope {
  explicit Scope(const char* name) { nvtxRangePushA(name); }
  ~Scope() { nvtxRangePop(); }
};
template<class F> decltype(auto) call(const std::string& label, F&& f) {
  Scope scope(label.c_str());
  return f();
}
}
"""
path = root / "src/cc/df_triples_cuda.cu"
source = path.read_text()
head, joint = source.split("DFCudaCombinedResponseResult pullback_and_fock_df_cuda(", 1)


def wrap(name: str, label: str) -> None:
    """Annotate one native submission without changing its arguments or order."""
    global joint
    start = joint.index(name + "(")
    end = joint.index(";", start)
    original = joint[start:end]
    joint = (
        joint[:start] + f"audit2516::call({label}, [&] {{ {original}; }})" + joint[end:]
    )


wrap("generated_df::build_panel", '"2516/panel/" + std::to_string(occupied)')
wrap(
    "generated_df::build_w",
    '"2516/w/" + std::to_string(occupied[order[0]]) + "/" + std::to_string(occupied[order[1]]) + "/" + std::to_string(occupied[order[2]])',
)
for name, label in [
    ("pullback_w", "reverse_w"),
    ("pullback_v", "reverse_v"),
    ("pullback_panel", "reverse_panel"),
    ("fused_response_tile", "scalar"),
    ("resolvent_tile", "resolvent"),
    ("fock_vv", "fock_vv"),
    ("fock_oo", "fock_oo"),
]:
    wrap("generated_df::" + name, '"2516/' + label + '"')
joint = joint.replace(
    "  const auto started = Clock::now();",
    '  audit2516::Scope audit_joint("2516/joint");\n  const auto started = Clock::now();',
    1,
)
path.write_text(
    header + head + "DFCudaCombinedResponseResult pullback_and_fock_df_cuda(" + joint
)
path = root / "src/cc/df_lambda_cuda.cu"
source = path.read_text()
needle = "std::vector<double> DFLambdaActions::parameter(std::string_view name, std::size_t count) {"
assert source.count(needle) == 1
source = source.replace(
    needle,
    needle
    + '\n  const auto audit_label = "2516/parameter/" + std::string(name);\n  audit2516::Scope audit_parameter(audit_label.c_str());',
)
needle = "std::pair<std::vector<double>, std::vector<double>> DFLambdaActions::virtual_factors() {"
assert source.count(needle) == 1
source = source.replace(
    needle, needle + '\n  audit2516::Scope audit_factors("2516/factors");'
)
path.write_text(header + source)
