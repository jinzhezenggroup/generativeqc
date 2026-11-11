"""Keep host DF probes on the production convergence-policy owner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def df_convergence_owner() -> str:
    """Extract the complete host policy, leaving only numerical work injected."""
    policy = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    policy_start = policy.index("HfConvergenceRequest parse_hf_convergence_request(")
    policy_end = policy.index("RhfBlockConvergenceRequest parse_", policy_start)
    rhf = (ROOT / "src/scf/rhf.cpp").read_text()
    owner_start = rhf.index("unsigned cuda_df_iteration_diis_history(")
    owner_end = rhf.index(
        "\n/** Preserve a complete final occupied projection", owner_start
    )
    return (
        "#include <array>\n#include <cstdlib>\n#include <cstring>\n#include <stdexcept>\n"
        '#include "scf/convergence_limits.hpp"\n'
        '#include "scf/cuda/rhf_policy.hpp"\n'
        '#include "scf/mean_field.hpp"\n'
        "namespace generativeqc::scf { namespace cuda_policy {\n"
        + policy[policy_start:policy_end]
        + "}\n"
        + rhf[owner_start:owner_end]
        + "}\n"
    )
