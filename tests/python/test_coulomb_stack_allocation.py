"""An optional recurrence-stack reservation must share the allocation fallback."""

import subprocess
from pathlib import Path

import pytest
from test_coulomb_optional_allocation import DRIVER, ROOT, STUBS


@pytest.fixture(scope="module")
def stack_probe(tmp_path_factory: pytest.TempPathFactory, native_cxx: object) -> Path:
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    begin = source.index("void configure_direct_coulomb_recurrence(")
    end = source.index("GeneratedExchangePlan::~GeneratedExchangePlan()", begin)
    preparation = source[begin:end]
    get_limit = "*n=100000; return cudaSuccess;"
    set_limit = (
        "cudaError_t cudaDeviceSetLimit(int,std::size_t) { return cudaSuccess; }"
    )
    assert STUBS.count(get_limit) == STUBS.count(set_limit) == 1
    stubs = STUBS.replace(get_limit, "*n=0; return cudaSuccess;")
    stubs = stubs.replace(
        set_limit,
        """cudaError_t cudaDeviceSetLimit(int,std::size_t) {
  if (injected_stage == 6) {
    if (injected_kind == 1) return cudaErrorMemoryAllocation;
    if (injected_kind == 2) return cudaErrorUnknown;
    fault(6);
  }
  return cudaSuccess;
}""",
    )
    directory = tmp_path_factory.mktemp("coulomb-stack-allocation")
    cpp, binary = directory / "probe.cpp", directory / "probe"
    cpp.write_text(stubs + preparation + DRIVER)
    native_cxx.build_executable(
        [cpp],
        binary,
        compile_args=("-std=c++17", "-I", str(ROOT / "src")),
        compile_timeout=30,
        link_timeout=30,
    )
    return binary


@pytest.mark.parametrize("kind", [1, 2, 3], ids=["oom", "cuda-error", "logic-error"])
@pytest.mark.parametrize("reachable", range(4))
@pytest.mark.parametrize("hermite", range(4))
@pytest.mark.parametrize("pair_materialized", [False, True])
@pytest.mark.parametrize("pair_derivatives", [False, True])
def test_stack_reservation_oom_falls_back_but_other_failures_propagate(
    stack_probe: Path,
    kind: int,
    reachable: int,
    hermite: int,
    pair_materialized: bool,
    pair_derivatives: bool,
) -> None:
    # The shared driver also checks stream fencing, no leaked allocations,
    # and successful fresh preparation after each rejected optional owner.
    subprocess.run(
        [
            str(stack_probe),
            "6",
            str(kind),
            str(int(reachable)),
            str(hermite),
            str(int(pair_materialized)),
            str(int(pair_derivatives)),
        ],
        check=True,
        timeout=10,
    )
