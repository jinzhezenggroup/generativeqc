"""Host qualification of bounded page planning and the real launch composition.

CUDA math/concurrency needs the independent native device gates. These tests
exercise generated admission, disjoint domains and production host error paths;
they do not treat emulated launch counts as GPU performance evidence.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_force_pages import (
    emit_direct_force_page_header,
    force_page_consumers,
)
from test_direct_shell_derivative_host_lifetime import _extract_function

ROOT = Path(__file__).resolve().parents[2]


def compile_probe(folder: Path, source: str) -> Path:
    """Compile a CPU-only capsule of generated policy/production control flow."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cache = shutil.which("ccache")
    if cache is None:
        pytest.skip("host C++ qualification requires ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    header = emit_direct_force_page_header()
    (folder / "generated_direct_force_pages.hpp").write_text(header)
    program, executable = folder / "probe.cpp", folder / "probe"
    program.write_text(source)
    result = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-I",
            str(folder),
            "-I",
            str(ROOT / "src"),
            str(program),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return executable


def test_class_schedule_and_budget_admission(tmp_path: Path) -> None:
    consumers = force_page_consumers()
    assert len(consumers) == 55
    assert {item.shell_class for item in consumers} == set(range(55))
    assert emit_direct_force_page_header() == emit_direct_force_page_header()
    binary = compile_probe(
        tmp_path,
        r"""
#include <cassert>
#include <limits>
#include "generated_direct_force_pages.hpp"
#include "scf/direct_task_layout.hpp"
using namespace generativeqc::scf::cuda_execution;
int main() {
  using namespace generativeqc::scf::detail;
#define VERIFY_CLASS(shell, order, threads, tasks_per_warp) \
  static_assert(direct_quartet_shell_class_angular_order(shell) == order); \
  static_assert(threads == 256 && (tasks_per_warp == 32) == (order <= 3));
  GENERATIVEQC_FOR_EACH_FORCE_PAGE_CLASS(VERIFY_CLASS)
#undef VERIFY_CLASS
  const auto huge = std::numeric_limits<std::size_t>::max();
  for (auto products : {std::size_t{0},1UL,3UL,4095UL,4096UL,4097UL,huge}) {
    for (auto budget : {std::size_t{0},1UL,26483UL,26484UL,1000000UL,huge}) {
      const auto layout = plan_force_page(products, budget);
      assert(layout.bytes <= budget);
      if (!layout.blocks) { assert(!layout.candidates && !layout.bytes); continue; }
      assert(layout.blocks <= products && layout.blocks <= 4096);
      assert(layout.candidates == 1024 * layout.blocks);
      assert(layout.candidates < std::numeric_limits<std::uint32_t>::max());
      assert(layout.input == 0 && layout.tasks == 12 * layout.candidates);
      assert(layout.classes == 24 * layout.candidates);
      assert(layout.counts == 25 * layout.candidates);
      assert(layout.offsets == layout.counts + 4 * 55);
      assert(layout.writes == layout.offsets + 4 * 56);
      assert(layout.heads == layout.writes + 4 * 55);
      assert(layout.bytes == layout.heads + 4 * 55);
      assert(layout.counts % 4 == 0);
      if (products < 10000) {
        std::size_t covered=0;
        while (covered < products) covered += std::min(layout.blocks, products-covered);
        assert(covered == products);
      }
    }
  }
}
""",
    )
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.fixture(scope="module")
def launch_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    body = _extract_function(
        source, "static cudaError_t enqueue_compact_full_range_force("
    )
    folder = tmp_path_factory.mktemp("force-page-launches")
    return compile_probe(folder, LAUNCH_PREFIX + body + LAUNCH_DRIVER)


@pytest.mark.parametrize("failure", range(49))
@pytest.mark.parametrize("throws", [False, True])
def test_disjoint_pages_and_every_launch_error(
    launch_probe: Path, failure: int, throws: bool
) -> None:
    subprocess.run(
        [str(launch_probe), str(failure), str(int(throws))], check=True, timeout=10
    )


LAUNCH_PREFIX = r"""
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "generated_direct_force_pages.hpp"
using namespace generativeqc::scf::cuda_execution;
using cudaError_t=int;
constexpr int cudaSuccess=0;
int operations=0, failure=0; bool throws=false;
int operation() {
  if (++operations != failure) return 0;
  if (throws) throw std::runtime_error("injected");
  return 7;
}
// On Darwin uint64_t and size_t have equal widths but distinct underlying types.
struct Domain { const unsigned* prefix{}; unsigned long long quartet_count{}; };
struct Batch { std::size_t total_shell_pair_block_quartets=7; };
struct BoundedForcePage {
  std::size_t block_capacity=2;
  unsigned *counts{},*offsets{},*writes{},*heads{};
  unsigned char *classes{}; void *input{},*tasks{};
};
struct Shared {
  unsigned worker_blocks=3,stream=1; Batch batch;
  double screening=0.5;
  double *shell_bounds{},*schwarz{}; unsigned char* active{};
};
struct GeneratedExchangePlan {
  Shared* shared;
  BoundedForcePage force_page;
  Domain bounded_block_domain;
  void *shell_pair_density_bounds{},*bounded_pair_order{},*shell_pair_block_bounds{},
       *system_density_bounds{},*direct_spin{},*force{};
  unsigned long long *force_cursor{};
  std::uint64_t force_page_class_mask=(1ULL<<4)|(1ULL<<12);
  std::size_t last_force_page_count=0;
};
std::size_t next_begin=0, page_slots=0;
unsigned stage=0, launches=0;
int cudaMemsetAsync(void* pointer,int value,std::size_t bytes,unsigned) {
  const auto status=operation(); if (!status) std::memset(pointer,value,bytes); return status;
}
int cudaGetLastError() { return operation(); }
void launch_classify_bounded_force_page(
    bool,unsigned,unsigned,Batch,double,void*,void*,void*,void*,void*,void*,
    Domain,std::size_t begin,std::size_t end,unsigned long long*,BoundedForcePage page) {
  assert(stage==0 && begin==next_begin && end>begin && end-begin<=page.block_capacity);
  next_begin=end; page_slots=(end-begin)*kForcePageBlockCandidates;
  for(std::size_t slot=0;slot<page_slots;++slot) assert(page.classes[slot]==255);
  assert(page.counts[4]==0 && page.counts[12]==0);
  stage=1;
}
void launch_prefix_generated_shell_task_counts_kernel(
    unsigned,unsigned,unsigned,unsigned,void*,void*,void*,void*) {
  assert(stage==1); stage=2;
}
void launch_materialize_compact_force_tiles(
    unsigned blocks,unsigned threads,unsigned,std::size_t capacity,void*,void*,void*,void*,void*) {
  assert(stage==2 && capacity==page_slots && blocks*threads>=capacity); stage=3;
}
int launch_compact_force_class(bool,unsigned shell,unsigned,unsigned,Batch,double,
    void*,void*,void*,void*,double coulomb,double exchange,BoundedForcePage) {
  assert(stage==3 && coulomb==1 && exchange==-0.125);
  assert(shell == (launches % 2 ? 12 : 4));
  if (++launches % 2 == 0) stage=0;
  return operation();
}
"""

LAUNCH_DRIVER = r"""
int main(int argc,char** argv) {
  assert(argc==3); failure=std::stoi(argv[1]); throws=std::stoi(argv[2]);
  std::vector<unsigned> counts(55),offsets(56),writes(55),heads(55);
  std::vector<unsigned char> classes(2048);
  Shared shared; unsigned long long cursor=0;
  GeneratedExchangePlan owner{&shared};
  owner.force_page={2,counts.data(),offsets.data(),writes.data(),heads.data(),classes.data()};
  owner.force_cursor=&cursor;
  bool caught=false; int status=0;
  try { status=enqueue_compact_full_range_force(owner,false,1,-0.125); }
  catch(const std::runtime_error&) { caught=true; }
  const bool injected=failure!=0 && operations>=failure;
  if(injected) { assert(operations==failure); assert(throws ? caught : status==7); }
  else {
    assert(status==0 && !caught && next_begin==7 && owner.last_force_page_count==4);
    assert(launches==8 && stage==0);
  }
}
"""
