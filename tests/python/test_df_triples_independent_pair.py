"""Independent W grouping reuses the original typed scientific descriptors."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.cc.occupied_triples import moment_program
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter

from tools.generate_df_occupied_triples import _gemm, header, native_execution_header

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def emitted() -> str:
    """Generate the runtime-independent typed region once for all structure gates."""
    return native_execution_header()


def test_pairing_is_backend_owned_and_preserves_the_two_original_cuts(
    emitted: str,
) -> None:
    pair = emitted[emitted.index("bool build_w_pair(") : emitted.index(" private:")]
    assert pair.count("table.execute_independent_pair(") == 2
    assert "table.independent_pair_supported(0,o,v,q,context.stream)" in pair
    assert "table.independent_pair_supported(1,o,v,q,context.stream)" in pair
    assert pair.index("independent_pair_supported(1") < pair.index(
        "execute_independent_pair("
    )
    assert "occupied_second==occupied_third" in pair
    assert "second_output!=first_output+v3" in pair
    assert "in.t2+(occupied_third*o+occupied_second)*v*v" in pair
    assert "in.t2+(occupied_second*o+occupied_third)*v*v" in pair
    assert "in.ovoo+(occupied_first*v*o+occupied_second)*o" in pair
    assert "in.ovoo+(occupied_first*v*o+occupied_third)*o" in pair
    for implementation in (
        "cublasDgemmBatched",
        "__global__",
        "cudaMalloc",
        "cudaMemcpy",
    ):
        assert implementation not in emitted
    assert (
        "prepared-w-v3-independent-pairs"
        in (ROOT / "tools/generate_df_occupied_triples.py").read_text()
    )


def test_native_admission_preserves_capacity_and_complete_fallback_groups() -> None:
    owner = (ROOT / "src/cc/df_triples_cuda.cu").read_text()
    admission = owner[
        owner.index("const auto unpaired_layout = p;") : owner.index(
            "validate_inputs(o, v, q, p, host, threshold)"
        )
    ]
    assert "p.panel_capacity == 3" in admission
    assert "if (paired.total <= max_bytes) p = paired;" in admission
    assert "planned_layout(1)" not in admission
    preparation = owner[
        owner.index("std::optional<generativeqc_tensor::Context>") : owner.index(
            "generated_df::WExecution execution"
        )
    ]
    assert (
        "DeviceAllocationError" in preparation and "p = unpaired_layout;" in preparation
    )
    assert "std::array<std::size_t, 6> seed_permutations{};" in owner
    assert "seed_count == 2 && p.paired_pointer_bytes" in owner
    assert "result.moment_gemms += 4;" in owner
    response = owner[owner.index("static DFCudaResponseResult pullback_df_cuda_impl") :]
    assert "build_w_pair" not in response


@pytest.fixture
def first_cut() -> tuple[Any, Any, dict[str, tuple[str, str]], list[str]]:
    """Prepare one ordinary descriptor before attempting to group its views."""
    program = moment_program(2, 3)
    adapter = TensorLoweringAdapter(program)
    node = program.outputs["w"].inputs[0]
    views = {"panel": ("first_panel", "v"), "t2_kj": ("first_t2", "v")}
    descriptors: list[str] = []
    _gemm(node, views, "(1.0/1.0)", "0.0", adapter=adapter, descriptors=descriptors)
    return node, adapter, views, descriptors


def test_paired_projection_reuses_descriptor_without_appending(first_cut: Any) -> None:
    node, adapter, views, descriptors = first_cut
    before = descriptors.copy()
    emitted = _gemm(
        node,
        views,
        "(1.0/1.0)",
        "0.0",
        adapter=adapter,
        descriptors=descriptors,
        paired_bindings={"panel": ("second_panel", "v"), "t2_kj": ("second_t2", "v")},
        paired_output_pointer="second_output",
        paired_slot=0,
    )
    assert "execute_independent_pair(0," in emitted
    assert "first_panel,second_panel,first_t2,second_t2" in emitted
    assert descriptors == before


def test_actual_owner_grouping_preserves_every_seed_in_strict_and_fallback_domains(
    tmp_path: Path,
) -> None:
    """Compile the actual owner loop, including its six-entry fallback storage."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    owner = (ROOT / "src/cc/df_triples_cuda.cu").read_text()
    visit = owner[owner.index("auto visit_tile =") :]
    loop = visit[
        visit.index(
            "for (std::size_t index = 0; index < occupied.size(); ++index)"
        ) : visit.index("const double degeneracy")
    ]
    (tmp_path / "generated.hpp").write_text(header())
    source = tmp_path / "grouping.cpp"
    source.write_text(
        """#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "generated.hpp"
namespace generated_df=generativeqc::cc::triples::generated_df;
struct Context { unsigned char arena[64]{}; };
struct Inputs {};
struct Execution {
  double* moments;
  bool refuse_pairs;
  std::vector<std::array<std::size_t,4>> seeds;
  std::size_t pairs{};
  void build_w(Context&,const Inputs&,std::size_t first,std::size_t second,std::size_t third,
               const double*,double* output,std::size_t) {
    seeds.push_back({first,second,third,std::size_t(output-moments)});
  }
  bool build_w_pair(Context& context,const Inputs& inputs,std::size_t first,std::size_t second,
                    std::size_t third,const double* panel,double* output,double* other,void*) {
    if(refuse_pairs) return false;
    if(other!=output+1 || second==third) throw std::logic_error("invalid owner pair");
    build_w(context,inputs,first,second,third,panel,output,0);
    build_w(context,inputs,first,third,second,panel,other,0); ++pairs; return true;
  }
};
int main() {
  std::size_t cases=0;
  for(std::size_t count=1;count<=32;++count)
    for(bool strict:{false,true})
      for(bool admitted:{false,true})
        for(bool refusal:{false,true}) {
          std::size_t seeds=0,pairs=0;
          for(std::size_t first=0;first<count;++first)
            for(std::size_t second=0;second<=first;++second)
              for(std::size_t third=0;third<=second;++third) {
                const std::array<std::size_t,3> occupied{first,second,third};
                const auto sources=strict ? generated_df::occupied_moment_sources(first,second,third)
                                          : generated_df::MomentSourceMap{};
                struct { std::size_t v3=1,paired_pointer_bytes{},paired_pointers{}; } p;
                p.paired_pointer_bytes=strict && admitted ? 48 : 0;
                struct { std::size_t moment_gemms{}; } result;
                Context context; Inputs in;
                double panels[3]{},storage[6]{}; double* moments=storage;
                Execution execution{moments,refusal,{}};
                const auto panel_slot_for=[](std::size_t) { return std::size_t{0}; };
"""
        + loop
        + """
                std::array<bool,6> seen{};
                for(const auto& seed:execution.seeds) {
                  const auto index=seed[3];
                  if(index>=6 || seen[index]) throw std::logic_error("repeated seed");
                  seen[index]=true;
                  for(std::size_t axis=0;axis<3;++axis)
                    if(seed[axis]!=occupied[generated_df::permutations[index][axis]])
                      throw std::logic_error("wrong physical tuple");
                }
                for(std::size_t index=0;index<6;++index)
                  if(seen[index]!=(sources.index[index]==index)) throw std::logic_error("missing seed");
                if(result.moment_gemms!=2*execution.seeds.size()) throw std::logic_error("wrong work count");
                seeds+=execution.seeds.size(); pairs+=execution.pairs;
              }
          const auto tiles=count*(count+1)*(count+2)/6;
          if(seeds!=(strict ? count*count*count : 6*tiles)) throw std::logic_error("wrong total seeds");
          if(pairs!=(strict && admitted && !refusal ? count*count*(count-1)/2 : 0))
            throw std::logic_error("wrong total pairs");
          ++cases;
        }
  std::cout << cases << " source-matched grouping domains passed\\n";
}
"""
    )
    obj, binary = tmp_path / "grouping.o", tmp_path / "grouping"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)], check=True, capture_output=True
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "256 source-matched grouping domains passed"


def test_actual_owner_optional_allocation_retry_is_preexecution_and_typed(
    tmp_path: Path,
) -> None:
    """Inject allocation failures into the exact in-place preparation boundary."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    owner = (ROOT / "src/cc/df_triples_cuda.cu").read_text()
    preparation = owner[
        owner.index(
            "std::optional<generativeqc_tensor::Context> context_owner;"
        ) : owner.index("generated_df::WExecution execution")
    ]
    source = tmp_path / "allocation.cpp"
    source.write_text(
        """#include <cstddef>
#include <optional>
#include <stdexcept>
static int attempts{},live{},failure{};
namespace generativeqc_tensor {
struct DeviceAllocationError:std::runtime_error { using std::runtime_error::runtime_error; };
struct Context {
  std::size_t arena{};
  Context() { ++live; }
  ~Context() { --live; }
  void prepare(int,int,int,std::size_t bytes,std::size_t,std::size_t,int,int,bool) {
    ++attempts; arena=bytes;
    if(failure==3) throw std::runtime_error("nonallocation preparation error");
    if(failure==2 || (failure==1 && attempts==1)) throw DeviceAllocationError("injected OOM");
  }
};
}
static bool run(bool optional,int injected) {
  attempts=live=0; failure=injected;
  struct Layout { std::size_t arena{},error{},library{},paired_pointer_bytes{}; };
  const Layout unpaired_layout{100,0,0,0};
  Layout p=optional ? Layout{356,0,0,48} : unpaired_layout;
  struct { int major=12,minor=0; } properties;
  int device=0;
  try {
"""
        + preparation
        + """
    if(context.arena!=(optional && !injected ? 356 : 100) || live!=1)
      throw std::logic_error("retry retained wrong context");
  } catch(const std::runtime_error&) {
    if(live) throw std::logic_error("failed preparation leaked context");
    return false;
  }
  if(live) throw std::logic_error("successful context leaked");
  return true;
}
int main() {
  for(bool optional:{false,true})
    for(int injected:{0,1,2,3}) {
      const bool expected=injected==0 || (optional && injected==1);
      if(run(optional,injected)!=expected) return 1;
      const int expected_attempts=optional && (injected==1 || injected==2) ? 2 : 1;
      if(attempts!=expected_attempts) return 2;
    }
}
"""
    )
    obj, binary = tmp_path / "allocation.o", tmp_path / "allocation"
    subprocess.run(
        [cache, compiler, "-std=c++20", "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)], check=True, capture_output=True
    )
    subprocess.run([str(binary)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "change", ["alpha", "beta", "stride", "slot", "output", "untyped"]
)
def test_projection_rejects_unprepared_or_scientifically_different_pair(
    first_cut: Any, change: str
) -> None:
    node, adapter, views, descriptors = first_cut
    before = descriptors.copy()
    paired = {"panel": ("second_panel", "v"), "t2_kj": ("second_t2", "v")}
    if change == "stride":
        paired["panel"] = ("second_panel", "v*v")
    with pytest.raises(ValueError):
        _gemm(
            node,
            views,
            "2.0" if change == "alpha" else "(1.0/1.0)",
            "1.0" if change == "beta" else "0.0",
            adapter=None if change == "untyped" else adapter,
            descriptors=descriptors,
            paired_bindings=paired,
            paired_output_pointer=None if change == "output" else "second_output",
            paired_slot=1 if change == "slot" else 0,
        )
    assert descriptors == before
