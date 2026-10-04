"""Execute the production page classifier/prefix/scatter in a serial host capsule.

This proves indexing/coverage, not GPU concurrency or integral numerics. The
screen predicate is deliberately labelled synthetic; native ERI oracle gates
remain necessary for the actual derivative consumer.
"""

import subprocess
from pathlib import Path

from test_direct_force_pages import ROOT, compile_probe


def test_actual_page_compaction_matches_exhaustive_pairs(tmp_path: Path) -> None:
    source = (ROOT / "src/scf/cuda/direct_bounded_tasks.cu").read_text()
    begin = source.index(
        "template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Materialize,"
    )
    end = source.index("\nvoid launch_classify_bounded_force_page(", begin)
    classify = source[begin:end]
    source = (ROOT / "src/scf/cuda/direct_generated_tasks.cu").read_text()
    begin = source.index("__global__ void prefix_generated_shell_task_counts_kernel(")
    end = source.index("/** Prefix selected scalar signature", begin)
    prefix = source[begin:end]
    begin = source.index("template <typename Task>")
    end = source.index("\nvoid launch_materialize_compact_force_tiles(", begin)
    scatter = source[begin:end]
    binary = compile_probe(tmp_path, PREFIX + classify + prefix + scatter + DRIVER)
    subprocess.run([str(binary)], check=True, timeout=30)


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>
#include "scf/direct_block_domain.hpp"
#include "scf/direct_block_schedule.hpp"
using namespace generativeqc::scf;
using std::min; using std::max;
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
struct { unsigned x=0; } threadIdx,blockIdx;
struct { unsigned x=1; } blockDim;
template<class Value> Value atomicAdd(Value* pointer,Value increment) {
  auto old=*pointer; *pointer+=increment; return old;
}
template<class Value> void atomicExch(Value* pointer,Value value) {*pointer=value;}
struct ActiveShellQuartetTile { std::uint32_t first_pair,second_pair,tile; };
struct GeneratedShellTask {};
struct ShellPairDensityBounds {};
struct DeviceBatch {
  std::size_t batch_size,total_shell_pair_block_quartets;
  const std::int64_t *system_shell_pair_offsets,*system_shell_pair_block_offsets,
                     *system_shell_pair_block_quartet_offsets;
  const std::int32_t *shell_pair_first,*shell_pair_second;
  const unsigned* shell_angular;
};
struct BoundedForcePage {
  ActiveShellQuartetTile* input; unsigned char* classes; unsigned* counts; void* profile;
};
enum class DirectScreeningPurpose { Fock,Force };
template<class Offset> std::size_t bounded_direct_block_row(
    const Offset* prefix,std::size_t rows,std::size_t ordinal) {
  return std::upper_bound(prefix,prefix+rows+1,ordinal)-prefix-1;
}
std::int32_t shell_pair_block_quartet_system(DeviceBatch batch,std::size_t ordinal) {
  return bounded_direct_block_row(batch.system_shell_pair_block_quartet_offsets,
                                   batch.batch_size,ordinal);
}
void decode_lower_triangle(std::size_t ordinal,std::size_t& first,std::size_t& second) {
  first=0; while ((first+1)*(first+2)/2<=ordinal) ++first;
  second=ordinal-first*(first+1)/2;
}
unsigned direct_quartet_shell_class_device(unsigned first,unsigned second,
                                           unsigned third,unsigned fourth) {
  const auto left=max(first,second)*(max(first,second)+1)/2+min(first,second);
  const auto right=max(third,fourth)*(max(third,fourth)+1)/2+min(third,fourth);
  return max(left,right)*(max(left,right)+1)/2+min(left,right);
}
template<DirectScreeningPurpose> bool bounded_direct_block_pair_survives_screening(
    std::size_t first,std::size_t second,std::int32_t,double threshold,
    const double* bounds,const double*) {return !(bounds[first]*bounds[second]<threshold);}
unsigned screens=0;
std::set<std::pair<std::size_t,std::size_t>> visited_shells;
template<bool Unrestricted,DirectScreeningPurpose> bool direct_shell_quartet_survives_screening(
    DeviceBatch,std::size_t first,std::size_t second,double threshold,const double* bounds,
    const ShellPairDensityBounds*) {
  ++screens;
  assert(visited_shells.emplace(first,second).second);
  return !(bounds[first]*bounds[second]<threshold) &&
      (first*17+second*3+(Unrestricted?1:0))%5!=0;
}
bool bounded_generated_class_enabled(unsigned shell,const std::uint64_t*,std::uint64_t mask) {
  return shell<55 && (mask&(1ULL<<shell));
}
void profile_bounded_direct_shell_quartet(DeviceBatch,ActiveShellQuartetTile,void*) {}
template<class Task> void populate_generated_shell_task(DeviceBatch,ActiveShellQuartetTile,Task&) {}
constexpr unsigned kLowOrderSignatureClassCount=0;
unsigned generated_low_order_signature_bucket(DeviceBatch,ActiveShellQuartetTile) {return 0;}
unsigned generated_low_order_signature_index(unsigned,unsigned) {return 0;}
constexpr unsigned char kNoGeneratedShellClass=255;
"""

DRIVER = r"""
template<bool Unrestricted> void run(bool indexed,std::size_t page_blocks,bool masked,bool empty) {
  visited_shells.clear();
  std::vector<std::int64_t> pairs{0,35,39,39,72},blocks{0,2,3,3,5},products{0,3,4,4,7};
  std::uint8_t active[]{1,0,1,1};
  std::vector<double> bounds(72);
  std::vector<std::int32_t> first(72),second(72);
  std::vector<unsigned> angular(144);
  for(unsigned pair=0;pair<72;++pair) {
    first[pair]=2*pair+1; second[pair]=2*pair;
    angular[first[pair]]=pair%4; angular[second[pair]]=(pair/4)%4;
    bounds[pair]=pair%9==0 ? 0.0 : 1.0/(1+pair%7);
  }
  if(empty) {
    std::fill(pairs.begin(),pairs.end(),0);
    std::fill(blocks.begin(),blocks.end(),0);
    std::fill(products.begin(),products.end(),0);
    bounds.clear();
  }
  const double threshold=0.12;
  const auto schedule=detail::make_bounded_direct_schedule(pairs,bounds,threshold);
  auto order=schedule.pair_order;
  if(!indexed) for(unsigned pair=0;pair<bounds.size();++pair) order[pair]=pair;
  std::vector<double> block_bounds(5,0.0);
  for(unsigned system=0;system<4;++system)
    for(auto pair=pairs[system];pair<pairs[system+1];++pair) {
      const auto block=blocks[system]+(pair-pairs[system])/32;
      block_bounds[block]=max(block_bounds[block],bounds[order[pair]]);
    }
  DeviceBatch batch{4,static_cast<std::size_t>(products.back()),pairs.data(),blocks.data(),products.data(),
                    first.data(),second.data(),angular.data()};
  detail::BoundedDirectBlockDomain domain{};
  if(indexed) domain={schedule.block_prefix.data(),static_cast<std::uint64_t>(blocks.back()),
                      schedule.block_prefix.back()};
  const auto total=indexed ? domain.quartet_count : batch.total_shell_pair_block_quartets;
  std::set<std::pair<unsigned,unsigned>> expected,actual;
  for(unsigned system=0;system<4;++system)
    for(auto bra=pairs[system];bra<pairs[system+1];++bra)
      for(auto ket=pairs[system];ket<=bra;++ket)
        if ((!masked || active[system]) && !(bounds[bra]*bounds[ket]<threshold) &&
            (bra*17+ket*3+(Unrestricted?1:0))%5!=0) expected.emplace(bra,ket);
  for(std::size_t begin=0;begin<total;) {
    const auto end=min(begin+page_blocks,total),capacity=(end-begin)*1024;
    std::vector<ActiveShellQuartetTile> input(capacity+1,{9999,9999,9999}),tasks(capacity+1,{9999,9999,9999});
    std::vector<unsigned char> classes(capacity+1,255);
    std::vector<unsigned> counts(55,0),offsets(56),writes(55),heads(55);
    BoundedForcePage page{input.data(),classes.data(),counts.data(),nullptr};
    unsigned long long cursor=0;
    compact_bounded_generated_tasks_kernel<Unrestricted,DirectScreeningPurpose::Force,true,true>(
        batch,threshold,bounds.data(),nullptr,order.data(),block_bounds.data(),nullptr,
        masked ? active : nullptr,nullptr,~std::uint64_t{0},0,nullptr,nullptr,&cursor,nullptr,nullptr,nullptr,nullptr,
        domain,begin,end,page);
    const auto screened=screens;
    blockIdx.x=0;
    prefix_generated_shell_task_counts_kernel(counts.data(),offsets.data(),writes.data(),heads.data());
    for(blockIdx.x=0;blockIdx.x<capacity;++blockIdx.x)
      materialize_generated_shell_tasks_kernel(batch,capacity,input.data(),classes.data(),offsets.data(),
          writes.data(),tasks.data(),0,nullptr,nullptr);
    assert(screens==screened && offsets.back()<=capacity);
    for(unsigned shell=0;shell<55;++shell) {
      assert(writes[shell]==counts[shell] && heads[shell]==0);
      for(auto index=offsets[shell];index<offsets[shell+1];++index) {
        const auto task=tasks[index];
        assert(task.tile==0 && task.first_pair>=task.second_pair);
        assert(shell==direct_quartet_shell_class_device(
            angular[first[task.first_pair]],angular[second[task.first_pair]],
            angular[first[task.second_pair]],angular[second[task.second_pair]]));
        assert(actual.emplace(task.first_pair,task.second_pair).second);
      }
    }
    assert(input.back().first_pair==9999 && tasks.back().first_pair==9999 && classes.back()==255);
    begin=end;
  }
  assert(actual==expected);
}
int main() {
  for(bool indexed:{false,true}) for(std::size_t page:{1,2,3,16})
    for(bool masked:{false,true}) for(bool empty:{false,true}) {
      run<false>(indexed,page,masked,empty); run<true>(indexed,page,masked,empty);
    }
}
"""
