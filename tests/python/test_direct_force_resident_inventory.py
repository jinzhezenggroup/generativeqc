"""Check complete resident ownership independently of HF/DFT method policy."""

import subprocess
from pathlib import Path

from generativeqc_compiler.integral.direct_resident_schedule import (
    emit_direct_resident_psss_schedule_header,
)
from test_coulomb_optional_allocation import compile_cached_probe
from test_direct_jk_optional_allocation import _definition

ROOT = Path(__file__).resolve().parents[2]


def test_shared_inventory_has_exact_batched_ownership_and_budget(
    tmp_path: Path,
) -> None:
    """Exercise multi-chunk bras, mixed classes, empty systems and exact budgets."""
    source = (ROOT / "src/scf/cuda/topology.cpp").read_text()
    definition = _definition(source, "bool make_direct_force_resident_bra_schedule(")
    cpp, binary = tmp_path / "inventory.cpp", tmp_path / "inventory"
    cpp.write_text(
        "#include <algorithm>\n#include <cassert>\n#include <cstdint>\n"
        "#include <map>\n#include <set>\n#include <vector>\n"
        + (ROOT / "src/runtime/bounded_workspace.hpp").read_text()
        + emit_direct_resident_psss_schedule_header()
        + "namespace runtime = generativeqc::runtime;\n"
        + "using namespace generativeqc::scf::cuda_execution;\n"
        + r"""
struct PsssResidentTask { std::uint32_t bra_pair, ket_begin, ket_count; };
struct HostBatch {
  std::vector<std::uint8_t> shell_angular;
  std::vector<std::int32_t> shell_pair_first, shell_pair_second;
  std::vector<std::int64_t> shell_pair_primitive_offsets{0}, system_shell_pair_offsets{0};
};
"""
        + definition
        + r"""
int main() {
  HostBatch host;
  std::set<std::pair<unsigned,unsigned>> expected;
  for (unsigned shells : {24,0,18}) {
    const auto shell_begin = host.shell_angular.size();
    const auto pair_begin = host.shell_pair_first.size();
    for (unsigned shell = 0; shell < shells; ++shell)
      host.shell_angular.push_back(shell % 8 < 6 ? 0 : shell % 8 == 6 ? 1 : 2);
    for (unsigned first = shell_begin; first < host.shell_angular.size(); ++first)
      for (unsigned second = shell_begin; second <= first; ++second) {
        host.shell_pair_first.push_back(first);
        host.shell_pair_second.push_back(second);
        const unsigned order = host.shell_angular[first]+host.shell_angular[second];
        host.shell_pair_primitive_offsets.push_back(
            host.shell_pair_primitive_offsets.back()+(order==1 ? 7 : 1));
      }
    const auto pair_end = host.shell_pair_first.size();
    host.system_shell_pair_offsets.push_back(pair_end);
    for (unsigned bra = pair_begin; bra < pair_end; ++bra)
      for (unsigned ket = pair_begin; ket < pair_end; ++ket)
        if (host.shell_angular[host.shell_pair_first[bra]]+
              host.shell_angular[host.shell_pair_second[bra]] == 1 &&
            host.shell_angular[host.shell_pair_first[ket]]+
              host.shell_angular[host.shell_pair_second[ket]] == 0)
          expected.emplace(bra,ket);
  }
  std::vector<PsssResidentTask> tasks;
  std::vector<std::uint32_t> pairs;
  std::size_t capacity = 0;
  const auto unlimited = std::numeric_limits<std::size_t>::max();
  assert(make_direct_force_resident_bra_schedule(host,tasks,pairs,capacity,unlimited,true));
  assert(capacity == 7 && !tasks.empty());
  std::set<std::pair<unsigned,unsigned>> visited;
  for (const auto task : tasks) {
    assert(task.ket_count && task.ket_count <= kResidentPsssThreads);
    for (unsigned ket = 0; ket < task.ket_count; ++ket)
      assert(visited.emplace(task.bra_pair,pairs.at(task.ket_begin+ket)).second);
  }
  assert(visited == expected);
  const auto task_count = tasks.size(), pair_count = pairs.size();
  const auto bytes = task_count*sizeof(PsssResidentTask)+pair_count*sizeof(std::uint32_t);
  assert(!make_direct_force_resident_bra_schedule(host,tasks,pairs,capacity,bytes-1,true));
  assert(tasks.empty() && pairs.empty() && capacity == 0);
  assert(make_direct_force_resident_bra_schedule(host,tasks,pairs,capacity,bytes,true));
  assert(tasks.size()==task_count && pairs.size()==pair_count && capacity==7);
  const auto expected_pairs = pairs;
  assert(make_direct_force_resident_bra_schedule(host,tasks,pairs,capacity,unlimited,false));
  assert(tasks.empty() && pairs==expected_pairs);
  auto malformed = host;
  malformed.system_shell_pair_offsets[1] = host.shell_pair_first.size()+1;
  assert(!make_direct_force_resident_bra_schedule(malformed,tasks,pairs,capacity,unlimited,true));
  assert(tasks.empty() && pairs.empty());
}
"""
    )
    compile_cached_probe(cpp, binary)
    subprocess.run([str(binary)], check=True, timeout=10)
