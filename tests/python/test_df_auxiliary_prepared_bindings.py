"""Full/tail Q bindings prepare once and charge all retained descriptors."""

from __future__ import annotations

import re
import shutil
import subprocess
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

from tools.generate_df_ccsd_hoisted import cuda_header, cuda_source


def test_prepared_auxiliary_batch_variants_and_host_capacity(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = cuda_source()
    binders = re.findall(r"static void bind_.*?\n}\n", source, flags=re.DOTALL)
    assert len(binders) == 4
    prepare = re.search(r"void prepare_contractions\(.*?\n}\n", source, flags=re.DOTALL)
    assert prepare is not None
    capacity = next(
        line
        for line in cuda_header().splitlines()
        if line.startswith("inline constexpr std::size_t contraction_host_bytes(")
    )
    unit = tmp_path / "bindings.cpp"
    unit.write_text(PREFIX + capacity + "\n" + "\n".join(binders) + prepare[0] + MAIN)
    binary = tmp_path / "bindings"
    compile_owner(compiler, tmp_path, [unit], binary)
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "tensor/native_contraction.hpp"
#include <vector>
#include <iostream>
namespace generativeqc::tensor {
struct CudaContractionContext {};
struct PreparedContractions {
  std::vector<std::size_t> batches;
  std::size_t requests{};
  static constexpr std::size_t storage_bytes(std::size_t n, std::size_t variants=1) {
    return 128+2*variants*n*sizeof(ContractionRequest);
  }
  void add(std::size_t, std::size_t, std::size_t q, std::vector<ContractionRequest> rows,
           CudaContractionContext&, std::size_t&, std::size_t&) {
    for(const auto& row:rows) row.validate();
    if(batches.size()==2) throw std::runtime_error("unbounded Q variants");
    for(const auto batch:batches)
      if(batch==q) throw std::runtime_error("duplicate Q variant");
    batches.push_back(q);
    if(requests && rows.size()!=requests) throw std::runtime_error("changed table width");
    requests=rows.size();
  }
};
}
std::size_t checked_product(std::initializer_list<std::size_t> factors) {
  std::size_t n=1;
  for(auto factor:factors) n=generativeqc::tensor::contraction_product(n,factor);
  return n;
}
struct CudaState {
  std::size_t o{},v{};
  generativeqc::tensor::PreparedContractions prepare_contractions,auxiliary_contractions,
      iteration_contractions,auxiliary_batched_contractions;
};
"""

MAIN = r"""
int main() {
  using generativeqc::tensor::PreparedContractions;
  for(std::size_t o:{1,2,4}) for(std::size_t v:{1,3,7})
  for(std::size_t batch:{1,2,3,8}) for(std::size_t tail=0;tail<batch;++tail) {
    CudaState state; state.o=o; state.v=v;
    generativeqc::tensor::CudaContractionContext context;
    std::size_t calls=0,summands=0;
    prepare_contractions(state,context,batch,tail,calls,summands);
    if(calls || summands) return 1;
    std::size_t expected=0;
    for(auto* table:{&state.prepare_contractions,&state.auxiliary_contractions,
                    &state.iteration_contractions}) {
      if(table->batches!=std::vector<std::size_t>{1}) return 2;
      expected+=PreparedContractions::storage_bytes(table->requests);
    }
    std::vector<std::size_t> variants;
    if(batch>1) variants.push_back(batch);
    if(tail>1) variants.push_back(tail);
    const auto& batched=state.auxiliary_batched_contractions;
    if(batched.batches!=variants || (!variants.empty() && !batched.requests)) return 3;
    expected+=PreparedContractions::storage_bytes(batched.requests,variants.size());
    if(contraction_host_bytes(variants.size())!=expected) return 4;
  }
  std::cout << "126 full/tail shape combinations retain one-Q bindings and exact descriptor charges\n";
}
"""
