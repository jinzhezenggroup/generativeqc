"""Validate actual strided triples recipes against their semantic mode axes."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from _cc_owner_test_support import compile_owner

from tools.generate_df_occupied_triples import native_execution_header

if TYPE_CHECKING:
    from pathlib import Path


def test_all_triples_views_validate_at_unit_and_nonunit_extents(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    source = native_execution_header()
    groups = re.findall(r"\.add\(o,v,q,\{(.*?)\},provider_", source, re.DOTALL)
    assert len(groups) == 3
    unit = tmp_path / "affine.cpp"
    unit.write_text(
        """
#include "tensor/native_contraction.hpp"
#include <vector>
using namespace generativeqc::tensor;
std::size_t checked_product(std::initializer_list<std::size_t> sizes) {
  std::size_t result=1;for(auto n:sizes) result=contraction_product(result,n);return result;
}
int main() {
 for(std::size_t o:{1,2,4}) for(std::size_t v:{1,3,7}) for(std::size_t q:{1,2,5}) {
  const std::vector<ContractionRequest> requests{
"""
        + ",".join(groups)
        + """
  };
  for(const auto& request:requests) request.validate();
  if(requests[2].beta!=1 || requests[4].beta!=0) return 1;
  if(requests[2].leading_dimension(0)!=o*o || requests[2].leading_dimension(1)!=o*v*v) return 2;
  auto reject=[](const auto& r){try{r.validate();}catch(const std::exception&){return true;}return false;};
  auto bad=requests[2];bad.leading_dimensions[0]=1;
  if(o>1 && !reject(bad)) return 3;
  bad=requests[2];bad.operands[0].strides[0]+=1;if(!reject(bad)) return 4;
  bad=requests[2];bad.beta=std::numeric_limits<double>::infinity();if(!reject(bad)) return 5;
 }
}
"""
    )
    binary = tmp_path / "affine"
    compile_owner(compiler, tmp_path, [unit], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
