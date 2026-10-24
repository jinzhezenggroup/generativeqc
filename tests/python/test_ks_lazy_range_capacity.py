"""Execute the native KS capacity policy with an observable dry owner query."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_lazy_range_quote_excludes_optional_coulomb_storage(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Keep primary SCF storage while quoting only required lazy force owners."""
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    begin = source.index("enum class KsProviderPurpose")
    end = source.index("/** An explicit DF envelope", begin)
    unit = tmp_path / "capacity.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include <cstddef>
#include <vector>
#define GENERATIVEQC_HAS_CUDA 1
using generativeqc_backend=int;
constexpr int GENERATIVEQC_BACKEND_CUDA=1;
namespace core {
struct Shell { std::vector<double> primitives; };
struct System { std::vector<int> atoms; std::vector<Shell> shells; };
}
namespace runtime { std::size_t add_capacity(std::size_t a,std::size_t b) {return a+b;} }
namespace molecule {
std::size_t ao_count(const core::System&) {return 864;}
std::size_t cartesian_ao_count(const core::System&) {return 900;}
}
namespace scf {
std::size_t cuda_direct_coulomb_device_bytes(std::size_t batch,std::size_t nao,
    std::size_t atoms,std::size_t shells,std::size_t primitives,unsigned order,
    bool optional_md,std::size_t cart) {
  assert(batch==1 && nao==864 && atoms==108 && shells==2 && primitives==5);
  assert(order<=1);
  if(cart==0) assert(optional_md);
  else assert(cart==900 && !optional_md);
  // Independent capacity sentinel makes the reservation observable without
  // pretending that host emulation validates CUDA allocation or numerics.
  return 1000+order+(optional_md ? (128U<<20) : 0);
}
}
"""
        + source[begin:end]
        + r"""
int main() {
  const core::System system{std::vector<int>(108),{{{1,2}},{{3,4,5}}}};
  for(unsigned order : {0U,1U}) {
    assert(ks_provider_bytes(system,1,order)==1000+order+(128U<<20));
    assert(ks_provider_bytes(system,1,order,KsProviderPurpose::derivative_only)==1000+order);
    assert(ks_provider_bytes(system,0,order)==0);
    assert(ks_provider_bytes(system,0,order,KsProviderPurpose::derivative_only)==0);
  }
}
"""
    )
    binary = native_cxx.build_executable(
        [unit], tmp_path / "capacity", compile_args=("-std=c++20",)
    )
    subprocess.run([str(binary)], check=True, timeout=10)
