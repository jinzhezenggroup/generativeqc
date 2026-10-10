"""Compile the dddd consumer bridge and compare its scatter with dense J/K.

The emitted materialized helper and native stream body execute unchanged. Test
stubs provide a single-lane, one-component recurrence, not a CUDA simulator or
an ERI oracle. Independent dense contraction controls RHF/UHF factors, index
coincidences and offsets; the real-device fixture controls recurrence/lifetime.
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition
from generativeqc_compiler.integral.direct_pair_materialized_cuda import (
    emit_direct_pair_materialized_support,
)
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_materialized_exchange_bridge_matches_dense_contraction(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    native = (ROOT / "src/scf/cuda/direct_bounded_dddd.cu").read_text()
    # CUDA translation units need a host shim here; use named production
    # implementations instead of matching an entire template signature.
    helper = cpp_function_definition(
        emit_direct_pair_materialized_support(),
        "contract_materialized_direct_pair_fock",
        include_template=True,
    )
    kernel = cpp_function_definition(
        native, "bounded_direct_dddd_streaming_kernel", include_template=True
    )
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __host__\n"
        "#define __forceinline__ inline\n#define __global__\n#define __shared__\n"
        "#define __launch_bounds__(...)\n"
        "template<class T> T atomicAdd(T* p,T value) { T old=*p; *p+=value; return old; }\n"
        "inline void __syncthreads() {}\ninline void __syncwarp() {}\n"
        "inline bool __syncthreads_or(bool value) { return value; }\n"
    )
    (tmp_path / "scatter.hpp").write_text(emit_direct_fock_accumulation_header())
    source, binary = tmp_path / "bridge.cpp", tmp_path / "bridge"
    source.write_text(PREFIX + helper + STUBS + kernel + DRIVER)
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=["-std=c++20", "-O0", f"-I{tmp_path}", f"-I{ROOT / 'src'}"],
        compile_timeout=60,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=False, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2640 independent bridge comparisons passed" in result.stdout


PREFIX = r"""
#include "scatter.hpp"
#include "scf/generated_shell_task.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
namespace generativeqc::integrals { enum class CoulombRange { Full, Short, Long }; }
namespace generativeqc::scf::detail {
constexpr unsigned kDirectQuartetThreads = 32, kDirectQuartetTileSize = 256;
constexpr unsigned direct_quartet_subtiles_per_tile(unsigned) { return 8; }
}
namespace generativeqc::scf::cuda_execution {
struct { unsigned x{}; } threadIdx;
enum class DirectScreeningPurpose { Fock, Force };
constexpr unsigned kDdddAngularOrder = 8;
template<bool Materialized> constexpr unsigned kDdddStreamThreads = Materialized ? 256 : 32;
template<bool Materialized> void synchronize_dddd_stream() {}
struct ActiveShellQuartetTile { unsigned first_pair, second_pair, tile; };
struct ShellPairDensityBounds {};
struct DeviceShellClassProfileEntry {};
struct PrimitivePairData { double exponent_sum = 1.0, product_center = 0.0; };
struct DeviceBatch {
  int total_shell_pairs = 1, batch_size = 2, direct_nbf = 4;
  const int *shell_pair_systems, *shell_pair_first, *shell_pair_second, *shell_angular;
  const int *shell_atoms, *shell_pair_primitive_offsets, *shell_primitive_offsets;
  const PrimitivePairData* shell_primitive_pairs;
  const double *primitive_coefficients, *direct_ao_coefficients;
};
struct GeneratedShellPairStream {
  int batch_size = 2;
  const unsigned *pair_class_offsets, *pair_order;
  const double* shell_pair_bounds;
  const void* shell_pair_density_bounds{};
  detail::GeneratedFockConsumer fock_consumer;
};
struct MaterializedDirectPairWork {
  unsigned long long bra_preparations{}, ket_preparations{}, coulomb_preparations{};
  unsigned long long component_contractions{}, published_components{};
};
struct CoulombState {
  double value{};
  double& at(unsigned, unsigned, unsigned, unsigned) { return value; }
};
template<unsigned Order> struct MaterializedDirectPairRecurrence {
  PrimitivePairData first, second;
  double coefficients[4];
  int bra[3]{}, ket[3]{};
  CoulombState coulomb{};
};
std::array<std::size_t, 4> quartet;
unsigned expected_mode, screening_calls, fallback_calls;
std::size_t shell_ao_pair_count(DeviceBatch, unsigned) { return 1; }
unsigned direct_quartet_shell_class_device(unsigned, unsigned, unsigned, unsigned) { return 0; }
bool decode_direct_tile_ao_ordinal(DeviceBatch, ActiveShellQuartetTile, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t,
    std::size_t& i, std::size_t& j, std::size_t& k, std::size_t& l) {
  i=quartet[0]; j=quartet[1]; k=quartet[2]; l=quartet[3]; return true;
}
bool direct_ao_quartet_survives_schwarz(const double*, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t, double) { return true; }
template<class T> T atom_position(DeviceBatch, int, int) { return 0; }
int direct_ao_angular(DeviceBatch, std::size_t) { return 0; }
void prepare_materialized_direct_pair(const PrimitivePairData&, unsigned, unsigned,
    double, double, int (&)[3]) {}
template<unsigned Order> void fill_coulomb(double, double, double, CoulombState&) {}
template<unsigned Order> bool fill_range_coulomb(double, double, double,
    generativeqc::integrals::CoulombRange, double, CoulombState&) {
  assert(false && "full-range bridge must not request a range recurrence");
  return false;
}
template<unsigned Order> double consume_cartesian_coulomb(
    double, double, int, int, int, int, const int (&)[3], const int (&)[3],
    const CoulombState&) {
  return 0.75;
}
"""

STUBS = r"""
void decode_lower_triangle(std::size_t ordinal, std::size_t& row, std::size_t& column) {
  assert(ordinal == 0); row=column=0;
}
template<bool Unrestricted, DirectScreeningPurpose Purpose>
bool direct_shell_quartet_survives_screening(DeviceBatch, unsigned, unsigned, double,
    const double*, const ShellPairDensityBounds* density_bounds, double*,
    bool exchange_only, bool coulomb_only) {
  assert(density_bounds != nullptr);
  assert(exchange_only == (expected_mode == 2 || expected_mode == 3));
  assert(coulomb_only == (expected_mode == 1));
  ++screening_calls;
  return true;
}
void profile_bounded_direct_shell_quartet(
    DeviceBatch, ActiveShellQuartetTile, DeviceShellClassProfileEntry*) { assert(false); }
template<bool Unrestricted, unsigned Order, class... Args>
void contract_two_electron_force_quartet_subtile_scaled(Args...) { assert(false); }
template<bool Unrestricted, unsigned Order>
void contract_fock_direct_quartet_subtile(
    DeviceBatch batch, const unsigned*, const ActiveShellQuartetTile*, double,
    const double*, const double* density, const std::uint8_t*, double* output,
    const std::uint64_t*, std::size_t subtile, unsigned, bool coulomb_only,
    bool exchange_only, generativeqc::integrals::CoulombRange range, double omega,
    bool hf_exchange) {
  assert(coulomb_only == (expected_mode == 1));
  assert(exchange_only == (expected_mode == 2 || expected_mode == 3));
  assert(hf_exchange == (expected_mode == 3));
  assert(range == generativeqc::integrals::CoulombRange::Full && omega == 0.0);
  ++fallback_calls;
  if (subtile != 0) return;
  const std::size_t n=batch.direct_nbf, physical=n*n, spin=2*physical;
  accumulate_direct_fock_integral<Unrestricted>(n,physical,spin,density,output,
      quartet[0],quartet[1],quartet[2],quartet[3],0.75,coulomb_only,exchange_only,hf_exchange);
}
"""

DRIVER = r"""
template<bool Unrestricted> unsigned qualify() {
  constexpr std::size_t n=4, matrix=n*n, physical=matrix, spin_offset=2*matrix;
  const int system[]{1}, shell[]{0}, angular[]{2}, offsets[]{0,1};
  const double coefficients[]{1.0}, ao_coefficients[]{1,1,1,1,1,1,1,1};
  const PrimitivePairData primitive[]{{}};
  DeviceBatch batch{1,2,4,system,shell,shell,angular,shell,offsets,offsets,
                    primitive,coefficients,ao_coefficients};
  const std::uint8_t active[]{1,1};
  const double bound[]{1.0};
  const unsigned order[]{0};
  unsigned class_offsets[18]{};
  class_offsets[17]=1;
  GeneratedShellPairStream topology{2,class_offsets,order,bound,nullptr,{}};
  const ShellPairDensityBounds density_bounds{};
  std::vector<double> density(80);
  for(std::size_t i=0;i<density.size();++i)
    density[i]=0.013*(int(i%11)-4)+0.002*double(i);
  auto index=[](std::size_t a,std::size_t b,std::size_t c,std::size_t d) {
    return ((a*n+b)*n+c)*n+d;
  };
  unsigned cases=0;
  // Cover density-aware admission and the metadata-free Schwarz fallback.
  for(bool with_density_bounds:{false,true})
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<=i;++j)
  for(std::size_t k=0;k<n;++k) for(std::size_t l=0;l<=k;++l) {
    if(i*(i+1)/2+j < k*(k+1)/2+l) continue;
    quartet={i,j,k,l};
    topology.shell_pair_density_bounds=with_density_bounds ? &density_bounds : nullptr;
    const std::set<std::array<std::size_t,4>> orbit{
      {i,j,k,l},{j,i,k,l},{i,j,l,k},{j,i,l,k},
      {k,l,i,j},{l,k,i,j},{k,l,j,i},{l,k,j,i}};
    std::vector<double> eri(n*n*n*n);
    for(const auto& abcd:orbit) eri[index(abcd[0],abcd[1],abcd[2],abcd[3])]=0.75;
    for(unsigned mode=0;mode<4;++mode) {
      expected_mode=mode;
      topology.fock_consumer=static_cast<detail::GeneratedFockConsumer>(mode);
      std::vector<double> expected(80);
      for(std::size_t a=0;a<n;++a) for(std::size_t b=0;b<n;++b)
      for(std::size_t c=0;c<n;++c) for(std::size_t d=0;d<n;++d) {
        const double total = Unrestricted
          ? density[spin_offset+c+n*d]+density[spin_offset+matrix+c+n*d]
          : density[physical+c+n*d];
        for(unsigned spin=0;spin<(Unrestricted ? 2U : 1U);++spin) {
          const auto offset=Unrestricted ? spin_offset+spin*matrix : physical;
          if(mode==0 || mode==1)
            expected[offset+a+n*b]+=total*eri[index(a,b,c,d)];
          if(mode!=1)
            expected[offset+a+n*b]+=(mode==2 ? 1.0 : (Unrestricted ? -1.0 : -0.5))*
                density[offset+c+n*d]*eri[index(a,c,b,d)];
        }
      }
      // Direct helper, production materialized call site, retained call site.
      for(unsigned route=0;route<3;++route) {
        std::vector<double> actual(80);
        MaterializedDirectPairWork work{};
        screening_calls=fallback_calls=0;
        if(route==0) {
          MaterializedDirectPairRecurrence<8> shared{};
          double checked=-99.0;
          if(mode==3)
            contract_materialized_direct_pair_fock<Unrestricted,8>(batch,{0,0,0},0.0,
                bound,density.data(),active,actual.data(),nullptr,shared,&work,
                false,true,&checked,true);
          else
            // The existing positional checked_components call remains valid.
            contract_materialized_direct_pair_fock<Unrestricted,8>(batch,{0,0,0},0.0,
                bound,density.data(),active,actual.data(),nullptr,shared,&work,
                mode==1,mode==2,&checked);
          assert(checked==0.75);
        } else {
          unsigned cursor=0;
          unsigned long long census=0;
          if(route==1)
            bounded_direct_dddd_streaming_kernel<Unrestricted,DirectScreeningPurpose::Fock,
                false,true>(batch,&topology,0.0,bound,density.data(),active,actual.data(),
                            &cursor,nullptr,&census,1.0,-0.5,&work);
          else
            bounded_direct_dddd_streaming_kernel<Unrestricted,DirectScreeningPurpose::Fock,
                false,false>(batch,&topology,0.0,bound,density.data(),active,actual.data(),
                             &cursor,nullptr,&census,1.0,-0.5,&work);
          assert(cursor==2 && census==1);
          assert(screening_calls==(with_density_bounds ? 1U : 0U));
          assert(fallback_calls==(route==2 ? 8U : 0U));
        }
        assert(work.coulomb_preparations==(route==2 ? 0U : 1U));
        assert(work.published_components==(route==2 ? 0U : 1U));
        for(std::size_t item=0;item<actual.size();++item) {
          if(!std::isfinite(actual[item]) || std::abs(actual[item]-expected[item])>2e-13) {
            std::cerr << "bridge mismatch spin=" << Unrestricted << " mode=" << mode
                      << " route=" << route << " item=" << item << '\n';
            std::abort();
          }
        }
        ++cases;
      }
    }
  }
  return cases;
}
} // namespace generativeqc::scf::cuda_execution
int main() {
  using namespace generativeqc::scf::cuda_execution;
  const unsigned cases=qualify<false>()+qualify<true>();
  assert(cases==2640);
  std::cout << cases << " independent bridge comparisons passed\n";
}
"""
