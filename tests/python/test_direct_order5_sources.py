"""Execute weighted order-five roots and the native separate-source adapter.

Displaced-value Hermite ERIs provide an independent recurrence for the new
weighted Wick gradients. A second check uses the retained AO-quartet force
consumer to validate source indexing, symmetry, screening and repeated atoms.
Neither host control substitutes for real-device endpoint qualification.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)
from generativeqc_compiler.integral.direct_pair_gradient_cuda import (
    emit_direct_high_order_pair_gradient_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)
from generativeqc_compiler.integral.weighted_eri_cuda import emit_order5_weighted_header
from test_direct_low_order_sources import PROBE as LOW_ORDER_PROBE
from test_direct_low_order_sources import prepare_source_probe

ROOT = Path(__file__).resolve().parents[2]


def test_weighted_order5_independent_values_and_native_source_adapter(
    tmp_path: Path,
) -> None:
    """Cover complete component partitions, both spins, screening and zero sources."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("native host qualification requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = prepare_source_probe(tmp_path)
    headers = {
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        **emit_direct_cartesian_contraction_headers(),
        "generated_direct_high_order_pair_gradient.cuh": emit_direct_high_order_pair_gradient_header(),
        "generated_direct_source_contraction.cuh": emit_direct_source_contraction_header(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "order5_weighted_eri.cuh": emit_order5_weighted_header(),
    }
    for name, contents in headers.items():
        (tmp_path / name).write_text(contents)
    shim = tmp_path / "cuda_runtime.h"
    shim.write_text(
        shim.read_text()
        + "\ninline unsigned __popc(unsigned x) { return __builtin_popcount(x); }\n"
    )
    # Reuse the established density/primitive-pair fixture, not its low-order
    # oracle or comparison. Add the raw AO metadata required by the old warp
    # consumer so both receive exactly the same contracted basis.
    fixture = LOW_ORDER_PROBE[
        LOW_ORDER_PROBE.index("struct Fixture {") : LOW_ORDER_PROBE.index(
            "template<bool Unrestricted>"
        )
    ]
    source.write_text(PREFIX + fixture + PROBE)
    executable = tmp_path / "order5"
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(source),
            "-o",
            str(executable),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "order-five weighted gates PASS" in result.stdout
    assert "independent coordinates 54864 " in result.stdout
    assert "adapter comparisons 746496 " in result.stdout
    print(result.stdout)


PREFIX = r"""
#include "scf/cuda/direct_force_order5_sources.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "generated_direct_shell_class.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace generativeqc::scf::cuda_execution;
namespace weighted = generativeqc::scf::generated_weighted_eri;
unsigned long long independent_checks=0, adapter_checks=0;
double max_independent_error=0, max_adapter_error=0;

std::vector<Angular> components(unsigned order) {
  std::vector<Angular> result;
  for(int x=static_cast<int>(order);x>=0;--x)
    for(int y=static_cast<int>(order)-x;y>=0;--y)
      result.push_back({static_cast<unsigned>(x),static_cast<unsigned>(y),order-x-y});
  return result;
}

PrimitivePairData pair(double a,double b,Vec3<double> A,Vec3<double> B) {
  const double p=a+b,mu=a*b/p;
  const double r2=(A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y)+(A.z-B.z)*(A.z-B.z);
  return {p,mu,{(a*A.x+b*B.x)/p,(a*A.y+b*B.y)/p,(a*A.z+b*B.z)/p},
          std::exp(-mu*r2),a/p,b/p};
}

template<unsigned Class,unsigned A,unsigned B,unsigned C,unsigned D>
void independent() {
  const auto aa=components(A),bb=components(B),cc=components(C),dd=components(D);
  constexpr unsigned count=Order5SourceRoots<Class>::component_count;
  for(unsigned coincident=0;coincident<2;++coincident) {
    std::array<Vec3<double>,4> positions{{{.1,-.2,-.8},{.3,.1,.7},{-.5,.6,.2},{.8,-.4,.3}}};
    if(coincident)positions[1]=positions[0];
    for(double exponent_scale:{0.4,1.0,3.0}) {
      const double a=.8*exponent_scale,b=.6*exponent_scale,c=.7*exponent_scale,d=.9*exponent_scale;
      weighted::Geometry g;
      const auto first=pair(a,b,positions[0],positions[1]);
      const auto second=pair(c,d,positions[2],positions[3]);
      const double T=weighted::make_direct_cached_geometry(first,second,false,false,
                            positions[0],positions[1],positions[2],positions[3],g);
      boys_values<6>(T,g.boys);
      // One-hot weights cover every emitted component, across every 64-entry boundary.
      // The final signed dense vector additionally checks the additive split.
      for(unsigned target=0;target<=count;++target) {
        std::array<double,count> weights{};
        if(target<count)weights[target]=1;
        else for(unsigned i=0;i<count;++i)weights[i]=std::sin(.7*i+.2);
        const auto result=Order5SourceRoots<Class>::evaluate(g,weights.data());
        double gradient[4][3]{};
        for(unsigned i=0;i<3;++i)for(unsigned axis=0;axis<3;++axis) {
          gradient[i][axis]=result.center[i][axis];gradient[3][axis]-=result.center[i][axis];
        }
        for(unsigned center=0;center<4;++center)for(unsigned axis=0;axis<3;++axis) {
          auto value=[&](double displacement) {
            auto p=positions;
            double* moved=axis==0?&p[center].x:axis==1?&p[center].y:&p[center].z;
            *moved+=displacement;
            double total=0;unsigned component=0;
            for(auto ai:aa)for(auto bi:bb)for(auto ci:cc)for(auto di:dd) {
              const double w=weights[component++];if(w==0)continue;
              total+=w*primitive_eri_cartesian_shell_pairs<A,B,C,D,double>(
                  a,p[0],ai,b,p[1],bi,c,p[2],ci,d,p[3],di);
            }
            return total;
          };
          for(double h:{2e-4,1e-4}) {
            const double expected=(value(-2*h)-8*value(-h)+8*value(h)-value(2*h))/(12*h);
            const double error=std::abs(gradient[center][axis]-expected);
            if(!std::isfinite(error)||error>2e-8*(1+std::abs(expected))) {
              std::fprintf(stderr,"independent class=%u component=%u axis=%u/%u %.17g %.17g\n",
                           Class,target,center,axis,gradient[center][axis],expected);std::exit(1);
            }
            max_independent_error=std::max(max_independent_error,error);++independent_checks;
          }
        }
      }
    }
  }
}
"""


PROBE = r"""
template<bool Unrestricted>
void adapters() {
  for(const auto momenta:{std::array<std::uint8_t,4>{2,1,1,1},{2,1,2,0},{2,2,1,0},
                          {1,2,0,2},{0,2,1,2},{1,0,2,2},{1,1,1,2}})
    for(unsigned centers=0;centers<4;++centers)for(unsigned density_mode=0;density_mode<4;++density_mode) {
      Fixture fixture(momenta,centers,density_mode,Unrestricted);
      auto& b=fixture.batch;
      std::vector<std::int32_t> ao_shells;
      std::vector<std::uint8_t> ao_angular;
      std::array<std::int64_t,5> primitive_offsets{0,2,4,6,8};
      std::array<double,8> exponents,coefficients;
      for(unsigned shell=0;shell<4;++shell) {
        for(auto v:components(momenta[shell])) {
          ao_shells.push_back(shell);
          ao_angular.insert(ao_angular.end(),{static_cast<std::uint8_t>(v.x),static_cast<std::uint8_t>(v.y),static_cast<std::uint8_t>(v.z)});
        }
        for(unsigned primitive=0;primitive<2;++primitive) {
          exponents[2*shell+primitive]=.5+.13*shell+.2*primitive;
          coefficients[2*shell+primitive]=.7+.1*primitive;
        }
      }
      b.direct_ao_shells=ao_shells.data();b.direct_ao_angular=ao_angular.data();
      b.shell_primitive_offsets=primitive_offsets.data();
      b.primitive_exponents=exponents.data();b.primitive_coefficients=coefficients.data();
      for(std::uint32_t bra=0;bra<10;++bra)for(std::uint32_t ket=0;ket<=bra;++ket) {
        const unsigned cls=direct_quartet_shell_class_device(
            momenta[fixture.pair_first[bra]],momenta[fixture.pair_second[bra]],
            momenta[fixture.pair_first[ket]],momenta[fixture.pair_second[ket]]);
        if(!weighted_order5_source_class(cls))continue;
        for(double screening:{0.0,.6,2.0})for(const auto scales:{std::array<double,2>{1.7,-.23},{0,-.23},{1.7,0},{0,0}}) {
          std::array<double,24> expected{},actual{};std::uint8_t active=1;
          const std::uint32_t task_count=1;
          ActiveShellQuartetTile task{bra,ket,0};
          const auto layout=direct_shell_ao_quartet_layout(b,bra,ket);
          for(unsigned tile=0;tile*generativeqc::scf::detail::kDirectQuartetTileSize<layout.quartet_count;++tile) {
            task.tile=tile;
            for(unsigned subtile=0;subtile<generativeqc::scf::detail::direct_quartet_subtiles_per_tile(5);++subtile)
              for(unsigned lane=0;lane<32;++lane)
                contract_two_electron_force_quartet_subtile_scaled<Unrestricted,5,true>(
                    b,&task_count,&task,screening,fixture.schwarz.data(),fixture.density.data(),&active,
                    expected.data(),0,scales[0],scales[1],subtile,lane);
          }
          task.tile=0;
          contract_two_electron_force_order5_sources<Unrestricted>(cls,b,task,screening,
              fixture.schwarz.data(),fixture.density.data(),&active,actual.data(),scales[0],scales[1]);
          for(unsigned i=0;i<24;++i) {
            const double error=std::abs(expected[i]-actual[i]);
            if(!std::isfinite(error)||error>2e-10*(1+std::abs(expected[i]))) {
              std::fprintf(stderr,"adapter class=%u spin=%u pairs=%u,%u axis=%u %.17g %.17g\n",
                  cls,Unrestricted,bra,ket,i,actual[i],expected[i]);std::exit(2);
            }
            max_adapter_error=std::max(max_adapter_error,error);++adapter_checks;
          }
          actual.fill(0);active=0;
          contract_two_electron_force_order5_sources<Unrestricted>(cls,b,task,screening,
              fixture.schwarz.data(),fixture.density.data(),&active,actual.data(),scales[0],scales[1]);
          active=1;task.tile=1;
          contract_two_electron_force_order5_sources<Unrestricted>(cls,b,task,screening,
              fixture.schwarz.data(),fixture.density.data(),&active,actual.data(),scales[0],scales[1]);
          for(double v:actual)if(v!=0)std::exit(3);
        }
      }
    }
}
int main() {
  independent<kDpppShellClass,2,1,1,1>();independent<kDpdsShellClass,2,1,2,0>();
  independent<kDdpsShellClass,2,2,1,0>();
  adapters<false>();adapters<true>();
  std::printf("order-five weighted gates PASS; independent coordinates %llu error %.12g; "
              "adapter comparisons %llu error %.12g\n",
              independent_checks,max_independent_error,adapter_checks,max_adapter_error);
}
"""
