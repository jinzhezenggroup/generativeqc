"""Execute the native two-source scheduler against retained scalar workers.

This host-only control checks indexing, zero-channel handling, reduction order
and executed geometry/force call counts. It is not an independent scientific
oracle or a GPU performance measurement; native CPU ERIs qualify those separately.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)
from generativeqc_compiler.integral.weighted_eri_cuda import (
    emit_low_order_weighted_header,
)

ROOT = Path(__file__).resolve().parents[2]


def prepare_source_probe(directory: Path) -> Path:
    """Emit real algebra with test-only counters, retaining all native helpers."""
    header = emit_low_order_weighted_header()
    namespace = "namespace generativeqc::scf::generated_weighted_eri {"
    header = header.replace(
        namespace,
        namespace + "\ninline unsigned long long geometry_calls = 0, force_calls = 0;",
        1,
    )
    for symbol, counter in (
        ("make_direct_cached_geometry", "geometry_calls"),
        *(
            (name + "_force", "force_calls")
            for name in (
                "ssss",
                "psss",
                "psps",
                "ppss",
                "dsss",
                "ppps",
                "dsps",
                "dpss",
                "fsss",
            )
        ),
    ):
        opening = header.index("{", header.index(symbol + "(")) + 1
        header = header[:opening] + f"\n++{counter};" + header[opening:]
    (directory / "weighted_eri.cuh").write_text(header)
    (directory / "generated_direct_fock_accumulation.cuh").write_text(
        emit_direct_fock_accumulation_header()
    )
    (directory / "cuda_runtime.h").write_text(
        "#pragma once\n#include <algorithm>\n#include <cmath>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "#define __noinline__\nusing std::min; using std::max;\n"
        "inline double atomicAdd(double* target, double value) {\n"
        "  const double previous = *target; *target += value; return previous;\n}\n"
    )
    source = directory / "probe.cpp"
    source.write_text(PROBE)
    return source


def test_two_sources_preserve_scalar_results_and_share_geometry(tmp_path: Path) -> None:
    """Cover all low-order angular orientations, repeated centers and AO masks."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("native host control requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = prepare_source_probe(tmp_path)
    executable = tmp_path / "probe"
    result = subprocess.run(
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
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "bitwise equal" in result.stdout
    print(result.stdout)


PROBE = r"""
#include "scf/cuda/direct_force_low_order_sources.cuh"
#include "scf/cuda/direct_force_execution.cuh"
#include "scf/cuda/direct_force_low_order.cuh"
#include "scf/cuda/direct_force_order2.cuh"
#include "scf/cuda/direct_force_order3.cuh"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace generativeqc::scf::cuda_execution;
namespace weighted = generativeqc::scf::generated_weighted_eri;
unsigned long long comparisons = 0, controls = 0, shared = 0, force_calls = 0;

struct Fixture {
  DeviceBatch batch{};
  std::array<std::int32_t, 4> atoms{0,1,2,3};
  std::array<std::uint8_t, 4> angular;
  std::array<std::int64_t, 5> ao_offsets{};
  std::array<std::int32_t, 10> pair_first{}, pair_second{}, pair_system{};
  std::array<std::int64_t, 11> pair_offsets{};
  std::array<std::int64_t, 2> system_offsets{0,4}, system_pair_offsets{0,10};
  std::array<double, 12> positions{0.1,-0.2,0.3, 0.8,0.4,-0.7,
                                  -0.3,0.9,0.2, 0.5,-0.6,1.1};
  std::vector<PrimitivePairData> primitives;
  std::vector<double> coefficients, density, schwarz;

  Fixture(const std::array<std::uint8_t,4>& momenta, unsigned center_mode,
          unsigned density_mode, bool unrestricted) : angular(momenta) {
    if (center_mode == 1) atoms = {0,0,1,2};
    if (center_mode == 2) atoms = {0,1,1,0};
    if (center_mode == 3) atoms = {0,0,0,0};
    for (unsigned shell = 0; shell < 4; ++shell)
      ao_offsets[shell+1] = ao_offsets[shell] + (angular[shell]+1)*(angular[shell]+2)/2;
    const std::size_t dimension = ao_offsets.back();
    const std::size_t matrix = dimension*dimension;
    coefficients.resize(dimension);
    for (std::size_t index = 0; index < dimension; ++index)
      coefficients[index] = 0.7 + 0.03*index;
    schwarz.resize(matrix);
    density.resize((unrestricted ? 2 : 1)*matrix);
    for (std::size_t index = 0; index < matrix; ++index) {
      const auto row = index/dimension, column = index%dimension;
      schwarz[index] = 0.1 + 0.13*((row+column)%11);
      double alpha = std::cos(0.31*row + 0.17*column)/dimension;
      double beta = std::sin(0.27*row - 0.23*column)/dimension;
      if (density_mode == 1) beta = -alpha;
      if (density_mode == 2) alpha = beta = 0;
      if (density_mode == 3 && row != column) alpha = beta = 0;
      density[index] = alpha;
      if (unrestricted) density[matrix+index] = beta;
    }
    unsigned pair = 0;
    for (unsigned first = 0; first < 4; ++first)
      for (unsigned second = 0; second <= first; ++second, ++pair) {
        pair_first[pair] = first;
        pair_second[pair] = second;
        pair_offsets[pair] = primitives.size();
        for (unsigned bra = 0; bra < 2; ++bra)
          for (unsigned ket = 0; ket < 2; ++ket) {
            const double alpha = 0.5 + 0.13*first + 0.2*bra;
            const double beta = 0.5 + 0.13*second + 0.2*ket;
            const double exponent = alpha+beta, reduced = alpha*beta/exponent;
            const unsigned first_atom = atoms[first], second_atom = atoms[second];
            double product[3], squared = 0;
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double first_position = positions[3*first_atom+axis];
              const double second_position = positions[3*second_atom+axis];
              product[axis] = (alpha*first_position + beta*second_position)/exponent;
              squared += (first_position-second_position)*(first_position-second_position);
            }
            primitives.push_back({exponent, reduced, {product[0],product[1],product[2]},
                                  (0.7+0.1*bra)*(0.7+0.1*ket)*std::exp(-reduced*squared),
                                  alpha/exponent, beta/exponent});
          }
      }
    pair_offsets.back() = primitives.size();
    batch.batch_size = 1;
    batch.direct_nbf = dimension;
    batch.total_atoms = 4;
    batch.positions = positions.data();
    batch.shell_atoms = atoms.data();
    batch.shell_angular = angular.data();
    batch.shell_direct_ao_offsets = ao_offsets.data();
    batch.shell_pair_first = pair_first.data();
    batch.shell_pair_second = pair_second.data();
    batch.shell_pair_systems = pair_system.data();
    batch.shell_pair_primitive_offsets = pair_offsets.data();
    batch.shell_primitive_pairs = primitives.data();
    batch.system_shell_offsets = system_offsets.data();
    batch.system_shell_pair_offsets = system_pair_offsets.data();
    batch.direct_ao_coefficients = coefficients.data();
  }
};

template<bool Unrestricted>
void check(Fixture& fixture, ActiveShellQuartetTile task, double screening,
           double coulomb, double exchange, std::uint8_t active) {
  const auto& batch = fixture.batch;
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[batch.shell_pair_first[task.first_pair]],
      batch.shell_angular[batch.shell_pair_second[task.first_pair]],
      batch.shell_angular[batch.shell_pair_first[task.second_pair]],
      batch.shell_angular[batch.shell_pair_second[task.second_pair]]);
  if (direct_shell_class_angular_order(shell_class) > 3) return;
  std::array<double,24> expected{}, actual{};
  const double* bounds = fixture.schwarz.data();
  const double* density = fixture.density.data();
  weighted::geometry_calls = weighted::force_calls = 0;
  for (unsigned source = 0; source < 2; ++source) {
    const double source_j = source == 0 ? coulomb : 0;
    const double source_k = source == 1 ? exchange : 0;
    if (source_j == 0 && source_k == 0) continue;
    double* output = expected.data()+12*source;
    const unsigned order = direct_shell_class_angular_order(shell_class);
    if (order == 0)
      contract_two_electron_force_ssss_task_scaled<Unrestricted>(
          batch,task,screening,bounds,density,&active,output,source_j,source_k);
    else if (order == 1)
      contract_two_electron_force_psss_task_scaled<Unrestricted>(
          batch,task,screening,bounds,density,&active,output,0,source_j,source_k);
    else if (order == 2) {
      contract_two_electron_force_psps_task_scaled<Unrestricted>(
          batch,task,screening,bounds,density,&active,output,0,source_j,source_k);
      contract_two_electron_force_pair_order2_task_scaled<Unrestricted,kPpssShellClass>(
          batch,task,screening,bounds,density,&active,output,0,source_j,source_k);
      contract_two_electron_force_pair_order2_task_scaled<Unrestricted,kDsssShellClass>(
          batch,task,screening,bounds,density,&active,output,0,source_j,source_k);
    } else
      contract_two_electron_force_order3_task_scaled<Unrestricted>(
          batch,task,screening,bounds,density,&active,output,0,source_j,source_k);
  }
  const auto control_geometry = weighted::geometry_calls;
  const auto control_forces = weighted::force_calls;
  weighted::geometry_calls = weighted::force_calls = 0;
  contract_direct_force_precontracted_task<Unrestricted,DirectForceOutputMode::Separate>(
      batch,task,screening,bounds,density,&active,actual.data(),0,coulomb,exchange);
  for (std::size_t coordinate = 0; coordinate < actual.size(); ++coordinate) {
    if (!std::isfinite(actual[coordinate]) ||
        std::bit_cast<std::uint64_t>(actual[coordinate]) !=
            std::bit_cast<std::uint64_t>(expected[coordinate])) {
      std::fprintf(stderr,"class=%u spin=%u pairs=%u,%u coordinate=%zu: %.17g != %.17g\n",
                   shell_class,Unrestricted,task.first_pair,task.second_pair,coordinate,
                   actual[coordinate],expected[coordinate]);
      std::exit(1);
    }
    ++comparisons;
  }
  if (control_forces != weighted::force_calls ||
      control_geometry != control_forces || weighted::geometry_calls > control_geometry ||
      (control_geometry == 32 && weighted::geometry_calls != 16) ||
      (control_geometry <= 16 && weighted::geometry_calls != control_geometry)) {
    std::fprintf(stderr,"incorrect shared-source work for class %u: %llu/%llu vs %llu/%llu\n",
                 shell_class,weighted::geometry_calls,weighted::force_calls,
                 control_geometry,control_forces);
    std::exit(1);
  }
  controls += control_geometry;
  shared += weighted::geometry_calls;
  force_calls += weighted::force_calls;

  // An independent historical HF consumer checks the Combined layout. It
  // receives the original signed scales once, without a method-dependent
  // factor or a reduction of separately rounded J/K output arrays.
  std::array<double,12> combined_control{}, combined{};
  const unsigned order = direct_shell_class_angular_order(shell_class);
  if (order == 0)
    contract_two_electron_force_ssss_task_scaled<Unrestricted>(
        batch,task,screening,bounds,density,&active,combined_control.data(),coulomb,exchange);
  else if (order == 1)
    contract_two_electron_force_psss_task_scaled<Unrestricted>(
        batch,task,screening,bounds,density,&active,combined_control.data(),0,coulomb,exchange);
  else if (order == 2) {
    contract_two_electron_force_psps_task_scaled<Unrestricted>(
        batch,task,screening,bounds,density,&active,combined_control.data(),0,coulomb,exchange);
    contract_two_electron_force_pair_order2_task_scaled<Unrestricted,kPpssShellClass>(
        batch,task,screening,bounds,density,&active,combined_control.data(),0,coulomb,exchange);
    contract_two_electron_force_pair_order2_task_scaled<Unrestricted,kDsssShellClass>(
        batch,task,screening,bounds,density,&active,combined_control.data(),0,coulomb,exchange);
  } else
    contract_two_electron_force_order3_task_scaled<Unrestricted>(
        batch,task,screening,bounds,density,&active,combined_control.data(),0,coulomb,exchange);
  contract_direct_force_precontracted_task<Unrestricted,DirectForceOutputMode::Combined>(
      batch,task,screening,bounds,density,&active,combined.data(),0,coulomb,exchange);
  for (unsigned coordinate = 0; coordinate < 12; ++coordinate) {
    if (std::bit_cast<std::uint64_t>(combined[coordinate]) !=
        std::bit_cast<std::uint64_t>(combined_control[coordinate])) {
      std::fprintf(stderr,"combined class=%u spin=%u coordinate=%u: %.17g != %.17g\n",
                   shell_class,Unrestricted,coordinate,combined[coordinate],
                   combined_control[coordinate]);
      std::exit(3);
    }
    ++comparisons;
  }
  if (shell_class == kPsssShellClass) {
    // Resident p-s records are oriented exactly as in the immutable global
    // cache. Test every p-shell slot and a partial lease that must fall back.
    const bool first_is_bra =
        fixture.angular[fixture.pair_first[task.first_pair]] +
        fixture.angular[fixture.pair_second[task.first_pair]] == 1;
    const auto bra = first_is_bra ? task.first_pair : task.second_pair;
    const auto begin = fixture.pair_offsets[bra];
    const auto count = fixture.pair_offsets[bra+1] - begin;
    std::vector<PrimitivePairData> resident(fixture.primitives.begin()+begin,
                                          fixture.primitives.begin()+begin+count);
    for (auto capacity : {count,count-1}) {
      auto leased = resident;
      if (capacity != count) {
        leased.resize(capacity);
        // A partial view is both smaller and deliberately different. Matching
        // forces must therefore come from the global-cache fallback, not from
        // accidentally reading identical resident records despite bad capacity.
        for (auto& primitive : leased) primitive.weighted_coefficient *= 7.0;
      }
      std::array<double,24> resident_separate{};
      std::array<double,12> resident_combined{};
      contract_two_electron_force_low_order_sources_task<Unrestricted,kPsssShellClass,false,
          LowOrderSourceRoots<kPsssShellClass>,DirectForceOutputMode::Separate,true>(
          batch,task,screening,bounds,density,&active,resident_separate.data(),coulomb,exchange,
          0.0,leased.data(),capacity);
      contract_two_electron_force_low_order_sources_task<Unrestricted,kPsssShellClass,false,
          LowOrderSourceRoots<kPsssShellClass>,DirectForceOutputMode::Combined,true>(
          batch,task,screening,bounds,density,&active,resident_combined.data(),coulomb,exchange,
          0.0,leased.data(),capacity);
      if (resident_separate != actual || resident_combined != combined) std::exit(4);
    }
  }
}

template<bool Unrestricted>
void campaign() {
  for (std::uint8_t first = 0; first <= 3; ++first)
    for (std::uint8_t second = 0; second <= 3-first; ++second)
      for (std::uint8_t third = 0; third <= 3-first-second; ++third)
        for (std::uint8_t fourth = 0; fourth <= 3-first-second-third; ++fourth)
          for (unsigned centers = 0; centers < 4; ++centers)
            for (unsigned density = 0; density < 4; ++density) {
              Fixture fixture({first,second,third,fourth},centers,density,Unrestricted);
              for (std::uint32_t bra = 0; bra < 10; ++bra)
                for (std::uint32_t ket = 0; ket <= bra; ++ket)
                  for (double screening : {0.0,0.6,2.0})
                    for (const auto scales : {std::array<double,2>{1.7,-0.23},
                                             {0.0,-0.23},{1.7,0.0},{0.0,0.0}})
                      check<Unrestricted>(fixture,{bra,ket,0},screening,scales[0],scales[1],1);
              check<Unrestricted>(fixture,{9,4,0},0,1.7,-0.23,0);
              check<Unrestricted>(fixture,{9,4,1},0,1.7,-0.23,1);
            }
}
int main() {
  campaign<false>();
  campaign<true>();
  if (!(controls > shared && shared > 0 && controls == force_calls)) return 2;
  std::printf("%llu coordinates bitwise equal; geometry %llu -> %llu; force calls %llu\n",
               comparisons,controls,shared,force_calls);
}
"""
