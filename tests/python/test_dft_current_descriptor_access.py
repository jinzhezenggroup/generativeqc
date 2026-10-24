"""Compile the DFT grid descriptor reader without legacy prefix helpers.

Minimal dependent types isolate grid descriptor defaults; real auxiliary-system
copy ownership is covered by test_native_api_inputs.cpp through the C API.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _source() -> str:
    return (ROOT / "src/methods/dft_method_preparation.cpp").read_text(encoding="utf-8")


def test_current_dft_readers_do_not_probe_legacy_prefixes() -> None:
    assert "field_present(" not in _source()


def test_current_dft_descriptor_readers_compile_and_keep_defaults(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    text = _source()
    grid = text[
        text.index("dft::GridSpec ks_grid_options(") : text.index(
            "void validate_ks_spin_state("
        )
    ]
    unit = tmp_path / "descriptor.cpp"
    unit.write_text(
        r"""
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>
constexpr unsigned GENERATIVEQC_ABI_VERSION=1;
constexpr int GENERATIVEQC_STATUS_ABI_MISMATCH=2,GENERATIVEQC_STATUS_NOT_IMPLEMENTED=3,GENERATIVEQC_STATUS_INVALID_ARGUMENT=4;
constexpr unsigned GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED=0,GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED=1;
struct MethodError:std::runtime_error { MethodError(int,const char* m):std::runtime_error(m){} };
struct NativeKsExecutionPlan {bool nonlocal_correlation=false;};
const char* expected_scf_domain(const NativeKsExecutionPlan&) {return "current-domain";}
namespace core {struct System {std::vector<int> atoms;};}
namespace dft {
struct GridSpec {
 unsigned version=1,radial_points=4,angular_polar=3,angular_azimuth=2,partition_iterations=3;
 double coincident_tolerance=1e-12;
 std::array<double,119> element_radii{};
 bool operator==(const GridSpec&) const = default;
};
// Distinct sentinels exercise default propagation and legacy reset without
// copying the native grid-construction implementation into this fixture.
GridSpec default_grid_spec() {
 GridSpec grid; grid.version=4; grid.radial_points=3;
 grid.angular_polar=grid.angular_azimuth=0; grid.element_radii.fill(2.0); return grid;
}
void validate_grid_spec(const GridSpec& g) {if(!g.radial_points)throw std::invalid_argument("radial");}
}
namespace scf {struct ScfOptions {
 std::optional<dft::GridSpec> nonlocal_grid_spec;
 enum class XcExecutionSchedule {DeviceFused,HostUnfused};
 std::uint64_t xc_tile_points=256;
 XcExecutionSchedule xc_execution_schedule=XcExecutionSchedule::DeviceFused;
};}
struct generativeqc_ks_options {
 std::size_t struct_size=sizeof(generativeqc_ks_options); unsigned abi_version=1;
 const char* scf_domain="current-domain"; std::uint64_t tile_points=37;
 unsigned xc_execution_schedule=GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED;
 unsigned grid_version=2,radial_points=7,angular_polar=5,angular_azimuth=9,partition_iterations=4;
 double coincident_tolerance=1e-10; const double* element_radii=nullptr; std::size_t element_radius_count=0;
};
namespace dft {struct NonlocalGridKsOptions {
 generativeqc_ks_options base;
 double energy{},maximum_force{},rms_force{};
 std::uint32_t version=1,reserved=0;
 const generativeqc_ks_options* grid=nullptr;
};}
struct generativeqc_method_descriptor {
 std::size_t struct_size=sizeof(generativeqc_method_descriptor);
 const generativeqc_ks_options* ks_options=nullptr;
};
"""
        + grid
        + r"""
int main() {
 generativeqc_method_descriptor d; scf::ScfOptions options; NativeKsExecutionPlan plan;
 auto defaults=ks_grid_options(d,options,plan);
 assert(defaults==dft::default_grid_spec() && options.xc_tile_points==256);
 generativeqc_ks_options input; d.ks_options=&input;
 auto copied=ks_grid_options(d,options,plan);
 assert(copied.version==2 && copied.radial_points==7 && options.xc_tile_points==37);
 assert(copied.element_radii==dft::GridSpec{}.element_radii);
 auto explicit_default=input; explicit_default.grid_version=4; d.ks_options=&explicit_default;
 assert(ks_grid_options(d,options,plan).element_radii==defaults.element_radii);
 d.ks_options=&input;
 assert(options.xc_execution_schedule==scf::ScfOptions::XcExecutionSchedule::HostUnfused);
 unsigned rejected=0;
 for(int mode=0;mode<6;++mode) {
  auto bad=input;
  if(mode==0)bad.struct_size=0;
  if(mode==1)bad.abi_version=0;
  if(mode==2)bad.scf_domain="wrong";
  if(mode==3)bad.tile_points=0;
  if(mode==4)bad.xc_execution_schedule=9;
  if(mode==5)bad.element_radius_count=119;
  d.ks_options=&bad;
  try {(void)ks_grid_options(d,options,plan);}catch(const std::exception&){++rejected;}
 }
 assert(rejected==6);

}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "descriptor"
    native_cxx.build_executable(
        [unit], binary, compile_args=["-std=c++20", "-O0"], compile_timeout=30
    )
    subprocess.run([str(binary)], check=True, timeout=10)
