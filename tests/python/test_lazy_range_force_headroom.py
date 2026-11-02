"""Exercise native deferral admission under a deterministic driver memory budget.

This host contract covers the boundary and driver failures. Complete scheduled
E+F populations must separately establish that the method supplies its real force
envelope and that the first warm endpoint does not inherit cold preparation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_ks_warm_factor import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def headroom_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Replace only resource/provider observations around the real admission."""
    source = (Path(__file__).resolve().parents[2] / "src/dft/cuda_ks.cpp").read_text()
    method = _definition(source, "  bool defer_range_warm_energy()")
    harness = r"""
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
constexpr int GENERATIVEQC_PRECISION_FP64=0;
namespace scf { struct ScfOptions { enum class XcExecutionSchedule {DeviceFused,Host}; }; }
std::size_t free_bytes=1024, queries=0; bool driver_failure=false;
int cudaMemGetInfo(std::size_t* free,std::size_t* total) {
  ++queries; if(driver_failure)return 1; *free=free_bytes;*total=4096;return 0;
}
void check(int s) {if(s)throw std::runtime_error("driver memory query failed");}
struct Owner {
  bool active=true, replay=true, fitted_coulomb=true, fitted_exchange=true;
  bool warm_factor_storage=true;int spins=1;std::size_t following_force_device_bytes=0;
  struct Precision {bool lower=false;bool any_lower_precision()const{return lower;}} precision_schedule;
  struct Options {std::optional<int> precision_mode=0;
    scf::ScfOptions::XcExecutionSchedule xc_execution_schedule=scf::ScfOptions::XcExecutionSchedule::DeviceFused;
  } options;
  bool range_source_active()const{return active;}
  bool retain_range_source_for_replay(bool use_warm)const{return replay&&use_warm;}
  METHOD
};
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  const std::string mode=argv[1];Owner owner;
  if(mode=="fits")owner.following_force_device_bytes=1023;
  if(mode=="equal")owner.following_force_device_bytes=1024;
  if(mode=="pressure")owner.following_force_device_bytes=1025;
  if(mode=="driver-failure") {owner.following_force_device_bytes=1;driver_failure=true;}
  if(mode=="no-replay") {owner.replay=false;owner.following_force_device_bytes=1025;}
  if(mode=="mixed") {owner.precision_schedule.lower=true;owner.following_force_device_bytes=1025;}
  try {bool defer=owner.defer_range_warm_energy();std::cout<<defer<<" "<<queries<<"\n";}
  catch(const std::runtime_error&) {std::cout<<"error "<<queries<<"\n";}
}
""".replace("METHOD", method)
    folder = tmp_path_factory.mktemp("lazy-headroom")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(harness)
    native_cxx.build_executable(
        [cpp], executable, compile_args=("-std=c++17", "-Wall", "-Werror")
    )
    return executable


@pytest.mark.parametrize(
    ("mode", "enabled", "expected"),
    [
        ("fits", None, "1 1"),
        ("equal", None, "1 1"),
        ("pressure", None, "0 1"),
        ("energy-only", None, "1 0"),
        ("driver-failure", None, "error 1"),
        ("no-replay", None, "0 0"),
        ("mixed", None, "0 0"),
        ("fits", "1", "1 1"),
        ("equal", "1", "1 1"),
        ("pressure", "1", "0 1"),
        ("energy-only", "1", "1 0"),
        ("driver-failure", "1", "error 1"),
        ("no-replay", "1", "0 0"),
        ("mixed", "1", "0 0"),
        ("fits", "0", "0 0"),
        ("fits", "invalid", "0 0"),
    ],
)
def test_imminent_force_headroom(
    headroom_probe: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    enabled: str | None,
    expected: str,
) -> None:
    """Insufficient force headroom keeps eager work; driver errors never select it."""
    if enabled is None:
        monkeypatch.delenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", enabled)
    result = subprocess.run(
        [str(headroom_probe), mode], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == expected
