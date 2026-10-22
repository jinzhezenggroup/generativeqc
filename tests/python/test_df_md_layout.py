"""Executable admission boundaries for the optional native DF Hermite layout."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
UNLIMITED = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


@pytest.fixture(scope="module")
def layout_probe(tmp_path_factory: Any, required_native_cxx: NativeCxx) -> Any:
    folder = tmp_path_factory.mktemp("df-md-layout")
    source, library = folder / "probe.cpp", folder / "probe.so"
    source.write_text(_PROBE)
    required_native_cxx.build_shared(
        [source, ROOT / "src/molecule/basis.cpp"],
        library,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )
    owner = ctypes.CDLL(str(library))
    owner.layout.argtypes = (
        [ctypes.c_uint] * 4
        + [ctypes.c_size_t] * 2
        + [np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")]
    )
    owner.layout.restype = ctypes.c_int
    return owner


@pytest.mark.parametrize("angular", range(5))
@pytest.mark.parametrize("spherical", (False, True))
@pytest.mark.parametrize("auxiliary", (False, True))
def test_optional_capacity_and_full_angular_domain(
    layout_probe: Any, angular: int, spherical: bool, auxiliary: bool
) -> None:
    result = np.zeros(8, dtype=np.uint64)
    status = layout_probe.layout(
        angular, spherical, auxiliary, 0, UNLIMITED, UNLIMITED, result
    )
    if angular == 4 and not auxiliary:
        assert status == 2
        return
    assert status == 0
    device, host = map(int, result[:2])
    assert device > 0 and host > 0
    assert int(result[6]) == device  # The pre-allocation reservation is exact.
    # There are two shells with 2/3 primitives. J enumerates all ordered
    # primitive products within a self shell and each cross-shell pair once.
    assert result[2] == (5 if auxiliary else 4 + 6 + 9)
    assert result[3] == 2 * (
        2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2
    )
    assert (
        layout_probe.layout(angular, spherical, auxiliary, 0, device, host, result) == 0
    )
    for limits in ((device - 1, host), (device, host - 1), (0, host), (device, 0)):
        assert (
            layout_probe.layout(angular, spherical, auxiliary, 0, *limits, result) == 1
        )


@pytest.mark.parametrize("mutation", range(1, 8))
def test_invalid_metadata_fails_before_device_preparation(
    layout_probe: Any, mutation: int
) -> None:
    result = np.zeros(8, dtype=np.uint64)
    assert (
        layout_probe.layout(2, True, False, mutation, UNLIMITED, UNLIMITED, result) == 2
    )


_PROBE = r"""
#include <limits>
#include "scf/cuda/df_md_layout.hpp"
using namespace generativeqc;
using namespace generativeqc::scf::cuda_execution;
extern "C" int layout(unsigned angular, unsigned spherical, unsigned auxiliary, unsigned mutation,
                       std::size_t device, std::size_t host, std::uint64_t* output) {
  try {
    core::System system;
    system.atoms = {{2, {0.13,-0.31,0.24}}, {2, {-0.43,0.27,0.51}}};
    system.shells = {{0,angular,{{0.57,0.83},{1.28,-0.17}}},
                     {1,angular,{{0.8,0.83},{1.51,-0.17},{2.22,-0.085}}}};
    system.basis_representation = spherical ? GENERATIVEQC_BASIS_SPHERICAL
                                            : GENERATIVEQC_BASIS_CARTESIAN;
    std::string detail;
    if (molecule::validate_and_normalize(system,detail)!=GENERATIVEQC_STATUS_SUCCESS) return 3;
    switch(mutation) {
      case 1: system.shells.clear(); break;
      case 2: system.shells[0].atom_index=system.atoms.size(); break;
      case 3: system.shells[0].primitives.clear(); break;
      case 4: system.shells[0].primitives[0].exponent=0; break;
      case 5: system.shells[0].primitives[0].coefficient=std::numeric_limits<double>::quiet_NaN(); break;
      case 6: system.atoms[0].position[0]=std::numeric_limits<double>::infinity(); break;
      case 7: system.shells[0].primitives[0].coefficient=1e308; break;
    }
    output[6]=df_md_layout_device_bytes(system,auxiliary,device,host);
    const auto prepared=make_df_md_layout(system,auxiliary,device,host);
    output[0]=prepared.device_bytes; output[1]=prepared.host_peak_bytes;
    output[2]=prepared.primitives.size(); output[3]=prepared.aos.size();
    output[4]=prepared.transform_elements; output[5]=prepared.hermite_elements;
    // Independently check the upload boundary: component and Hermite ranges
    // are contiguous, disjoint and terminate at their advertised capacities.
    std::size_t e=0,h=0;
    for (const auto& p:prepared.primitives) {
      if (p.transform_offset!=e || p.hermite_offset!=h) return 4;
      const auto l=p.first_angular+p.second_angular;
      const auto count=(l+1)*(l+2)*(l+3)/6;
      e+=p.first_count*p.second_count*count; h+=count;
      if (p.second_ao+p.second_count>prepared.aos.size() ||
          (!auxiliary && p.first_ao+p.first_count>prepared.aos.size())) return 4;
    }
    if (e!=prepared.transform_elements || h!=prepared.hermite_elements) return 4;
    return 0;
  } catch (const std::length_error&) { return 1; }
    catch (const std::invalid_argument&) { return 2; }
    catch (...) { return 3; }
}
"""
