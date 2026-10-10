"""Compile the native phased launch sequence with explicit failure injection."""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("capable", (False, True))
def test_bulk_geometry_launches_preserve_sticky_status_and_fail_closed(
    tmp_path: Path, native_cxx: "NativeCxx", capable: bool
) -> None:
    """Preserve launch failures and count each successfully submitted producer."""
    header = (
        Path(__file__).resolve().parents[2] / "src/dft/stationary_gradient_cuda.cuh"
    ).read_text()
    branch = _block(header, "if (precomputed_point) {")
    branch, replacements = re.subn(
        r"(geometry_point_kernel<(?:true|false)>|geometry_cooperative_kernel<true>)\s*<<<.*?>>>\(.*?\);",
        lambda match: (
            f"producer_launch({str(match.group(1) == 'geometry_point_kernel<true>').lower()});"
            if match.group(1).startswith("geometry_point_kernel<")
            else "consumer_launch();"
        ),
        branch,
        flags=re.DOTALL,
    )
    assert replacements == 4
    source = tmp_path / "launch.cpp"
    source.write_text(
        PREFIX
        + f"constexpr bool stationary_pbe0_restricted_point_capable={str(capable).lower()};\n"
        + "void execute(bool precomputed_point=true) {\n"
        + branch
        + "}\n"
        + DRIVER
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "launch", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


PREFIX = r"""
#include <cassert>
#include <initializer_list>
int sticky_error, producer_error, consumer_error;
int producers, consumers, peeks;
bool restricted_point, selected_restricted;
void* external;
struct { int npoint=3; } view;
struct { int launches,restricted_point_batches,restricted_point_count; } owner;
int cudaPeekAtLastError() { ++peeks; return sticky_error; }
void cuda_check(int error) { if(error) throw error; }
void producer_launch(bool restricted) {
  ++producers; selected_restricted=restricted; sticky_error=producer_error;
}
void consumer_launch() { ++consumers; sticky_error=consumer_error; }
"""

DRIVER = r"""
int main() {
  for(int failure=0;failure<4;++failure) {
    sticky_error=failure==1?71:0;
    producer_error=failure==2?72:0;
    consumer_error=failure==3?73:0;
    producers=consumers=peeks=0;
    owner.launches=19;
    restricted_point=true;
    int caught=0;
    try { execute(); } catch(int error) { caught=error; }
    assert(caught==(failure?70+failure:0));
    assert(sticky_error==caught);
    assert(producers==(failure==1?0:1));
    assert(consumers==((failure==1 || failure==2)?0:1));
    assert(peeks==(failure==1?1:failure==2?2:3));
    assert(owner.launches==19+((failure==0 || failure==3)?1:0));
  }
  sticky_error=producer_error=consumer_error=0;
  owner.launches=0;
  for(int batch=1;batch<=3;++batch) {
    execute();
    assert(owner.launches==batch);
  }
  producers=consumers=peeks=0;
  execute(false);
  assert(owner.launches==3 && !producers && !consumers && !peeks);
  for(bool requested:{false,true}) {
    for(bool seeded:{false,true}) {
      restricted_point=requested;
      external=seeded?&view:nullptr;
      owner.restricted_point_batches=owner.restricted_point_count=0;
      execute();
      const bool expected=requested && !seeded && stationary_pbe0_restricted_point_capable;
      assert(selected_restricted==expected);
      assert(owner.restricted_point_batches==int(expected));
      assert(owner.restricted_point_count==3*int(expected));
    }
  }
}
"""
