"""Compile the native phased launch sequence with explicit failure injection."""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_bulk_geometry_launches_preserve_sticky_status_and_fail_closed(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Preserve launch failures and count each successfully submitted producer."""
    header = (
        Path(__file__).resolve().parents[2] / "src/dft/stationary_gradient_cuda.cuh"
    ).read_text()
    branch = _block(header, "if (precomputed_point) {")
    branch, replacements = re.subn(
        r"(geometry_point_kernel|geometry_cooperative_kernel<true>)\s*<<<.*?>>>\(.*?\);",
        lambda match: (
            "producer_launch();"
            if match.group(1) == "geometry_point_kernel"
            else "consumer_launch();"
        ),
        branch,
        flags=re.DOTALL,
    )
    assert replacements == 2
    source = tmp_path / "launch.cpp"
    source.write_text(
        PREFIX
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
int sticky_error, producer_error, consumer_error;
int producers, consumers, peeks;
struct { int launches; } owner;
int cudaPeekAtLastError() { ++peeks; return sticky_error; }
void cuda_check(int error) { if(error) throw error; }
void producer_launch() { ++producers; sticky_error=producer_error; }
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
}
"""
