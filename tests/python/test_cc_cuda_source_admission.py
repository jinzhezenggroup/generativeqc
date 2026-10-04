"""Host-execute native-reference dispatch and optional CUDA source admission."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, program: str) -> None:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler unavailable")
    source, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text(program)
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_provider_schedule_checks_attempt_deltas(tmp_path: Path) -> None:
    text = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    assert "const auto initial_work = provider_work;" in text
    validation = text[
        text.index(
            "  const auto source_scans = provider_work.source_scans"
        ) : text.index("  p.provider_peak_bytes =")
    ]
    _run(
        tmp_path,
        r"""
#include <vector>
#include "posthf/native_provider.hpp"
using namespace generativeqc;
void validate(const posthf::ProviderWork& initial_work, const posthf::ProviderWork& provider_work) {
  constexpr std::size_t n=2;
  struct {std::vector<int> batches{0};} reuse;
  struct {std::size_t source_reads=4;} source_tile_plan;
"""
        + validation
        + r"""
}
int main() {
  posthf::ProviderWork before{}, after{};
  before.source_scans=2; before.source_reads=7; before.source_values=27;
  after=before; after.source_scans+=1; after.source_reads+=4; after.source_values+=16;
  validate(before,after);
  if (after.source_scans!=3 || after.source_reads!=11 || after.source_values!=43) return 1;
  for (int field=0;field<3;++field) {
    auto invalid=after;
    if (field==0) ++invalid.source_scans;
    if (field==1) ++invalid.source_reads;
    if (field==2) ++invalid.source_values;
    bool refused=false;
    try { validate(before,invalid); } catch (const std::logic_error&) { refused=true; }
    if (!refused) return 2;
  }
}
""",
    )
