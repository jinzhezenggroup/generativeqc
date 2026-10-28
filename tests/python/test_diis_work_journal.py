"""Execute the terminal receipt publisher; no CUDA context or source matching."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_terminal_diis_receipts(tmp_path: Path, required_native_cxx: NativeCxx) -> None:
    source = tmp_path / "diis_work.cpp"
    source.write_text(
        r"""
#include "scf/diis_work.hpp"
#include <cassert>
using namespace generativeqc::scf;
int main(int argc, char** argv) {
  assert(argc == 2);
  constexpr auto variable = "GENERATIVEQC_DF_PROGRESS_TRACE";
  unsetenv(variable);
  assert(!diis_work_requested());
  publish_diis_work({}, "RHF", 1, 1, 0, 3, 1, 4, true, true, true, false);
  setenv(variable, "", 1);
  assert(!diis_work_requested());
  setenv(variable, argv[1], 1);
  assert(diis_work_requested());
  // Work is cumulative even when the live history has retired older entries.
  DiisWorkReadback receipt{{17, 153}, 2, 3};
  publish_diis_work(receipt, "RHF", 7, 2, 1, 3, 1, 4, true, true, true, false);
  publish_diis_work({{4, 1048576}, 1, 0}, "RKS", 8, 3, 0, 512, 1, 4,
                    true, true, false, true);
  for (bool incremental : {false, true})
    publish_diis_work({{}, 2, 3}, "UKS", 9, 4, 0, 3, 2, 4,
                      false, incremental, true, false);
  receipt.work.vector_elements = 152;
  publish_diis_work(receipt, "RHF", 10, 5, 0, 3, 1, 4, true, true, false, true);
  publish_diis_work({}, "RHF", 11, 6, 0, std::numeric_limits<std::size_t>::max(),
                    2, 4, true, true, false, true);
  publish_diis_work({{}, 5, 0}, "RHF", 12, 7, 0, 3, 1, 4, true, true, false, true);
  publish_diis_work({}, "RHF", 13, 8, 0, 3, 1, 0, true, true, false, true);
}
"""
    )
    executable = tmp_path / "diis_work"
    required_native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", "-I" + str(ROOT / "src"))
    )
    journal = tmp_path / "work.jsonl"
    subprocess.run([str(executable), str(journal)], check=True, timeout=30)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    scopes = {}
    for record in records:
        assert record["name"] == "cuda_diis_work"
        assert record["execution"] == "cuda_completed"
        scopes.setdefault(record["id"], []).append(record)
    assert len(scopes) == 8  # Disabled publication did not create a scope.
    values = []
    for events in scopes.values():
        assert events[0]["event"] == "BEGIN" and events[-1]["event"] == "END"
        values.append({e["key"]: e["value"] for e in events if e["event"] == "VALUE"})
    assert values[0]["gram_dot_reductions"] == 17
    assert values[0]["residual_vector_elements"] == 153
    assert values[0]["live_history"] == 2
    assert values[0]["owner"] == 7 and values[0]["solve_epoch"] == 2
    assert values[0]["device_counter_bytes"] == 16
    assert values[0]["host_readback_bytes"] == 24
    assert values[1]["route"] == "ordered-staged" and values[1]["failed"] == 1
    assert [v["route"] for v in values[2:4]] == ["serial", "incremental-unordered"]
    for value in values[2:]:
        assert value["work_coverage"] == "unavailable"
        assert "gram_dot_reductions" not in value
    for events in list(scopes.values())[4:]:
        assert events[-1]["status"] == "invalid-history-or-work"
