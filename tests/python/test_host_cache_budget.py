"""Execute optional RAM admission against independent machine/job fixtures."""

import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def budget_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("host-cache-budget")
    source = directory / "probe.cpp"
    source.write_text(
        '#include <iostream>\n#include "runtime/host_cache_budget.hpp"\n'
        "int main(int argc, char** argv) { if(argc != 3) return 1; "
        "std::cout << generativeqc::runtime::automatic_host_cache_bytes(argv[1],argv[2]); }\n"
    )
    executable = directory / "probe"
    native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", f"-I{ROOT / 'src'}")
    )
    return executable


@pytest.mark.parametrize(
    "mode,expected_gib",
    [
        ("unlimited", 32),
        ("job", 8),
        ("ancestor", 2),
        ("exhausted", 0),
        ("malformed", 0),
        ("missing-usage", 0),
        ("missing-group", 0),
        ("missing-memory", 0),
        ("v1", 4),
    ],
)
def test_available_ram_respects_job_ancestors(
    tmp_path: Path, budget_probe: Path, mode: str, expected_gib: int
) -> None:
    """A child with no cap must still obey its enclosing Slurm job limit."""
    proc, groups = tmp_path / "proc", tmp_path / "cgroups"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal: 536870912 kB\nMemAvailable: 268435456 kB\n"
    )
    (proc / "self/cgroup").write_text("0::/job/step\n")
    step = groups / "job/step"
    step.mkdir(parents=True)
    (step / "memory.max").write_text("max\n")
    if mode in {"job", "ancestor", "exhausted", "malformed", "missing-usage"}:
        (step.parent / "memory.max").write_text(str(64 << 30))
        (step.parent / "memory.current").write_text(str(32 << 30))
    if mode == "ancestor":
        (groups / "memory.max").write_text(str(80 << 30))
        (groups / "memory.current").write_text(str(72 << 30))
    if mode == "exhausted":
        (step.parent / "memory.current").write_text(str(65 << 30))
    if mode == "malformed":
        (step.parent / "memory.max").write_text("123garbage")
    if mode == "missing-usage":
        (step.parent / "memory.current").unlink()
    if mode == "missing-group":
        (proc / "self/cgroup").write_text("0::/unavailable\n")
    if mode == "missing-memory":
        (proc / "meminfo").unlink()
    if mode == "v1":
        (proc / "self/cgroup").write_text("8:cpu,cpuacct:/job\n7:memory:/job\n")
        job = groups / "memory/job"
        job.mkdir(parents=True)
        (job / "memory.limit_in_bytes").write_text(str(32 << 30))
        (job / "memory.usage_in_bytes").write_text(str(16 << 30))
    result = subprocess.run(
        [str(budget_probe), str(proc), str(groups)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout) == expected_gib << 30
