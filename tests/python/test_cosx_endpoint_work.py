"""Host-only gate for the CUDA endpoint harness's full/tail work assertions."""

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner


def test_endpoint_work_uses_effective_tile(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    harness = (root / "tests/native/cosx_endpoint_benchmark.cuh").read_text()
    runtime = (root / "src/dft/cuda_cosx.cu").read_text()
    # Execute the real assertion block and production clamp/site expressions,
    # without compiling CUDA or replacing its numerical work with a CPU path.
    checks = re.findall(
        r"(const auto& info = plans\[mask\]->diagnostic\(\);.*?)"
        r"\s+std::cout << std::setprecision\(17\)",
        harness,
        re.DOTALL,
    )
    clamps = re.findall(r"const std::size_t tile_points = [^;]+;", runtime)
    sites = re.findall(r"const std::size_t site = count == tile_points[^;]+;", runtime)
    assert len(checks) == len(clamps) == len(sites) == 1
    source = tmp_path / "endpoint_work.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Site {
  struct { std::string_view provider; } candidate;
  std::size_t calls{}, summands{};
};
struct Info {
  std::size_t tile_points{}, provider_allowance{};
  std::array<Site, 4> contractions;
};
struct Plan {
  Info info;
  const Info& diagnostic() const { return info; }
};
std::size_t effective_tile(std::size_t npoint, std::size_t requested_tile) {
  CLAMP
  return tile_points;
}
int main() {
  constexpr std::size_t repeats = 6, n = 13;
  // Oversized, exactly one tile, exact division, and nonempty tail shapes.
  constexpr std::array<std::array<std::size_t, 2>, 5> shapes{{
      {3, 4}, {1, 4096}, {3, 3}, {6, 3}, {7, 3}}};
  for (const auto shape : shapes) {
    const auto points = shape[0], tile = shape[1];
    const auto tile_points = effective_tile(points, tile);
    std::array<Plan, 4> owners;
    std::array<Plan*, 4> plans;
    for (unsigned mask = 0; mask < 4; ++mask) {
      plans[mask] = &owners[mask];
      auto& info = owners[mask].info;
      info.tile_points = tile_points;
      info.provider_allowance = mask ? 96ULL << 20 : 0;
      for (unsigned slot = 0; slot < 4; ++slot)
        info.contractions[slot].candidate.provider =
            mask & (1U << (slot % 2)) ? "cublas" : "generated.cuda";
      // Count executed tiles independently, rather than copying the harness's
      // division/remainder formulas into the expected diagnostics.
      for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        for (std::size_t begin = 0; begin < points; begin += tile_points) {
          const auto count = std::min(tile_points, points - begin);
          SITE
          for (const auto slot : {site, site + 1}) {
            ++info.contractions[slot].calls;
            info.contractions[slot].summands += count * n * n;
          }
        }
    }
    for (unsigned mask = 0; mask < 4; ++mask) {
      try {
        CHECKS
      } catch (const std::exception& error) {
        std::cerr << points << ' ' << tile << ' ' << mask << ": " << error.what();
        return 1;
      }
    }
  }
}
""".replace("CLAMP", clamps[0])
        .replace("SITE", sites[0])
        .replace("CHECKS", checks[0])
    )
    binary = tmp_path / "endpoint_work"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


RECEIPTS = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/cosx-contractions-1884"
)


def test_retained_endpoint_receipts_still_verify() -> None:
    result = subprocess.run(
        [sys.executable, str(RECEIPTS / "verify.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified 32 records / 192 endpoint samples" in result.stdout


@pytest.mark.parametrize("effective", [None, 144, 256, 0, -1, 144.5, True])
def test_receipt_verifier_handles_effective_tile(
    tmp_path: Path, effective: float | None
) -> None:
    # Synthetic accounting records, not new measurements. Start from one
    # retained four-route group and use a 48-atom, 1x1x3 grid: 144 < tile 256.
    records = json.loads(gzip.decompress((RECEIPTS / "endpoints.json.gz").read_bytes()))
    records = [
        r
        for r in records
        if r["atoms"] == 48 and r["tile"] == 256 and r["geometry"] == 0
    ]
    assert len(records) == 4
    for record in records:
        record["grid"], record["points"] = [1, 1, 3], 144
        if effective is not None:
            record["effective_tile"] = effective
        n = record["nao"]
        for slot, site in enumerate(record["sites"]):
            site["calls"] = 6 if slot < 2 else 0
            site["summands"] = site["calls"] * 144 * n * n
            site["m"], site["n"], site["k"] = (
                (144, n, n) if slot % 2 == 0 else (n, n, 144)
            )
    data = gzip.compress(json.dumps(records).encode(), mtime=0)
    (tmp_path / "endpoints.json.gz").write_bytes(data)
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    provenance["records"] = len(records)
    provenance["data_sha256"] = {"endpoints.json.gz": hashlib.sha256(data).hexdigest()}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    shutil.copyfile(RECEIPTS / "verify.py", tmp_path / "verify.py")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "verify.py"), "--write-summary"],
        capture_output=True,
        text=True,
        check=False,
    )
    if effective is None or type(effective) is int and effective == 144:
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary[0]["tile"] == 256
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr
