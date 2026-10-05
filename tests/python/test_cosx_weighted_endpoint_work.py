"""Host-only effective-tile gates for weighted COSX endpoint accounting."""

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

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "benchmarks/results/cosx-weighted-esp-1884"


def test_weighted_harness_uses_production_effective_tile(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    harness = (ROOT / "tests/native/cosx_weighted_endpoint_benchmark.cuh").read_text()
    runtime = (ROOT / "src/dft/cuda_cosx.cu").read_text()
    bounds = re.findall(
        r"constexpr std::size_t kRoutes = \d+, kReplays = \d+;", harness
    )
    assert len(bounds) == 1
    # Execute the actual harness assertions and production clamp/site choice.
    # This is a host accounting probe, not a substitute CUDA numerical path.
    checks = re.findall(
        r"(const auto& info = plans\[route\]->diagnostic\(\);.*?)"
        r"\s+std::cout << std::setprecision\(17\)",
        harness,
        re.DOTALL,
    )
    clamps = re.findall(r"const std::size_t tile_points = [^;]+;", runtime)
    sites = re.findall(r"const std::size_t site = count == tile_points[^;]+;", runtime)
    esp_sites = re.findall(
        r"contractions->execute\((count == tile_points \? 4 : 5),", runtime
    )
    assert len(checks) == len(clamps) == len(sites) == len(esp_sites) == 1
    source = tmp_path / "weighted_endpoint_work.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

BOUND_CONSTANTS
static_assert(kRoutes == 4 && kReplays == 6);

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Site {
  struct { std::string_view provider; } candidate;
  std::size_t calls{}, summands{}, scaled_elements{}, publication_passes{};
};
struct Info {
  std::size_t tile_points{}, provider_allowance{};
  std::array<Site, 6> contractions;
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
  constexpr std::size_t repeats = 6, n = 384;
  constexpr std::array<unsigned, 4> masks{0, 4, 3, 7};
  constexpr std::array<std::array<std::size_t, 2>, 7> shapes{{
      {2592, 4096}, {1, 64}, {64, 64}, {5184, 64},
      {2592, 64}, {2592, 256}, {3, 2}}};
  for (const auto shape : shapes) {
    const auto points = shape[0], tile = shape[1];
    const auto tile_points = effective_tile(points, tile);
    std::array<Plan, 4> owners;
    std::array<Plan*, 4> plans;
    for (std::size_t route = 0; route < masks.size(); ++route) {
      const auto mask = masks[route];
      plans[route] = &owners[route];
      auto& info = owners[route].info;
      info.tile_points = tile_points;
      info.provider_allowance = mask ? 96ULL << 20 : 0;
      for (unsigned slot = 0; slot < 6; ++slot) {
        const auto operation = slot < 4 ? slot % 2 : 2;
        info.contractions[slot].candidate.provider =
            mask & (1U << operation) ? "cublas" : "generated.cuda";
      }
      // Count actual iterations independently of division/remainder formulas.
      for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        for (std::size_t begin = 0; begin < points; begin += tile_points) {
          const auto count = std::min(tile_points, points - begin);
          SITE
          const std::size_t esp_site = ESP_SITE;
          for (const auto slot : {site, site + 1, esp_site}) {
            auto& diagnostic = info.contractions[slot];
            ++diagnostic.calls;
            diagnostic.summands += count * n * n;
            if (slot >= 4) {
              diagnostic.scaled_elements += count * n;
              diagnostic.publication_passes += bool(mask & 4U);
            }
          }
        }
    }
    for (std::size_t route = 0; route < masks.size(); ++route) {
      const auto mask = masks[route];
      try {
        CHECKS
      } catch (const std::exception& error) {
        std::cerr << points << ' ' << tile << ' ' << mask << ": " << error.what();
        return 1;
      }
    }
  }
}
""".replace("BOUND_CONSTANTS", bounds[0])
        .replace("CLAMP", clamps[0])
        .replace("ESP_SITE", esp_sites[0])
        .replace("SITE", sites[0])
        .replace("CHECKS", checks[0])
    )
    binary = tmp_path / "weighted_endpoint_work"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_weighted_harness_reports_requested_and_effective_tile() -> None:
    source = (ROOT / "tests/native/cosx_weighted_endpoint_benchmark.cuh").read_text()
    assert r'\"tile\":" << tile' in source
    assert r'\"effective_tile\":" << info.tile_points' in source


def test_retained_weighted_receipts_still_verify() -> None:
    result = subprocess.run(
        [sys.executable, str(RECEIPTS / "verify.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified 32 records / 192 endpoint samples" in result.stdout


@pytest.mark.parametrize("oversized", [False, True])
@pytest.mark.parametrize(
    "metadata", ["absent", "correct", "wrong", 0, -1, 64.5, True, None, "64"]
)
def test_weighted_verifier_effective_metadata(
    tmp_path: Path, oversized: bool, metadata: object
) -> None:
    # Synthetic accounting fixtures only. Preserve all four masks, both AO
    # dimensions/geometries and eight groups. Never replace measured receipts.
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    records = [
        record
        for name in provenance["data_sha256"]
        for record in json.loads(gzip.decompress((RECEIPTS / name).read_bytes()))
    ]
    for record in records:
        if oversized and record["tile"] == 256:
            record["tile"] = 4096
        points, requested = record["points"], record["tile"]
        effective = min(points, requested)
        if metadata != "absent":
            record["effective_tile"] = (
                effective
                if metadata == "correct"
                else effective + 1
                if metadata == "wrong"
                else metadata
            )
        n, mask = record["nao"], record["mask"]
        for slot, site in enumerate(record["sites"]):
            full = slot in (0, 1, 4)
            count = points // effective if full else int(points % effective != 0)
            extent = (
                effective if full or points % effective == 0 else points % effective
            )
            calls = 6 * count
            site["calls"] = calls
            site["summands"] = calls * extent * n * n
            site["scaled_elements"] = calls * extent * n if slot >= 4 else 0
            site["publication_passes"] = calls if slot >= 4 and mask & 4 else 0
            site["m"], site["n"], site["k"] = (
                (n, 1, n)
                if slot >= 4
                else (extent, n, n)
                if slot % 2 == 0
                else (n, n, extent)
            )
    data = gzip.compress(json.dumps(records).encode(), mtime=0)
    (tmp_path / "synthetic.json.gz").write_bytes(data)
    provenance["data_sha256"] = {"synthetic.json.gz": hashlib.sha256(data).hexdigest()}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    shutil.copyfile(RECEIPTS / "verify.py", tmp_path / "verify.py")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "verify.py"), "--write-summary"],
        capture_output=True,
        text=True,
        check=False,
    )
    if metadata in ("absent", "correct"):
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert len(summary) == 8
        for row in summary:
            effective = min(row["points"], row["tile"])
            assert (
                row["split_esp_publication_passes_per_evaluation"]
                == (row["points"] + effective - 1) // effective
            )
        assert {row["tile"] for row in summary} == (
            {64, 4096} if oversized else {64, 256}
        )
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr
