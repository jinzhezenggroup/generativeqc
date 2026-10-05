"""Host-only gates for real generated COSX endpoint receipt consumers.

Diagnostics are simulated counters, never numerical or timing evidence.
"""

import copy
import gzip
import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.dft.cosx_contraction import emit_cosx_contractions


@pytest.fixture(scope="module")
def current_receipts(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    tmp_path = tmp_path_factory.mktemp("cosx_endpoint_receipts")
    root = Path(__file__).resolve().parents[2]
    harness = (root / "tests/native/cosx_endpoint_benchmark.cuh").read_text()
    runtime = (root / "src/dft/cuda_cosx.cu").read_text()
    # Execute the real assertion AND complete receipt-output block, using real
    # compiler descriptors and the production clamp/full-tail slot expressions.
    checks = list(
        re.finditer(
            r"    for \(unsigned mask = 0; mask < (?:4|routes); \+\+mask\) \{\n"
            r"      const auto& info = plans\[mask\]->diagnostic\(\);",
            harness,
        )
    )
    clamps = re.findall(r"const std::size_t tile_points = [^;]+;", runtime)
    sites = re.findall(r"const std::size_t site = count == tile_points[^;]+;", runtime)
    esp_sites = re.findall(
        r"contractions->execute\(([^,]+), view.stream, esp.get\(\)", runtime
    )
    assert len(checks) == len(clamps) == len(sites) == len(esp_sites) == 1
    output = harness[checks[0].start() : harness.rindex("\n  }\n}")]
    serializer = (
        "void site_json"
        + harness.split("void site_json", 1)[1].split(
            "}  // namespace cosx_endpoint_test", 1
        )[0]
    )
    generated = (
        emit_cosx_contractions()
        .replace(
            '"tensor/cuda_contraction_sites.cuh"', '"tensor/contraction_sites.hpp"'
        )
        .split("using Prepared =", 1)[0]
        + "}\n"
    )
    (tmp_path / "cosx_descriptors.hpp").write_text(generated)
    owner = (root / "src/tensor/cuda_contraction_sites.cuh").read_text()
    reservations = re.findall(
        r"static constexpr std::size_t host_reservation = ([^;]+);", owner
    )
    site_counts = re.findall(
        r"using Prepared = tensor::PreparedContractionSites<(\d+)>;",
        emit_cosx_contractions(),
    )
    assert len(reservations) == len(site_counts) == 1
    source = tmp_path / "endpoint_work.cpp"
    source.write_text(
        r"""
#include "cosx_descriptors.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
using namespace generativeqc;
namespace cosx_lowering = generativeqc::dft::cosx_lowering;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
SERIALIZER
constexpr std::size_t Sites = SITE_COUNT;
struct Info {
  std::size_t tile_points{}, provider_allowance{}, device_bytes{}, retained_provider_bytes{};
  std::size_t contraction_host_bytes{HOST_RESERVATION};
  double contraction_prepare_seconds{};
  int provider_version{120901}, runtime_version{12090}, compute_major{12}, compute_minor{};
  std::array<tensor::ContractionSiteDiagnostic, 6> contractions;
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
  constexpr std::size_t repeats = 6, routes = 8, n = 13;
  // Oversized, singleton, exactly one tile, exact division, and nonempty tail.
  constexpr std::array<std::array<std::size_t, 2>, 5> shapes{{
      {3, 4}, {1, 4096}, {3, 3}, {6, 3}, {7, 3}}};
  for (const auto shape : shapes) {
    const auto points = shape[0], tile = shape[1];
    const auto tile_points = effective_tile(points, tile);
    std::array<Plan, routes> owners;
    std::array<Plan*, routes> plans;
    std::array<double, routes> setup{};
    std::array<std::array<double, repeats>, routes> samples{}, energies{};
    std::array<std::array<double, 3>, routes> oracle_errors{};
    std::array<std::array<std::array<double, 3>, repeats>, routes> paired_errors{};
    const struct { std::array<int, 1> atoms; } system{};
    const std::size_t radial = 1, polar = 1, azimuth = points, geometry = 0;
    const double geometry_seconds = 0;
    for (unsigned mask = 0; mask < routes; ++mask) {
      plans[mask] = &owners[mask];
      samples[mask].fill(1.0); // Synthetic accounting, not measured timings.
      auto& info = owners[mask].info;
      info.tile_points = tile_points;
      info.provider_allowance = mask ? 96ULL << 20 : 0;
      info.device_bytes = 1024 + info.provider_allowance;
      for (unsigned slot = 0; slot < 6; ++slot) {
        const auto p = slot < 2 || slot == 4 || !(points % tile_points)
                           ? tile_points : points % tile_points;
        const auto descriptor = slot >= 4 ? cosx_lowering::esp_application(p, n)
                              : slot % 2 ? cosx_lowering::accumulation(p, n)
                                         : cosx_lowering::projection(p, n);
        auto& value = info.contractions[slot];
        value.resolved = descriptor.resolved;
        value.batch_scale = descriptor.batch_scale;
        value.offer_count = descriptor.candidates.size();
        std::copy(descriptor.candidates.begin(), descriptor.candidates.end(), value.offers.begin());
        const bool library = mask & (1U << (slot >= 4 ? 2 : slot % 2));
        value.selected = library ? 0 : 1;
        if (!library) value.offers[0].rejection = "unqualified";
        for (std::size_t i = 2; i < value.offer_count; ++i)
          value.offers[i].rejection = "unqualified optional";
        value.candidate = value.offers[value.selected];
      }
      // Count executed tiles independently of the harness's division/remainder
      // work formulas. Populate each real descriptor's scalar/batch work.
      for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        for (std::size_t begin = 0; begin < points; begin += tile_points) {
          const auto count = std::min(tile_points, points - begin);
          SITE
          const std::size_t esp_site = ESP_SITE;
          for (const auto slot : {site, site + 1, esp_site}) {
            auto& value = info.contractions[slot];
            ++value.calls;
            value.summands += value.resolved.summands();
            if (value.batch_scale.rank) {
              value.scaled_elements += value.resolved.output_elements();
              if (value.candidate.provider == "cublas") ++value.publication_passes;
            }
          }
        }
    }
    OUTPUT
  }
}
""".replace("CLAMP", clamps[0])
        .replace("HOST_RESERVATION", reservations[0])
        .replace("SITE_COUNT", site_counts[0])
        .replace("ESP_SITE", esp_sites[0])
        .replace("SITE", sites[0])
        .replace("SERIALIZER", serializer)
        .replace("OUTPUT", output)
    )
    binary = tmp_path / "endpoint_work"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert (
        len(records) == len(shapes := ((3, 4), (1, 4096), (3, 3), (6, 3), (7, 3))) * 8
    )
    assert {(r["points"], r["tile"]) for r in records} == set(shapes)
    assert {r["schema"] for r in records} == {"cosx-endpoint-v2"}
    return records


def test_endpoint_work_uses_effective_tile(current_receipts: list[dict]) -> None:
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    summary = verifier["verify_records"](current_receipts, provenance)
    assert len(summary) == 5
    for item in summary:
        n, points = item["nao"], item["points"]
        assert item["esp_application_summands_per_evaluation"] == points * n * n
        assert item["esp_scaled_elements_per_evaluation"] == points * n
        passes = item["esp_publication_passes_per_evaluation_masks_0_1_2_3_4_5_6_7"]
        tiles = (points + min(points, item["tile"]) - 1) // min(points, item["tile"])
        assert passes == [0] * 4 + [tiles] * 4


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_esp",
        "missing_route",
        "schema",
        "label",
        "algorithm",
        "host_reservation",
        "batches",
        "shape",
        "calls",
        "summands",
        "scaled_elements",
        "publication_passes",
        "ordinary_scale",
        "ordinary_publication",
        "tail_shape",
        "tail_calls",
        "effective_tile",
        "scientific",
        "semantic",
        "precision",
        "catalog_identity",
        "catalog_provider",
        "catalog_short",
        "catalog_extra",
        "selected",
        "candidate",
        "rejection",
        "mask",
    ],
)
def test_current_receipts_reject_partial_or_relabelled_work(
    current_receipts: list[dict], mutation: str
) -> None:
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    records = copy.deepcopy(
        [r for r in current_receipts if r["points"] == 3 and r["tile"] == 4]
    )
    record = records[
        4
    ]  # ESP-only library route; tail descriptors are present but idle.
    site = record["sites"][4]
    if mutation == "missing_esp":
        record["sites"] = record["sites"][:4]
    elif mutation == "missing_route":
        records.pop()
    elif mutation == "schema":
        record["schema"] = "cosx-endpoint-v1"
    elif mutation == "host_reservation":
        record["host_reservation"] = 128 << 10
    elif mutation == "algorithm":
        site["algorithm"] = "batch-scaled-fused"
    elif mutation == "label":
        site["label"] = "projection_full"
    elif mutation == "shape":
        site["n"] += 1
    elif mutation in (
        "batches",
        "calls",
        "summands",
        "scaled_elements",
        "publication_passes",
    ):
        site[mutation] += 1
    elif mutation == "ordinary_scale":
        record["sites"][0]["scaled_elements"] = 1
    elif mutation == "ordinary_publication":
        record["sites"][0]["publication_passes"] = 1
    elif mutation == "tail_shape":
        record["sites"][5]["batches"] = 0
    elif mutation == "tail_calls":
        record["sites"][5]["calls"] = 1
    elif mutation == "effective_tile":
        del record["effective_tile"]
    elif mutation in ("scientific", "semantic", "precision"):
        site[mutation] = "0" * 64
    elif mutation == "catalog_identity":
        site["offers"][-1]["identity"] = "0" * 64
    elif mutation == "catalog_provider":
        site["offers"][-1]["provider"] = "unknown"
    elif mutation == "catalog_short":
        site["offers"].pop()
    elif mutation == "catalog_extra":
        site["offers"].append(site["offers"][-1].copy())
    elif mutation == "selected":
        site["selected"] = -1
    elif mutation == "candidate":
        site["candidate"] = "0" * 64
    elif mutation == "rejection":
        site["offers"][site["selected"]]["rejection"] = "rejected"
    elif mutation == "mask":
        record["mask"] = 0
    with pytest.raises((AssertionError, KeyError)):
        verifier["verify_records"](records, provenance)


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


@pytest.mark.parametrize("mutation", [None, "identity", "provider", "rejection"])
def test_current_receipts_preserve_exact_pending_cutlass_offer(
    current_receipts: list[dict], mutation: str | None
) -> None:
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    records = copy.deepcopy(current_receipts)
    for record in records:
        for slot, site in enumerate(record["sites"]):
            family = 2 if slot >= 4 else slot % 2
            site["offers"].append(
                {
                    "identity": verifier["CURRENT_OFFER_IDENTITIES"][family][4],
                    "provider": "cutlass-aot",
                    "rejection": "unqualified optional",
                }
            )
    offer = records[0]["sites"][4]["offers"][-1]
    if mutation is None:
        assert len(verifier["verify_records"](records, provenance)) == 5
    else:
        offer[mutation] = {
            "identity": "0" * 64,
            "provider": "unknown",
            "rejection": "",
        }[mutation]
        with pytest.raises(AssertionError):
            verifier["verify_records"](records, provenance)


def test_historical_and_current_groups_cannot_fill_each_others_routes(
    current_receipts: list[dict],
) -> None:
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    current = [r for r in current_receipts if r["points"] == 3 and r["tile"] == 4]
    historical = copy.deepcopy(current[:4])
    for record in historical:
        record["schema"] = "cosx-endpoint-v1"
        record["host_reservation"] = 128 << 10
        record["sites"] = record["sites"][:4]
    assert len(verifier["verify_records"](historical + current, provenance)) == 2
    with pytest.raises(AssertionError):
        verifier["verify_records"](historical + current[1:], provenance)
