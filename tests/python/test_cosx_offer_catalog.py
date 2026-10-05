"""Keep COSX native gates and historical receipts compatible with real offers."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.common import native_lowering
from generativeqc_compiler.common.lowering_provider import ProviderDescriptor
from generativeqc_compiler.dft.cosx_contraction import emit_cosx_contractions

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from generativeqc_compiler.common.lowering_provider import (
        LoweringCandidate,
        LoweringRequest,
    )
    from generativeqc_compiler.common.specialization import (
        CompilationIdentity,
        TargetCapabilities,
    )

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "benchmarks/results/cosx-contractions-1884"


def generated_catalogs() -> list[list[dict[str, str]]]:
    """Read identities from actual generated declarations, not copied fixtures."""
    source = emit_cosx_contractions()
    result = []
    for name in ("projection", "accumulation"):
        block = source.split(f"{name}_candidates{{{{", 1)[1].split("\n}};", 1)[0]
        fields = [
            re.findall(r'"([^"]*)"', line)
            for line in block.splitlines()
            if line.startswith('{"')
        ]
        result.append([{"identity": f[0], "provider": f[4]} for f in fields])
    return result


def test_pinned_receipt_catalog_matches_actual_generator() -> None:
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    for family, offers in enumerate(generated_catalogs()):
        assert len(offers) in (3, 4, 5)
        assert (
            tuple(o["provider"] for o in offers) == verifier["PROVIDERS"][: len(offers)]
        )
        assert (
            tuple(o["identity"] for o in offers)
            == verifier["OFFER_IDENTITIES"][family][: len(offers)]
        )


def test_pending_cutlass_identity_uses_canonical_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    emit = native_lowering.native_lowering_portfolio

    def capture(
        request: LoweringRequest,
        candidates: Sequence[LoweringCandidate],
        target: TargetCapabilities,
        compilation: CompilationIdentity,
        *,
        name: str,
    ) -> str:
        captured.append(candidates[0])
        return emit(request, candidates, target, compilation, name=name)

    monkeypatch.setattr(native_lowering, "native_lowering_portfolio", capture)
    emit_cosx_contractions()
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    assert len(captured) == 2
    for family, template in enumerate(captured):
        candidate = replace(
            template,
            implementation="region-cutlass-aot",
            providers=(
                ProviderDescriptor(
                    "cutlass-aot",
                    "generated",
                    "prepared-affine-region",
                    version="runtime-bound-qualified-artifact",
                ),
            ),
        )
        assert candidate.identity == verifier["OFFER_IDENTITIES"][family][4]


def test_native_gate_accepts_actual_generated_catalog(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    generated = emit_cosx_contractions().replace(
        '"tensor/cuda_contraction_sites.cuh"', '"tensor/contraction_sites.hpp"'
    )
    # Preserve real descriptors and candidate types; exclude only CUDA owner
    # preparation. This checks the native assertion block, not GPU execution.
    generated = generated.split("using Prepared =", 1)[0] + "}\n"
    (tmp_path / "cosx_descriptors.hpp").write_text(generated)
    native = (ROOT / "tests/native/cosx_contraction_cases.cuh").read_text()
    checks = re.findall(
        r"(const bool library = route < 4.*?)(?=\s+const auto calls =)",
        native,
        re.DOTALL,
    )
    assert len(checks) == 1
    source = tmp_path / "catalog.cpp"
    source.write_text(
        r"""
#include "cosx_descriptors.hpp"
#include <algorithm>
#include <iostream>
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
namespace cosx_lowering = generativeqc::dft::cosx_lowering;
int main() {
  const struct { int contraction_device=0, compute_major=12, runtime_version=12090; } info;
  for (unsigned route=0; route<6; ++route) {
    const unsigned mask=route<4?route:3;
    for (unsigned slot=0; slot<4; ++slot) {
      const auto descriptor=slot%2?cosx_lowering::accumulation(3,7):cosx_lowering::projection(3,7);
      generativeqc::tensor::ContractionSiteDiagnostic good{descriptor.resolved, {}};
      good.offer_count=descriptor.candidates.size();
      std::copy(descriptor.candidates.begin(),descriptor.candidates.end(),good.offers.begin());
      good.selected=route<4&&(mask&(1U<<(slot%2)))?0:1;
      if(good.selected==1) good.offers[0].rejection="unqualified or unavailable";
      good.candidate=good.offers[good.selected];
      for (std::size_t i=2;i<good.offer_count;++i) good.offers[i].rejection="unqualified optional";
      auto accepts=[&](const auto& site) {
        try { CHECKS return true; } catch(const std::exception&) { return false; }
      };
      if (!accepts(good)) { std::cerr<<"real generated catalog rejected"; return 1; }
      auto bad=good; bad.selected=bad.offer_count;
      if(accepts(bad)) return 2;
      bad=good; bad.offers[good.offer_count-1].provider="unknown";
      if(accepts(bad)) return 3;
      bad=good; bad.offers[good.offer_count-1].identity=bad.offers[0].identity;
      if(accepts(bad)) return 4;
      bad=good; bad.offers[good.offer_count-1].rejection={};
      if(accepts(bad)) return 5;
      bad=good; bad.offers[good.selected].rejection="rejected selected offer";
      if(accepts(bad)) return 6;
    }
  }
}
""".replace("CHECKS", checks[0])
    )
    binary = tmp_path / "catalog"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("count", [3, 4, 5])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "bounds",
        "negative",
        "provider",
        "duplicate",
        "identity",
        "selected_identity",
        "selected_provider",
        "selected_rejection",
        "optional_rejection",
        "cublas_rejection",
    ],
)
def test_receipt_catalog_integrity(
    tmp_path: Path, count: int, mutation: str | None
) -> None:
    records = json.loads(gzip.decompress((RECEIPTS / "endpoints.json.gz").read_bytes()))
    catalogs = generated_catalogs()
    verifier = runpy.run_path(str(RECEIPTS / "verify.py"))
    for record in records:
        for slot, site in enumerate(record["sites"]):
            if count >= 4:
                site["offers"].append(
                    {**catalogs[slot % 2][3], "rejection": "unqualified optional"}
                )
            if count == 5:
                site["offers"].append(
                    {
                        "identity": verifier["OFFER_IDENTITIES"][slot % 2][4],
                        "provider": "cutlass-aot",
                        "rejection": "unqualified optional",
                    }
                )
    site = records[0]["sites"][0]
    offers = site["offers"]
    if mutation == "bounds":
        site["selected"] = count
    elif mutation == "negative":
        site["selected"] = -1
    elif mutation == "provider":
        offers[-1]["provider"] = "unknown"
    elif mutation == "duplicate":
        offers[-1]["provider"] = offers[0]["provider"]
    elif mutation == "identity":
        offers[-1]["identity"] = offers[0]["identity"]
    elif mutation == "selected_identity":
        site["candidate"] = "0" * 64
    elif mutation == "selected_provider":
        site["provider"] = "cublas"
    elif mutation == "selected_rejection":
        offers[site["selected"]]["rejection"] = "rejected"
    elif mutation == "optional_rejection":
        offers[-1]["rejection"] = ""
    elif mutation == "cublas_rejection":
        offers[0]["rejection"] = ""
    data = gzip.compress(json.dumps(records).encode(), mtime=0)
    (tmp_path / "endpoints.json.gz").write_bytes(data)
    provenance = json.loads((RECEIPTS / "provenance.json").read_text())
    provenance["data_sha256"] = {"endpoints.json.gz": hashlib.sha256(data).hexdigest()}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    shutil.copyfile(RECEIPTS / "verify.py", tmp_path / "verify.py")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "verify.py"), "--write-summary"],
        capture_output=True,
        text=True,
        check=False,
    )
    if mutation is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr
