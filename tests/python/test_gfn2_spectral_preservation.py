"""Frozen full-API CPU spectral owner differential and independent physics gates.

The reference build has its own SHA-256 pinned header/source/generated closure.
It never includes live ownership code or regenerates its numerical dependency.
The mock cohort is admitted by the actual LP64 provider in each build.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "tests/native/fixtures/spectral_prechange_5c02b1ba"
COMMIT = "5c02b1ba3c69fbee712697565dcb79da6b768606"
METHOD = ROOT / "src/methods/gfn2_electronic_update.cpp"
OWNER = ROOT / "src/solver/cpu/prepared_spectral.cpp"
SCENARIOS = (
    "plan",
    "binding",
    "generations",
    "factor-backend",
    "factor-numerical",
    "solve-backend",
    "solve-numerical",
    "workers",
    "aliases",
    "oracle",
)


def test_frozen_spectral_oracle_bytes() -> None:
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    assert manifest["commit"] == COMMIT
    for original, record in manifest["sources"].items():
        frozen = FROZEN / record["fixture"]
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == record["sha256"], (
            original
        )
    for relative, record in manifest["generated"].items():
        assert record["commit"] == COMMIT
        assert (
            hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest()
            == record["sha256"]
        )
    # Public compatibility types and signatures were not part of the move.
    assert (ROOT / "src/xtb/native/src/model/gfn2/eigensolver.hpp").read_bytes() == (
        FROZEN / "model/gfn2/eigensolver.hpp"
    ).read_bytes()


@pytest.fixture(scope="module")
def spectral_probes(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> dict[str, tuple[Path, str, str]]:
    test_frozen_spectral_oracle_bytes()
    folder = tmp_path_factory.mktemp("spectral-preservation")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(folder / "generated_weighted_gram_native.hpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    compiler = replace(required_native_cxx, base_dir=ROOT)
    nm = shutil.which("nm")
    assert nm, "occupation call-count instrumentation requires nm"
    probes = {}
    for mode in ("frozen", "candidate"):
        if mode == "frozen":
            includes = [FROZEN]
            method = FROZEN / "model/gfn2/eigensolver.cpp"
            sources = [
                method,
                FROZEN / "tensor/cpu/lp64_provider.cpp",
                FROZEN / "tensor/cpu_linalg.cpp",
                FROZEN / "tests/native/fixtures/cpu_lp64/mock_provider.cpp",
            ]
        else:
            includes = [ROOT / "src", ROOT / "src/xtb/native/src", folder]
            method = METHOD
            sources = [
                METHOD,
                OWNER,
                ROOT / "src/tensor/cpu/lp64_provider.cpp",
                ROOT / "src/tensor/cpu_linalg.cpp",
                ROOT / "tests/native/fixtures/cpu_lp64/mock_provider.cpp",
            ]
        sources.append(ROOT / "tests/native/test_gfn2_spectral_preservation.cpp")
        args = [
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            *["-I" + str(path) for path in includes],
        ]
        objects = []
        for index, source in enumerate(sources):
            output = folder / f"{mode}-{index}.o"
            # Instrument the unedited complete method translation unit only.
            # The hook itself lives in the uninstrumented harness, cannot recurse,
            # and uses neither allocation nor production test-only hooks.
            instrumentation = (
                ["-finstrument-functions"] if source in (method, OWNER) else []
            )
            compiler.compile_object(source, output, args=[*args, *instrumentation])
            objects.append(output)
        binary = folder / mode
        compiler.link(objects, binary, args=["-ldl", "-pthread"])
        # Only defined code addresses are consumed below; debug/undefined
        # symbols and address sorting are unnecessary. Symbol inspection is
        # setup, not a performance gate: allow the same bounded budget as linking
        # under parallel CI, while retaining all exact body-count assertions.
        symbols = subprocess.run(
            [nm, "--defined-only", "--demangle", "--no-sort", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        addresses = re.findall(
            r"^([0-9a-f]+) [tT] generativeqc::xtb::detail::gfn2::"
            r"\(anonymous namespace\)::compute_occupations\(double const\*,[^\n]*\)$",
            symbols,
            flags=re.MULTILINE,
        )
        assert len(addresses) == 1, "expected one complete production occupation body"
        main = re.findall(r"^([0-9a-f]+) [tT] main$", symbols, flags=re.MULTILINE)
        assert len(main) == 1
        # Resolve relative to main so the hook works with PIE/ASLR enabled.
        scans = re.findall(
            r"^([0-9a-f]+) [tT] generativeqc::.*::"
            r"symmetric_finite_row_major\(double const\*,[^\n]*\)$",
            symbols,
            flags=re.MULTILINE,
        )
        assert scans, "missing complete matrix finite/symmetry preflight body"
        origin = int(main[0], 16)
        probes[mode] = (
            binary,
            f"{int(addresses[0], 16) - origin:x}",
            ",".join(f"{int(address, 16) - origin:x}" for address in scans),
        )
    return probes


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_frozen_cpu_spectral_publication(
    spectral_probes: dict[str, tuple[Path, str, str]], scenario: str
) -> None:
    traces = {}
    for mode, (binary, occupation_address, scan_addresses) in spectral_probes.items():
        result = subprocess.run(
            [str(binary), scenario, occupation_address, scan_addresses],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (mode, scenario, result.stdout, result.stderr)
        traces[mode] = result.stdout
    assert traces["candidate"] == traces["frozen"], scenario


def test_only_preparation_adds_one_immutable_owner_allocation(
    spectral_probes: dict[str, tuple[Path, str, str]],
) -> None:
    records = {}
    for mode, (binary, occupation_address, scan_addresses) in spectral_probes.items():
        result = subprocess.run(
            [str(binary), "setup", occupation_address, scan_addresses],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        match = re.fullmatch(r"setup (\d+) bytes (\d+) resident (\d+)\n", result.stdout)
        assert match, result.stdout
        records[mode] = tuple(map(int, match.groups()))
    frozen, candidate = records["frozen"], records["candidate"]
    # The reviewed migration permits exactly one extra immutable resource owner
    # at setup. All bind/factor/worker/batch calls independently require zero.
    assert candidate[0] == frozen[0] + 1
    assert candidate[1] > frozen[1]
    assert candidate[2] > frozen[2]
    assert candidate[1] - frozen[1] >= candidate[2] - frozen[2]
