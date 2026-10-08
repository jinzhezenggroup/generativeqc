"""Host-only ragged owner differential against exact, isolated prechange bytes."""

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
FROZEN = ROOT / "tests/native/fixtures/ragged_topology_prechange_882f5c5e"
COMMIT = "882f5c5e0bc060b5777d5060c9cb89714bb4429b"
COMMON = Path("backends/common/gfn2_plan_schema.cpp")
NATIVE = ROOT / "src/xtb/native/src"
OWNER = ROOT / "src/runtime/ragged_topology.cpp"
PROBE = ROOT / "tests/native/test_ragged_topology_preservation.cpp"
PROJECTIONS = ("atom", "shell_ownership", "ao_matrix", "packed_all_pair", "ao_bucket")
MOVED = {
    "validate_gfn2_topology_binding": "validate_topology_binding",
    "validate_gfn2_topology_host": "validate_topology_host",
    "bind_gfn2_topology_host": "bind_topology_host",
    "gfn2_element_identity_fingerprint_host": "element_identity_fingerprint_host",
    **{
        f"{action}_gfn2_{projection}_projection_{suffix}": f"{action}_{projection}_projection_{suffix}"
        for projection in (*PROJECTIONS, "element_identity")
        for action, suffix in (("validate", "binding"), ("project", "host"))
    },
}


def test_frozen_ragged_topology_bytes() -> None:
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    assert manifest["commit"] == COMMIT
    assert set(manifest["sources"]) == {
        "src/xtb/native/src/backends/common/gfn2_plan_schema.cpp",
        "src/xtb/native/src/backends/common/gfn2_plan_schema.hpp",
    }
    for original, record in manifest["sources"].items():
        data = (FROZEN / record["fixture"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], original
        git_blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        assert git_blob == record["git_blob"], original
    assert (NATIVE / COMMON.with_suffix(".hpp")).read_bytes() == (
        FROZEN / COMMON.with_suffix(".hpp")
    ).read_bytes(), "native types and declarations are outside this extraction"


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\([^;{{]*?\) noexcept\s*\{{", source)
    assert match, name
    start = match.end() - 1
    depth = 1
    cursor = start + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def test_shared_owner_retires_only_topology_policy() -> None:
    current = (NATIVE / COMMON).read_text()
    frozen = (FROZEN / COMMON).read_text()
    owner = OWNER.read_text()
    header = OWNER.with_suffix(".hpp").read_text()
    assert len(MOVED) == 16
    for native, shared in MOVED.items():
        body = function_body(current, native)
        assert f"shared_topology::{shared}(" in body, native
        assert not re.search(r"\b(for|while|switch)\s*\(", body), native
        assert len(body.splitlines()) <= 10, native
        function_body(owner, shared)
    retained = re.findall(
        r"^(?:Gfn2PlanSchemaDiagnostic|std::uint64_t) (\w+)\(",
        frozen,
        flags=re.MULTILINE,
    )
    retained = [name for name in retained if "gfn2" in name and name not in MOVED]
    assert len(retained) >= 9
    for name in retained:
        assert function_body(current, name) == function_body(frozen, name), name
    # Shared ownership has no embedded-method, runtime-driver, or CUDA headers.
    includes = re.findall(r'^#include [<"]([^>\"]+)', owner + header, re.MULTILINE)
    assert not any(
        re.search(r"xtb|cuda|cublas|cusolver|hip", path, re.IGNORECASE)
        for path in includes
    )
    assert "xtb::" not in owner + header
    assert not re.search(
        r"\b(spin_channels|geometry_generation|cutoff_bohr)\b", owner + header
    )


@pytest.fixture(scope="module")
def topology_probes(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> dict[str, Path]:
    if sys.platform != "linux":
        pytest.skip("native allocation/link and PROT_NONE qualification requires Linux")
    test_frozen_ragged_topology_bytes()
    native: NativeCxx = request.getfixturevalue("required_native_cxx")
    compiler = replace(native, base_dir=ROOT)
    folder = tmp_path_factory.mktemp("ragged-topology-preservation")
    probes = {}
    for mode in ("frozen", "candidate"):
        if mode == "frozen":
            sources = [FROZEN / COMMON, PROBE]
            includes = [FROZEN]
            defines = ["-Dgenerativeqc=frozen_generativeqc"]
        else:
            sources = [NATIVE / COMMON, OWNER, PROBE]
            includes = [ROOT / "src", NATIVE]
            defines = []
        objects = []
        for index, source in enumerate(sources):
            output = folder / f"{mode}-{index}.o"
            compiler.compile_object(
                source,
                output,
                args=[
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *defines,
                    *["-I" + str(path) for path in includes],
                ],
            )
            objects.append(output)
        binary = folder / mode
        compiler.link(
            objects,
            binary,
            args=[
                "-Wl,--wrap=" + name
                for name in (
                    "malloc",
                    "calloc",
                    "realloc",
                    "aligned_alloc",
                    "posix_memalign",
                )
            ],
        )
        probes[mode] = binary
    return probes


@pytest.mark.parametrize("scenario", ("abi", "host", "device", "element"))
def test_frozen_ragged_topology_contract(
    topology_probes: dict[str, Path], scenario: str
) -> None:
    traces = {}
    for mode, binary in topology_probes.items():
        result = subprocess.run(
            [str(binary), scenario],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (mode, scenario, result.stdout, result.stderr)
        assert result.stdout.endswith("zero production allocations\n")
        traces[mode] = result.stdout
    assert traces["candidate"] == traces["frozen"], scenario


def test_owner_links_without_cuda(topology_probes: dict[str, Path]) -> None:
    nm = shutil.which("nm")
    readelf = shutil.which("readelf")
    assert nm and readelf, "host-only qualification requires nm and readelf"
    for mode, binary in topology_probes.items():
        symbols = subprocess.run(
            [nm, "-C", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        dynamic = subprocess.run(
            [readelf, "-d", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        assert not re.search(
            r"\b(?:cuda[A-Z]|cu[A-Z]|hip[A-Z]|cublas|cusolver|nccl)", symbols
        )
        assert not re.search(
            r"NEEDED.*(?:cuda|cublas|cusolver|hip|nccl)", dynamic, re.IGNORECASE
        )
        if mode == "candidate":
            assert "generativeqc::runtime::ragged::validate_topology_host(" in symbols
        else:
            assert (
                "frozen_generativeqc::xtb::detail::validate_gfn2_topology_host("
                in symbols
            )
            assert "generativeqc::runtime::ragged" not in symbols
