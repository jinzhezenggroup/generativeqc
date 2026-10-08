"""CPU-host preservation of the unchanged CUDA topology adapter.

The production host definitions are compiled verbatim except for CUDA launch
syntax, which becomes an argument trace. Kernels are not compiled or executed.
The same adapter is linked against pinned pre-extraction common-schema sources
and the real current wrappers plus runtime owner. PROT_NONE metadata catches
accidental host dereferences. CUDA responses, including device diagnostics at
the copy boundary, are injected: this is no GPU, numerical, or runtime proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"
CUDA = NATIVE / "backends/cuda/gfn2_plan_schema.cu"
BASELINE = ROOT / "tests/native/fixtures/ragged_topology_prechange_882f5c5e"
BASE_COMMIT = "882f5c5e0bc060b5777d5060c9cb89714bb4429b"
# Independently obtained with git show BASE_COMMIT:path | sha256sum. These are
# full raw bytes, not normalized tokens, and are independent of the fixture's
# manifest and the source-extraction code below.
PINNED_SHA256 = {
    "backends/cuda/gfn2_plan_schema.cu": "947192b1f3e7076b84299bdfb9126b46d35092a380722c8538a6414dcc81e297",
    "backends/cuda/gfn2_plan_schema.cuh": "84c079a72fc0fc9550c1337549dbba5eb9b894c93760b7f444103be59f3a65c0",
    "backends/common/gfn2_plan_schema.hpp": "6261e0b03295e509064da4dfc96d52246b6f570e27cb5d9b3c8b5d7b747d6c5c",
}
BASELINE_CPP_SHA256 = "ed55d903522adcaa88d0eac702aa56df8f06482ae55893dfe44feebe9314e1d5"


def test_raw_cuda_adapter_and_native_header_are_unchanged() -> None:
    for path, digest in PINNED_SHA256.items():
        assert hashlib.sha256((NATIVE / path).read_bytes()).hexdigest() == digest, path
    assert (
        hashlib.sha256(
            (BASELINE / "backends/common/gfn2_plan_schema.cpp").read_bytes()
        ).hexdigest()
        == BASELINE_CPP_SHA256
    )
    assert (
        hashlib.sha256(
            (BASELINE / "backends/common/gfn2_plan_schema.hpp").read_bytes()
        ).hexdigest()
        == PINNED_SHA256["backends/common/gfn2_plan_schema.hpp"]
    )


def _host_source() -> str:
    source = CUDA.read_text()
    # Retain every host body, including its original namespace and includes.
    # Only the contiguous device definitions are omitted. No validator is
    # rewritten, copied into a model, or replaced with a test implementation.
    before = source[: source.index("__device__ bool square(")]
    after = source[source.index("bool pointer_is_cuda_accessible(") :]
    host = before.replace("__host__ __device__ ", "") + after
    host, count = re.subn(
        r"\b(\w+)<<<(.*?)>>>\s*\(",
        lambda match: f'trace_kernel("{match[1]}", {match[2]})(',
        host,
        flags=re.DOTALL,
    )
    assert count == 2
    assert "<<<" not in host and ">>>" not in host
    assert "__device__" not in host and "__global__" not in host
    assert host.count("cudaError_t bind_gfn2_") == 8
    return host


def _serialization() -> str:
    """Serialize every declared field, without inspecting padding bytes."""
    header = (NATIVE / "backends/common/gfn2_plan_schema.hpp").read_text()
    names = (
        "Gfn2PlanSchemaDiagnostic",
        "Gfn2RaggedTopologyView",
        "Gfn2GeometryCacheProvenanceView",
        "Gfn2AtomProjectionView",
        "Gfn2ShellOwnershipProjectionView",
        "Gfn2AOMatrixProjectionView",
        "Gfn2PackedAllPairProjectionView",
        "Gfn2AOBucketProjectionView",
        "Gfn2ElementIdentityProjectionView",
    )
    definitions = []
    for name in names:
        body = re.search(rf"struct {name} \{{(.*?)\n\}};", header, re.DOTALL)[1]
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", body, flags=re.DOTALL)
        fields = re.findall(r"\b(\w+)\s*=.*?;", body)
        assert fields and len(fields) == body.count(";")
        definitions.append(
            f"std::string describe(const {name}& value) {{\n"
            f'  std::string text = "{name}{{";\n'
            + "".join(
                f'  text += "{field}=" + describe(value.{field}) + ";";\n'
                for field in fields
            )
            + '  return text + "}";\n}\n'
        )
        definitions.append(
            f"void poison({name}& value) {{\n"
            + "".join(f"  poison_field(value.{field});\n" for field in fields)
            + "}\n"
        )
    return "\n".join(definitions)


CUDA_STUB = """#pragma once
#include <cstddef>
struct cudaStream;
using cudaStream_t = cudaStream*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
enum cudaMemoryType { cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2,
                      cudaMemoryTypeManaged = 3 };
struct cudaPointerAttributes { cudaMemoryType type{}; };
enum cudaMemcpyKind { cudaMemcpyDeviceToHost = 2 };
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes*, const void*);
cudaError_t cudaGetLastError();
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
"""


@pytest.fixture(scope="module")
def bridge_traces(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> dict[str, dict[str, dict]]:
    test_raw_cuda_adapter_and_native_header_are_unchanged()
    if os.name != "posix":
        pytest.skip("CPU-host CUDA trace probe requires POSIX mmap/PROT_NONE")
    # Check platform before requesting a compiler: static byte-preservation
    # checks remain universal even on Windows without a native toolchain.
    required_native_cxx: NativeCxx = request.getfixturevalue("required_native_cxx")
    folder = tmp_path_factory.mktemp("ragged-topology-cuda-host")
    (folder / "cuda_runtime_api.h").write_text(CUDA_STUB)
    (folder / "ragged_topology_cuda_host.inc").write_text(_host_source())
    (folder / "ragged_topology_cuda_serialization.inc").write_text(_serialization())
    compiler = replace(required_native_cxx, base_dir=ROOT)
    traces = {}
    for version in ("baseline", "current"):
        owner = BASELINE if version == "baseline" else NATIVE
        sources = [
            ROOT / "tests/native/test_ragged_topology_cuda_bridge.cpp",
            owner / "backends/common/gfn2_plan_schema.cpp",
        ]
        if version == "current":
            sources.append(ROOT / "src/runtime/ragged_topology.cpp")
        executable = compiler.build_executable(
            sources,
            folder / version,
            compile_args=[
                "-std=c++17",
                "-O0",
                "-g",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I" + str(folder),
                "-I" + str(owner),
                "-I" + str(NATIVE),
                "-I" + str(ROOT / "src"),
            ],
        )
        completed = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True, timeout=30
        )
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        traces[version] = {row["case"]: row for row in rows}
        assert len(traces[version]) == len(rows), "duplicate case names"
        assert len(rows) == 737, (
            "update the explicit trace coverage count when adding cases"
        )
    assert traces["baseline"].keys() == traces["current"].keys()
    return traces


@pytest.mark.parametrize(
    "operation",
    (
        "topology-bind",
        "topology-async",
        "provenance-bind",
        "provenance-async",
        "atom",
        "shell",
        "ao-matrix",
        "packed-pair",
        "ao-bucket",
        "element",
    ),
)
def test_actual_cuda_host_bridge_matches_frozen_owner(
    bridge_traces: dict,
    operation: str,
) -> None:
    baseline, current = bridge_traces["baseline"], bridge_traces["current"]
    cases = [name for name in baseline if name.startswith(operation + "/")]
    assert cases
    for name in cases:
        assert current[name] == baseline[name], name


def _calls(row: dict) -> list[str]:
    return [event.split("|", 1)[0] for event in row["events"]]


def test_topology_order_arguments_and_transactional_publication(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    valid = rows["topology-bind/valid"]
    assert _calls(valid) == ["attributes"] * 15 + [
        "launch",
        "last-error",
        "copy",
        "sync",
    ]
    assert valid["status"] == 0 and valid["published"] and not valid["cleared"]
    assert valid["binding"] == valid["input"]
    pointers = ["meta+61440"] + [f"meta+{4096 * index}" for index in range(1, 15)]
    assert valid["events"][:15] == [f"attributes|{pointer}|0|2" for pointer in pointers]
    assert valid["events"][15] == (
        "launch|topology_validation_kernel|1|1|0|ptr+4660|"
        + valid["input"]
        + "|meta+61440"
    )
    assert valid["events"][16:] == [
        "last-error|0",
        "copy|host-diagnostic|meta+61440|16|2|ptr+4660|0",
        "sync|ptr+4660|0",
    ]
    for stage, status, tail in (
        ("launch-error", 701, ["launch", "last-error"]),
        ("copy-error", 702, ["launch", "last-error", "copy"]),
        ("sync-error", 703, ["launch", "last-error", "copy", "sync"]),
    ):
        row = rows["topology-bind/" + stage]
        assert row["status"] == status and row["cleared"] and not row["published"]
        assert _calls(row) == ["attributes"] * 15 + tail
    semantic = rows["topology-bind/semantic-error"]
    assert semantic["status"] == 0 and semantic["cleared"]
    assert (
        semantic["diagnostic"] == "Gfn2PlanSchemaDiagnostic{error=9;field=2;index=1;}"
    )
    assert _calls(semantic) == _calls(valid)


def test_async_checks_use_no_cuda_pointer_queries_or_host_reads(
    bridge_traces: dict,
) -> None:
    for operation in ("topology-async", "provenance-async"):
        rows = bridge_traces["current"]
        valid = rows[operation + "/valid"]
        assert valid["status"] == 0
        assert _calls(valid) == ["launch", "last-error"]
        assert (
            valid["diagnostic"]
            == "Gfn2PlanSchemaDiagnostic{error=26;field=35;index=777;}"
        )
        assert all(
            "attributes" not in _calls(row) and "copy" not in _calls(row)
            for name, row in rows.items()
            if name.startswith(operation + "/")
        )
        for mutation in (
            "wrong-space",
            "zero-token",
            "bad-shape",
            "diagnostic-null",
            "diagnostic-misaligned",
            "diagnostic-overflow",
            "diagnostic-alias",
        ):
            row = rows[operation + "/" + mutation]
            assert row["status"] == 1 and row["events"] == []
            assert row["diagnostic"] == valid["diagnostic"]


def test_each_pointer_attribute_failure_stops_in_order(bridge_traces: dict) -> None:
    rows = bridge_traces["current"]
    for index in range(15):
        for kind in ("attribute-error", "host-pointer"):
            row = rows[f"topology-bind/{kind}-{index}"]
            assert row["cleared"] and not row["published"]
            expected = ["attributes"] * (index + 1)
            if kind == "attribute-error":
                expected.append("last-error")
                assert row["events"][-1] == "last-error|700"
            assert _calls(row) == expected
            assert row["status"] == (1 if index == 0 else 0)
            if index:
                assert row["diagnostic"] == (
                    f"Gfn2PlanSchemaDiagnostic{{error=1;field={index + 1};index=-1;}}"
                )


def test_all_projections_clear_on_failure_without_cuda_work(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    for operation in ("atom", "shell", "ao-matrix", "packed-pair", "ao-bucket"):
        valid = rows[operation + "/valid"]
        assert valid["status"] == 0 and valid["published"] and not valid["cleared"]
        for name, row in rows.items():
            if name.startswith(operation + "/"):
                assert row["events"] == [], name
                assert row["status"] == 0, name
                if not row["published"]:
                    assert row["cleared"], name
        for mutation in ("wrong-space", "zero-token", "bad-shape", "alias", "overflow"):
            assert rows[operation + "/" + mutation]["cleared"]


def test_provenance_trace_covers_active_mask_and_failure_boundaries(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    valid = rows["provenance-bind/valid"]
    assert valid["binding"] == valid["input"] and valid["published"]
    assert _calls(valid) == ["attributes"] * 3 + [
        "launch",
        "last-error",
        "copy",
        "sync",
    ]
    assert valid["events"][:3] == [
        "attributes|meta+61440|0|2",
        "attributes|meta+65536|0|2",
        "attributes|meta+69632|0|2",
    ]
    assert valid["events"][3] == (
        "launch|provenance_validation_kernel|1|1|0|ptr+4660|"
        + valid["input"]
        + "|37|meta+69632|meta+61440"
    )
    for mutation in (
        "active-count",
        "active-null-count",
        "active-alias-topology",
        "active-alias-generation",
        "generation-alias",
        "generation-count",
        "diagnostic-alias-generation",
        "diagnostic-alias-active",
    ):
        for operation in ("provenance-bind", "provenance-async"):
            row = rows[operation + "/" + mutation]
            assert "launch" not in _calls(row), row
            if operation.endswith("bind"):
                assert row["cleared"]
    for stage, status in (
        ("launch-error", 701),
        ("copy-error", 702),
        ("sync-error", 703),
    ):
        row = rows["provenance-bind/" + stage]
        assert row["status"] == status and row["cleared"]
    assert rows["provenance-bind/semantic-error"]["cleared"]


def test_element_identity_queries_only_nonempty_device_array(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    valid = rows["element/valid"]
    assert valid["published"] and _calls(valid) == ["attributes"]
    for mutation in ("zero-token", "zero-fingerprint", "bad-count", "negative-count"):
        row = rows["element/" + mutation]
        assert row["cleared"] and row["events"] == []
    for mutation in ("attribute-error", "host-pointer"):
        row = rows["element/" + mutation]
        assert row["cleared"] and not row["published"]
    empty = rows["element/empty"]
    assert empty["published"] and empty["events"] == []


def _fields(description: str) -> dict[str, str]:
    return dict(
        field.split("=", 1)
        for field in description.split("{", 1)[1][:-1].split(";")
        if field
    )


def test_projection_fields_preserve_exact_master_identity(bridge_traces: dict) -> None:
    for operation in ("atom", "shell", "ao-matrix", "packed-pair", "ao-bucket"):
        row = bridge_traces["current"][operation + "/valid"]
        master, projection = _fields(row["input"]), _fields(row["binding"])
        assert projection
        for field, value in projection.items():
            assert value == master[field], (operation, field)
    row = bridge_traces["current"]["element/valid"]
    original, binding = _fields(row["input"]), _fields(row["binding"])
    for field in (
        "plan_token",
        "total_atoms",
        "atomic_number_count",
        "element_fingerprint",
    ):
        assert original[field] == binding[field]
    assert binding["memory_space"] == "2"
    assert binding["atomic_numbers"] == "meta+77824"


def test_invalid_structure_rejects_before_any_cuda_call(bridge_traces: dict) -> None:
    rows = bridge_traces["current"]
    mutations = [
        "wrong-space",
        "unknown-space",
        "zero-token",
        "bad-shape",
        "negative-batch",
        "zero-batch",
        "negative-atoms",
        "negative-shells",
        "negative-orbitals",
        "negative-matrices",
        "negative-pairs",
        "negative-buckets",
        "overflow",
        "atom-overflow",
        "shell-overflow",
        "bucket-overflow",
        "alias",
        "address-overflow",
        "byte-count-overflow",
        "unknown-pairs",
    ] + [
        f"{prefix}-{index}"
        for prefix in ("count", "null", "misaligned", "alias")
        for index in range(14)
    ]
    for mutation in mutations:
        bound = rows["topology-bind/" + mutation]
        asynchronous = rows["topology-async/" + mutation]
        assert bound["status"] == 0 and bound["cleared"] and bound["events"] == [], (
            mutation
        )
        assert _fields(bound["diagnostic"])["error"] != "0", mutation
        assert asynchronous["status"] == 1 and asynchronous["events"] == [], mutation
    # Empty systems still carry the nonempty batch's offset endpoints. A zero
    # batch is rejected above, while zero atoms/shells/AOs is a valid shape.
    assert rows["topology-bind/empty-systems"]["published"]
    assert _calls(rows["topology-bind/empty-systems"]) == ["attributes"] * 7 + [
        "launch",
        "last-error",
        "copy",
        "sync",
    ]


def test_all_diagnostic_alias_fields_reject_without_launch(bridge_traces: dict) -> None:
    rows = bridge_traces["current"]
    for index in range(14):
        bound = rows[f"topology-bind/diagnostic-alias-{index}"]
        asynchronous = rows[f"topology-async/diagnostic-alias-{index}"]
        assert bound["status"] == 0 and bound["cleared"]
        assert _calls(bound) == ["attributes"] * 15
        assert bound["diagnostic"] == (
            f"Gfn2PlanSchemaDiagnostic{{error=8;field={index + 2};index=-1;}}"
        )
        assert asynchronous["status"] == 1 and asynchronous["events"] == []


def test_device_response_injection_does_not_model_kernel_semantics(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    # The zero generation is a device semantic failure. Host validation must
    # still enqueue it verbatim, and its outcome is outside this CPU gate.
    row = rows["provenance-async/zero-generation"]
    assert row["status"] == 0 and _calls(row) == ["launch", "last-error"]
    assert "|0|meta+69632|meta+61440" in row["events"][0]
    for operation in ("topology-bind", "provenance-bind", "element"):
        row = rows[operation + "/managed"]
        assert row["published"] and row["status"] == 0
        assert all(
            event.endswith("|0|3")
            for event in row["events"]
            if event.startswith("attributes|")
        )
    for mutation in ("active-null", "batch-generation"):
        row = rows["provenance-bind/" + mutation]
        assert row["published"]
        assert _calls(row) == ["attributes"] * 2 + [
            "launch",
            "last-error",
            "copy",
            "sync",
        ]


def test_self_binding_retains_clear_before_validate_semantics(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    for operation, field in (("topology-bind", "1"), ("provenance-bind", "16")):
        row = rows[operation + "/self-binding"]
        assert _fields(row["input"])["plan_token"] == str(0x5ABC)
        assert row["status"] == 0 and row["events"] == [] and row["cleared"]
        assert _fields(row["diagnostic"]) == {
            "error": "1",
            "field": field,
            "index": "-1",
        }
    row = rows["element/self-binding"]
    assert _fields(row["input"])["plan_token"] == str(0x5ABC)
    assert row["status"] == 0 and row["events"] == [] and row["cleared"]


def test_host_diagnostic_alias_preserves_mutation_and_publication_stage(
    bridge_traces: dict,
) -> None:
    rows = bridge_traces["current"]
    invalid = rows["topology-bind/host-diagnostic-index-invalid"]
    assert invalid["observations"] == [
        "before|index=777|output-token=99",
        "after|index=-1|output-token=0",
    ]
    assert invalid["events"] == [] and invalid["cleared"]
    valid = rows["topology-bind/host-diagnostic-index-valid"]
    assert _fields(valid["input"])["atom_shell_offsets"] == "host-diagnostic+8"
    assert valid["observations"] == (
        ["before|index=777|output-token=99"]
        + ["attributes|index=-1|output-token=0"] * 7
        + [
            f"{stage}|index=-1|output-token=0"
            for stage in (
                "launch",
                "last-error",
                "copy-enter",
                "copy-leave",
                "sync",
            )
        ]
        + [f"after|index=-1|output-token={0x5ABC}"]
    )
    assert valid["published"] and valid["binding"] == valid["input"]
    semantic = rows["topology-bind/host-diagnostic-index-semantic-error"]
    assert semantic["observations"][-4:] == [
        "copy-enter|index=-1|output-token=0",
        "copy-leave|index=1|output-token=0",
        "sync|index=1|output-token=0",
        "after|index=1|output-token=0",
    ]
    assert semantic["cleared"]
    for stage, status in (("copy-error", 702), ("sync-error", 703)):
        row = rows["topology-bind/host-diagnostic-index-" + stage]
        assert row["status"] == status and row["cleared"]
        assert row["observations"][-1] == "after|index=-1|output-token=0"
