"""Semantic work receipts bind the production DF schedule and trace counters."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.method.df_exchange_schedule import (
    native_header,
    projected_exchange_schedule,
)

from tools.audit_producer_work import (
    ReceiptError,
    compare,
    read_json,
    read_trace,
    schedule_receipt,
    trace_receipt,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/generativeqc_compiler/method/df_exchange_schedule.py"
HASH = "a" * 64


def schedule(**changes: object) -> dict:
    params = {
        "root": ROOT,
        "n": 12,
        "auxiliaries": 5,
        "rank": 2,
        "capacity": 48,
        "dense_row_blocks": 3,
        "dense_output_blocks": 3,
        "triangular": True,
        "scientific_problem": "shape:12x5:occupied-rank-2",
        "dependency_identity": "geometry+basis+occupied-coefficients:fixture-1",
        "build_sha256": HASH,
    }
    params.update(changes)
    return schedule_receipt(**params)


def test_production_schedule_census_and_source_bound_ratchet(tmp_path: Path) -> None:
    baseline = schedule()
    assert baseline["work"]["logical_elements"] == 12
    assert baseline["work"]["executed_elements"] == 16
    assert baseline["work"]["producer_callbacks"] == 4
    assert baseline["work"]["outer_consumer_multiplicity"] == 3
    assert compare(baseline, baseline, SOURCE, SOURCE)["status"] == "PASS"

    # A changed production schedule with more source visits must trip the ratchet.
    candidate = copy.deepcopy(baseline)
    candidate["work"]["executed_elements"] += 4
    candidate["work"]["producer_callbacks"] += 1
    result = compare(baseline, candidate, SOURCE, SOURCE)
    assert result["status"] == "FAIL"
    assert result["executed_ratio"] == "5/4"
    assert result["callback_ratio"] == "5/4"
    assert result["runtime_acceptance"] == "INCOMPLETE"
    assert result["classification"] == "work change; reuse unproven"

    changed_source = tmp_path / "changed_schedule.py"
    changed_source.write_bytes(SOURCE.read_bytes() + b"\n# changed\n")
    with pytest.raises(ReceiptError, match="stale source"):
        compare(baseline, baseline, SOURCE, changed_source)
    candidate["identity"]["source_sha256"] = hashlib.sha256(
        changed_source.read_bytes()
    ).hexdigest()
    assert compare(baseline, candidate, SOURCE, changed_source)["status"] == "FAIL"


def test_schedule_census_executes_captured_bytes_after_same_path_change(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "python/generativeqc_compiler"
    method = package / "method"
    method.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (method / "__init__.py").write_text("")
    shutil.copy2(SOURCE, method / SOURCE.name)
    shutil.copy2(
        ROOT / "python/generativeqc_compiler/method/df_occupied_gram_cuda.py",
        method / "df_occupied_gram_cuda.py",
    )
    script = r"""
import json
import sys
from pathlib import Path

repository, checkout = map(Path, sys.argv[1:])
sys.path.insert(0, str(repository))
from tools.audit_producer_work import schedule_receipt

sys.path.insert(0, str(checkout / "python"))
arguments = {
    "root": checkout,
    "n": 12,
    "auxiliaries": 5,
    "rank": 2,
    "capacity": 48,
    "dense_row_blocks": 3,
    "dense_output_blocks": 3,
    "triangular": True,
    "scientific_problem": "same-path-change",
    "dependency_identity": "fixture",
    "build_sha256": "a" * 64,
}
before = schedule_receipt(**arguments)
source = checkout / "python/generativeqc_compiler/method/df_exchange_schedule.py"
text = source.read_text()
needle = "maximum_rows = min(n, capacity // (auxiliaries * rank), capacity // n)"
assert needle in text
source.write_text(text.replace(needle, "maximum_rows = min(n, 2)"))
after = schedule_receipt(**arguments)
print(json.dumps({"before": before, "after": after}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["before"]["work"]["executed_elements"] == 16
    assert observed["before"]["work"]["producer_callbacks"] == 4
    assert observed["after"]["work"]["executed_elements"] == 32
    assert observed["after"]["work"]["producer_callbacks"] == 16
    assert (
        observed["before"]["identity"]["source_sha256"]
        != observed["after"]["identity"]["source_sha256"]
    )


def test_schedule_census_rejects_a_loaded_package_outside_source_root(
    tmp_path: Path,
) -> None:
    wrong_root = tmp_path / "other-checkout"
    wrong_source = wrong_root / SOURCE.relative_to(ROOT)
    wrong_source.parent.mkdir(parents=True)
    shutil.copy2(SOURCE, wrong_source)
    with pytest.raises(ReceiptError, match="outside the bound source root"):
        schedule(root=wrong_root)


def test_source_driven_once_and_nested_consumer_amplification() -> None:
    once = schedule(capacity=120)
    nested = schedule(capacity=48)
    assert once["work"]["executed_elements"] == once["work"]["logical_elements"]
    assert nested["work"]["executed_elements"] > nested["work"]["logical_elements"]
    assert nested["work"]["outer_consumer_multiplicity"] > 1


def test_unadmitted_budget_fails_closed() -> None:
    with pytest.raises(ReceiptError, match="not admitted"):
        schedule(capacity=1)


@pytest.mark.parametrize(
    "n,a,rank,capacity,dense_rows,dense_outputs,triangular",
    [
        (2, 3, 1, 6, 2, 2, True),
        (7, 5, 2, 20, 3, 3, True),
        (12, 5, 2, 48, 3, 3, True),
        (12, 5, 2, 48, 3, 3, False),
        (13, 7, 3, 84, 4, 4, True),
    ],
)
def test_census_enumerates_tail_and_retained_neighbor(
    n: int,
    a: int,
    rank: int,
    capacity: int,
    dense_rows: int,
    dense_outputs: int,
    triangular: bool,
) -> None:
    receipt = schedule(
        n=n,
        auxiliaries=a,
        rank=rank,
        capacity=capacity,
        dense_row_blocks=dense_rows,
        dense_output_blocks=dense_outputs,
        triangular=triangular,
    )
    assert receipt["work"]["executed_elements"] >= n
    assert receipt["work"]["memory_budget_bytes"] == capacity * 8
    assert receipt["work"]["peak_bytes"] is None


def test_domain_owner_coverage_and_bad_fields_fail_closed() -> None:
    baseline = schedule()
    candidate = copy.deepcopy(baseline)
    candidate["identity"]["resource_regime"] += ";other-budget"
    with pytest.raises(ReceiptError, match="domains differ"):
        compare(baseline, candidate, SOURCE, SOURCE)
    candidate = copy.deepcopy(baseline)
    candidate["identity"]["invalidation_owner"] = "other-owner"
    with pytest.raises(ReceiptError, match="domains differ"):
        compare(baseline, candidate, SOURCE, SOURCE)
    candidate = copy.deepcopy(baseline)
    candidate["work"]["memory_budget_bytes"] += 8
    with pytest.raises(ReceiptError, match="memory budgets differ"):
        compare(baseline, candidate, SOURCE, SOURCE)
    for field, bad in (
        ("logical_elements", True),
        ("executed_elements", 1.5),
        ("producer_callbacks", -1),
        ("memory_budget_bytes", 2**64),
    ):
        invalid = copy.deepcopy(baseline)
        invalid["work"][field] = bad
        with pytest.raises(ReceiptError):
            validate(invalid)
    invalid = copy.deepcopy(baseline)
    del invalid["identity"]["dependency_identity"]
    with pytest.raises(ReceiptError):
        validate(invalid)
    invalid = copy.deepcopy(baseline)
    invalid["evidence"]["coverage_complete"] = False
    assert compare(baseline, invalid, SOURCE, SOURCE)["status"] == "INCOMPLETE"
    invalid = copy.deepcopy(baseline)
    invalid["evidence"]["execution_complete"] = True
    with pytest.raises(ReceiptError, match="static schedule"):
        validate(invalid)
    invalid = copy.deepcopy(baseline)
    invalid["evidence"]["reusable_dependency_proven"] = True
    with pytest.raises(ReceiptError, match="reuse_proof_sha256"):
        validate(invalid)
    invalid = copy.deepcopy(baseline)
    invalid["evidence"]["phase"] = []
    with pytest.raises(ReceiptError, match="phase"):
        validate(invalid)
    with pytest.raises(ReceiptError, match="triangular"):
        schedule(triangular=1)


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ReceiptError, match="duplicate JSON key"):
        read_json(path)
    path.write_text('{"schema":NaN}')
    with pytest.raises(ReceiptError, match="nonstandard JSON"):
        read_json(path)


def test_jsonl_trace_selection_rejects_duplicate_and_partial(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    first = {"id": 1, **trace()}
    second = {"id": 2, **trace()}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    assert read_trace(path, 2)["id"] == 2
    with pytest.raises(ReceiptError, match="exactly one"):
        read_trace(path, 3)
    path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n")
    with pytest.raises(ReceiptError, match="exactly one"):
        read_trace(path, 1)
    path.write_text(json.dumps(first) + "\n{" + "\n")
    with pytest.raises(ReceiptError, match="invalid trace JSONL"):
        read_trace(path, 1)


def trace() -> dict:
    return {
        "schema": "generativeqc.df_trace",
        "version": 1,
        "execution": "stream",
        "valid": True,
        "cuda_error": 0,
        "dropped_tiles": 0,
        "dropped_regions": 0,
        "counters": {"raw_tile_productions": 3, "raw_value_bytes": 96},
        "tiles": [
            {
                "system": 0,
                "pair_begin": 0,
                "pair_count": 2,
                "auxiliary_begin": 0,
                "auxiliary_count": 2,
                "derivative_coordinate": -1,
                "transformed": False,
                "productions": 2,
            },
            {
                "system": 0,
                "pair_begin": 2,
                "pair_count": 1,
                "auxiliary_begin": 0,
                "auxiliary_count": 4,
                "derivative_coordinate": -1,
                "transformed": False,
                "productions": 1,
            },
        ],
    }


def adapted_trace(row: dict) -> dict:
    return trace_receipt(
        row,
        source=ROOT / "src/runtime/cuda_component_trace.cpp",
        scientific_problem="test-df",
        resource_regime="same-budget",
        dependency_identity="same-input",
        build_sha256=HASH,
        execution_identity="diagnostic-invocation-1",
        tile_kind="raw",
    )


def test_real_trace_counter_adapter_is_incomplete() -> None:
    adapted = adapted_trace(trace())
    assert adapted["work"]["logical_elements"] == 8
    assert adapted["work"]["executed_elements"] == 12
    assert adapted["work"]["producer_callbacks"] == 3
    assert adapted["evidence"]["execution_complete"] is False
    assert (
        compare(
            adapted,
            adapted,
            ROOT / "src/runtime/cuda_component_trace.cpp",
            ROOT / "src/runtime/cuda_component_trace.cpp",
        )["status"]
        == "INCOMPLETE"
    )
    captured = trace()
    captured["execution"] = "graph_capture"
    assert adapted_trace(captured)["evidence"]["phase"] == "graph_capture"
    forged = copy.deepcopy(adapted)
    forged["evidence"]["execution_complete"] = True
    with pytest.raises(ReceiptError):
        validate(forged)


def test_existing_streamed_row_counter_adapter() -> None:
    native = (ROOT / "src/scf/cuda/df_occupied_exchange.cpp").read_text()
    assert 'trace_counter("streamed_occupied_raw_generation_rows", count)' in native
    row = trace()
    row.update(source_backed=True, streamed=True, nbf=12, systems=1)
    row["counters"].update(
        streamed_occupied_raw_generation_rows=16,
        streamed_occupied_row_blocks=3,
        streamed_whitening_factor_gemms=4,
        streamed_occupied_source_first=1,
    )
    receipt = trace_receipt(
        row,
        source=ROOT / "src/scf/cuda/df_occupied_exchange.cpp",
        scientific_problem="shape:12x5:occupied-rank-2",
        resource_regime=schedule()["identity"]["resource_regime"],
        dependency_identity="geometry+basis+occupied-coefficients:fixture-1",
        build_sha256=HASH,
        execution_identity="diagnostic-invocation-2",
        tile_kind="streamed_rows",
    )
    assert receipt["work"]["executed_elements"] == 16
    assert receipt["work"]["producer_callbacks"] == 4
    assert receipt["work"]["outer_consumer_multiplicity"] == 3
    assert receipt["identity"]["reuse_owner"] == schedule()["identity"]["reuse_owner"]
    assert receipt["evidence"]["execution_complete"] is False


def test_trace_counter_duplicate_and_overflow_fail_closed() -> None:
    for mutation in (
        "counter",
        "duplicate",
        "overflow",
        "missing",
        "truncated",
        "invalid",
    ):
        row = trace()
        if mutation == "counter":
            row["counters"]["raw_value_bytes"] = 95
        elif mutation == "duplicate":
            row["tiles"].append(copy.deepcopy(row["tiles"][0]))
        elif mutation == "overflow":
            row["tiles"][0]["productions"] = 2**64
        else:
            if mutation == "missing":
                del row["counters"]["raw_tile_productions"]
            elif mutation == "truncated":
                row["dropped_tiles"] = 1
            else:
                row["valid"] = False
        with pytest.raises(ReceiptError):
            adapted_trace(row)
    row = trace()
    row["version"] = True
    with pytest.raises(ReceiptError, match="unsupported production trace"):
        adapted_trace(row)


def test_exact_ratio_and_legitimate_recomputation() -> None:
    baseline = schedule()
    candidate = copy.deepcopy(baseline)
    candidate["work"]["executed_elements"] *= 2
    candidate["work"]["outer_consumer_multiplicity"] *= 2
    result = compare(baseline, candidate, SOURCE, SOURCE)
    assert result["candidate_amplification"] == "8/3"
    assert result["reusable_dependency_proven"] is False
    assert "bug" not in result["classification"]
    candidate = copy.deepcopy(baseline)
    candidate["work"]["producer_callbacks"] += 1
    result = compare(baseline, candidate, SOURCE, SOURCE)
    assert result["status"] == "FAIL"
    assert result["executed_ratio"] == "1/1"
    assert result["callback_ratio"] == "5/4"


@pytest.fixture(scope="module")
def native_callbacks(tmp_path_factory: Any) -> Any:
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip(
            "host C++ and verified sccache/ccache required for native callback census"
        )
    subprocess.run([launcher, "--version"], check=True, capture_output=True, text=True)
    before = subprocess.run(
        [launcher, "--show-stats"], check=True, capture_output=True, text=True
    ).stdout
    directory = tmp_path_factory.mktemp("producer-work-native")
    (directory / "generated_df_exchange_schedule.hpp").write_text(native_header())
    source = directory / "callbacks.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "generated_df_exchange_schedule.hpp"

int main(int argc, char** argv) {
  if (argc != 4) return 1;
  const auto n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  const auto rows = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
  const bool triangular = argv[3][0] == '1';
  std::size_t begins[2] = {n, n}, counts[2] = {0, 0};
  std::size_t generated = 0, callbacks = 0;
  std::vector<int> coverage(n * n, 0);
  const bool ok = generativeqc::scf::generated::visit_projected_exchange(
      n, rows, triangular,
      [&](std::size_t begin, std::size_t count, std::size_t slot) {
        if (slot >= 2 || !count || begin + count > n) return false;
        begins[slot] = begin;
        counts[slot] = count;
        generated += count;
        ++callbacks;
        return true;
      },
      [&](std::size_t r, std::size_t nr, std::size_t c, std::size_t nc,
          std::size_t left, std::size_t right, bool) {
        if (left >= 2 || right >= 2 || begins[left] != r || counts[left] != nr ||
            begins[right] != c || counts[right] != nc) return false;
        for (std::size_t i = r; i < r + nr; ++i)
          for (std::size_t j = c; j < c + nc; ++j) {
            if (++coverage[i * n + j] != 1) return false;
            if (triangular && r != c && ++coverage[j * n + i] != 1) return false;
          }
        return true;
      });
  if (!ok || std::any_of(coverage.begin(), coverage.end(),
                         [](int value) { return value != 1; })) return 2;
  std::cout << generated << ' ' << callbacks << '\n';
  return 0;
}
"""
    )
    object_file = directory / "callbacks.o"
    executable = directory / "callbacks"
    compile_command = [
        launcher,
        compiler,
        "-std=c++20",
        "-O0",
        "-I" + str(directory),
        "-c",
        str(source),
        "-o",
        str(object_file),
    ]
    print("compiler-cache command:", " ".join(compile_command))
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
    link_command = [compiler, str(object_file), "-o", str(executable)]
    print("link command:", " ".join(link_command))
    subprocess.run(link_command, check=True, capture_output=True, text=True)
    after = subprocess.run(
        [launcher, "--show-stats"], check=True, capture_output=True, text=True
    ).stdout
    print("compiler-cache before:\n", before)
    print("compiler-cache after:\n", after)

    def query(n: int, rows: int, triangular: bool) -> tuple[int, int]:
        result = subprocess.run(
            [str(executable), str(n), str(rows), str(int(triangular))],
            check=True,
            capture_output=True,
            text=True,
        )
        return tuple(map(int, result.stdout.split()))

    return query


@pytest.mark.parametrize(
    "shape",
    [
        (12, 5, 2, 120, True),
        (12, 5, 2, 48, True),
        (12, 5, 2, 48, False),
        (13, 7, 3, 84, True),
    ],
)
def test_native_callback_census_matches_production_schedule(
    native_callbacks: Any, shape: tuple[int, int, int, int, bool]
) -> None:
    n, auxiliaries, rank, capacity, triangular = shape
    receipt = schedule(
        n=n,
        auxiliaries=auxiliaries,
        rank=rank,
        capacity=capacity,
        dense_row_blocks=4,
        dense_output_blocks=4,
        triangular=triangular,
    )
    policy = projected_exchange_schedule(
        n, auxiliaries, rank, capacity, 4, 4, triangular
    )
    assert native_callbacks(n, policy.rows, triangular) == (
        receipt["work"]["executed_elements"],
        receipt["work"]["producer_callbacks"],
    )


def test_native_callback_rejects_invalid_rows(native_callbacks: Any) -> None:
    # The production visitor must reject invalid widths before any callback.
    with pytest.raises(subprocess.CalledProcessError):
        native_callbacks(12, 0, True)
    with pytest.raises(subprocess.CalledProcessError):
        native_callbacks(12, 13, True)
