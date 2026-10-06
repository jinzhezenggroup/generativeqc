"""Run bounded CuMetal-backed QC CUDA tests on the Apple GPU.

Routine CI selects a small representative endpoint set. Scheduled/manual
qualification selects a broader explicit CuMetal-compatible manifest. Keep the
manifest explicit: NVIDIA-only/Slurm-only tests must never be counted as CuMetal
coverage, and adding every CUDA-named source test would make this lane both slow
and misleading.

Every selected pytest invocation must execute at least one Apple-GPU dispatch.
Unexpected skips, empty selections, missing provenance, per-test timeouts, and
exhaustion of the whole-suite time budget are failures. An explicit, pin-bound
upstream quarantine records only its exact endpoint groups as unqualified skips;
without that opt-in, all selected endpoints retain the strict acceptance gate.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

QUARANTINE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "manifests/cumetal_qc_quarantine.json"
)
MODE = os.environ.get("CUMETAL_CUDA_TEST_MODE", "gate").strip().lower()
TIMEOUT_SECONDS = int(os.environ.get("CUMETAL_CUDA_TEST_TIMEOUT_SECONDS", "90"))
SUITE_BUDGET_SECONDS = int(
    os.environ.get(
        "CUMETAL_CUDA_SUITE_BUDGET_SECONDS",
        "300" if MODE == "gate" else "2400",
    )
)

# Required on routine PR/merge-queue runs. These deliberately span the public
# RHF, UHF, density-fitting, and DFT CUDA owners without pulling
# in the long force/Hessian/post-HF qualification suites.
GATE_NODEIDS = (
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_rhf_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_uhf_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_density_fitting_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_pbe_rks_matches_cpu_reference",
)

# Scheduled/manual qualification. These are existing public CUDA endpoint tests
# that do not require an NVIDIA-only allocator, Slurm identity, or a native nvcc
# toolchain. Keep expensive production-grid/ECP/post-HF campaigns in their
# dedicated GPU qualification lanes rather than stretching this Apple runner.
QUALIFICATION_NODEIDS = GATE_NODEIDS + (
    "tests/python/test_dft_cuda.py::test_native_cuda_dft_matches_independently_converged_cpu_endpoint",
    "tests/python/test_calculator.py::test_cuda_energy_only_output_selection_omits_forces",
    "tests/python/test_batch.py::test_cuda_real_spherical_batch_reuses_fixed_topology_plan",
    "tests/python/test_batch.py::test_cuda_bounded_direct_streaming_matches_exact_replay",
    "tests/python/test_batch.py::test_cuda_uhf_ragged_batch_warm_start_and_failure_isolation",
    "tests/python/test_calculator.py::test_cuda_spherical_def2_svp_water_matches_pyscf",
    "tests/python/test_calculator.py::test_cuda_def2_tzvp_water_uses_graph_native_eigensolver",
    "tests/python/test_calculator.py::test_cartesian_d_f_cuda_matches_pyscf_libcint_reference",
    "tests/python/test_calculator.py::test_cuda_uhf_direct_jk_matches_pyscf_and_force_finite_difference",
    "tests/python/test_dft_cuda.py::test_native_cuda_dft_ragged_batch_replay_and_failure_isolation",
)


def selected_nodeids() -> list[str]:
    if MODE == "gate":
        return list(GATE_NODEIDS)
    if MODE == "full":
        return list(QUALIFICATION_NODEIDS)
    raise SystemExit(f"unsupported CUMETAL_CUDA_TEST_MODE={MODE!r}")


def quarantine() -> tuple[set[str], str]:
    """Admit only the reviewed provider contract and exact endpoint allowlist."""
    requested = os.environ.get("CUMETAL_CUDA_QUARANTINE", "")
    if not requested:
        return set(), ""
    manifest = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
    if requested != manifest["id"]:
        raise SystemExit(f"unknown CuMetal QC quarantine: {requested!r}")
    for variable, field in (
        ("CUMETAL_COMMIT", "provider_commit"),
        ("CUMETAL_PTX_BACKEND", "ptx_backend"),
        ("CUMETAL_FP64_MODE", "fp64_mode"),
    ):
        if os.environ.get(variable) != manifest[field]:
            raise SystemExit(
                f"CuMetal QC quarantine contract changed: {variable}; "
                "revisit the quarantine and qualify the actual endpoints"
            )
    if not os.environ.get("CUMETAL_ROOT"):
        raise SystemExit("CuMetal QC quarantine requires CUMETAL_ROOT")
    nodes = manifest["nodeids"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or not all(isinstance(node, str) for node in nodes)
        or len(set(nodes)) != len(nodes)
        or not set(nodes).issubset(QUALIFICATION_NODEIDS)
        or not isinstance(manifest["reason"], str)
        or not manifest["reason"].strip()
    ):
        raise SystemExit("invalid exact CuMetal QC quarantine manifest")
    return set(nodes), f"{requested}: {manifest['reason']}"


def record_quarantine(junit: Path, nodeid: str, reason: str) -> None:
    """Report a skipped endpoint group without manufacturing executed testcases."""
    suite = ET.Element(
        "testsuite", name="CuMetal QC quarantine", tests="1", skipped="1"
    )
    case = ET.SubElement(suite, "testcase", name=nodeid, classname="cumetal.quarantine")
    ET.SubElement(case, "skipped", message=reason)
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
    print(f"::warning::QUARANTINED (QC NOT QUALIFIED): {nodeid}: {reason}", flush=True)


def report_quarantine(nodeids: list[str], reason: str) -> None:
    if not nodeids:
        return
    summary = (
        f"CuMetal QC NOT QUALIFIED: {len(nodeids)} endpoint groups quarantined.\n"
        "Native build/runtime smoke evidence does not qualify these QC endpoints.\n"
        f"Reason: {reason}\n" + "".join(f"- SKIPPED: {nodeid}\n" for nodeid in nodeids)
    )
    print(summary, flush=True)
    if github_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(github_summary, "a", encoding="utf-8") as output:
            output.write(summary + "\n")


def stream_process(
    command: list[str], timeout_seconds: int
) -> tuple[int | None, str, bool]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    output: list[str] = []

    def pump() -> None:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(
            f"\nERROR: CUDA pytest exceeded {timeout_seconds}s; killing process group",
            flush=True,
        )
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return_code = None
        process.wait()
    reader.join(timeout=10)
    return return_code, "".join(output), timed_out


def junit_status(path: Path) -> tuple[int, list[str], list[str]]:
    if not path.exists():
        return 0, [], []
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    skipped = []
    missing_provenance = []
    for case in cases:
        skip = case.find("skipped")
        if skip is not None:
            reason = skip.get("message") or skip.text or "<no reason>"
            skipped.append(f"{case.get('name', '<unnamed>')}: {reason}")
        # Pytest's FD capture retains native CuMetal stdout/stderr separately
        # for each parameterized endpoint. A sibling's dispatch or a session
        # smoke must not qualify a case with no Apple-GPU execution of its own.
        output = "\n".join(
            case.findtext(stream, "") for stream in ("system-out", "system-err")
        )
        if "device=apple_gpu" not in output or "launch_success=true" not in output:
            missing_provenance.append(case.get("name", "<unnamed>"))
    return len(cases), skipped, missing_provenance


def main() -> None:
    nodeids = selected_nodeids()
    quarantined, quarantine_reason = quarantine()
    started = time.monotonic()
    print(
        f"Running CuMetal CUDA mode={MODE}: {len(nodeids)} endpoint groups; "
        f"{TIMEOUT_SECONDS}s/group timeout; {SUITE_BUDGET_SECONDS}s suite budget",
        flush=True,
    )
    failures: list[str] = []
    skipped_nodeids: list[str] = []
    executed = 0

    for index, nodeid in enumerate(nodeids):
        junit = Path(f"/tmp/generativeqc-cuda-test-{index}.xml")
        junit.unlink(missing_ok=True)
        if nodeid in quarantined:
            record_quarantine(junit, nodeid, quarantine_reason)
            skipped_nodeids.append(nodeid)
            continue
        elapsed = time.monotonic() - started
        remaining = int(SUITE_BUDGET_SECONDS - elapsed)
        if remaining <= 0:
            failures.append(
                f"SUITE BUDGET EXHAUSTED before {nodeid} "
                f"({SUITE_BUDGET_SECONDS}s total)"
            )
            break

        timeout = min(TIMEOUT_SECONDS, remaining)
        print(
            f"\n::group::CUDA pytest {index + 1}/{len(nodeids)}: {nodeid}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            nodeid,
            "-vv",
            "-ra",
            "--capture=fd",
            "-o",
            "junit_logging=all",
            "-o",
            "junit_log_passing_tests=true",
            f"--junitxml={junit}",
        ]
        return_code, _output, timed_out = stream_process(command, timeout)
        executed += 1
        cases, skipped, missing_provenance = junit_status(junit)
        failure: str | None = None
        if timed_out:
            # Only a real timeout may trigger the separate one-shot diagnostic.
            if (
                MODE == "gate"
                and nodeid == GATE_NODEIDS[0]
                and (github_output := os.environ.get("GITHUB_OUTPUT"))
            ):
                with open(github_output, "a", encoding="utf-8") as output:
                    output.write("rhf_timed_out=true\n")
            failure = f"TIMEOUT: {nodeid}"
        elif return_code != 0:
            failure = f"FAILED: {nodeid} (exit {return_code})"
        elif cases < 1:
            failure = f"INVALID RESULT: {nodeid} produced no testcases"
        elif skipped:
            failure = f"SKIPPED: {nodeid} ({len(skipped)}/{cases} cases): " + "; ".join(
                skipped
            )
        elif missing_provenance:
            failure = f"MISSING APPLE-GPU PROVENANCE: {nodeid}: " + ", ".join(
                missing_provenance
            )
        if failure is not None:
            failures.append(failure)
        print("::endgroup::", flush=True)
        if failure is not None and MODE == "gate":
            break

    report_quarantine(skipped_nodeids, quarantine_reason)
    if failures:
        print("\nCUDA test failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    elapsed = time.monotonic() - started
    print(
        f"\nExecuted {executed} CuMetal CUDA endpoint groups with zero unexpected skips; "
        f"{len(skipped_nodeids)} quarantined (not qualified) "
        f"in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
