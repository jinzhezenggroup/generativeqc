"""Run unchanged CPU CBS tests, retaining public endpoint returns verbatim."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import traceback


def serial(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def wrap(original, emit):
    """Preserve the original return/exception; a recorder failure cannot hide it."""
    @functools.wraps(original)
    def observed(owner, atoms, *args, **kwargs):
        request = {"atoms": atoms, "args": args, "kwargs": kwargs}
        try:
            result = original(owner, atoms, *args, **kwargs)
        except BaseException:
            try:
                emit({**request, "exception": traceback.format_exc()})
            except BaseException:
                # Preserve the scientific exception. The test failure is retained
                # separately; absence of a result cannot qualify as PASS.
                traceback.print_exc()
            raise
        emit({**request, "result": result})
        return result
    return observed


def selftest():
    """Cheap identity, mutation, cleanup, failure and restoration checks."""
    events = []
    atoms = [[1, [0.0, 0.0, 0.0]]]
    result = {"forces": [[1.0, 2.0, 3.0]]}
    error = RuntimeError("scientific failure")

    class Fixture:
        def singlepoint(self, supplied, *, fail=False):
            assert supplied is atoms
            try:
                if fail:
                    raise error
                return result
            finally:
                events.append("closed")

    original = Fixture.singlepoint
    records = []
    try:
        Fixture.singlepoint = wrap(original, records.append)
        assert Fixture().singlepoint(atoms) is result
        try:
            Fixture().singlepoint(atoms, fail=True)
        except RuntimeError as exc:
            assert exc is error
        else:
            raise AssertionError("exception swallowed")
        assert atoms == [[1, [0.0, 0.0, 0.0]]]
        assert result == {"forces": [[1.0, 2.0, 3.0]]}
        assert events == ["closed", "closed"]
        assert len(records) == 2 and "exception" in records[1]
        def broken(record):
            raise OSError("recorder failure")
        Fixture.singlepoint = wrap(original, broken)
        try:
            Fixture().singlepoint(atoms, fail=True)
        except RuntimeError as exc:
            assert exc is error
        try:
            Fixture().singlepoint(atoms)
        except OSError:
            pass
        else:
            raise AssertionError("recorder failure became PASS")
    finally:
        Fixture.singlepoint = original
    assert Fixture.singlepoint is original
    print("observer identity/immutability/cleanup/exception/restoration PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--case", choices=("all", "water-algebra", "h2-dz-tz", "support"), default="all")
    options = parser.parse_args()
    if options.selftest:
        selftest()
        return 0
    import pytest
    from generativeqc import CCSDTCBSCalculator
    if os.environ.get("GENERATIVEQC_CBS_LARGE_TEST") != "1":
        raise RuntimeError("explicit H2 force qualification required")
    if os.environ.get("GENERATIVEQC_CBS_CUDA_TEST") == "1":
        raise RuntimeError("this is a CPU-only slice")
    options.output.mkdir(exist_ok=False)
    stream = (options.output / "endpoints.jsonl").open("x")
    reports = []
    traces = {}
    def emit(record):
        stream.write(json.dumps(record, default=serial, allow_nan=False) + "\n")
        stream.flush()
    class Reports:
        def pytest_runtest_logreport(self, report):
            reports.append({"nodeid": report.nodeid, "when": report.when,
                            "outcome": report.outcome, "duration": report.duration,
                            "longrepr": str(report.longrepr) if report.longrepr else None})
        def pytest_runtest_teardown(self, item):
            path = item.funcargs.get("completed_work_path")
            case = item.funcargs.get("cbs_case")
            if path and path.exists() and case:
                traces[case["case"]] = path.read_text()
    original = CCSDTCBSCalculator.singlepoint
    try:
        CCSDTCBSCalculator.singlepoint = wrap(original, emit)
        arguments = ["-q", "-ra", "tests/python/test_ccsdt_cbs_native.py",
                     "--basetemp=" + str(options.output / "pytest-tmp")]
        if options.case == "support":
            arguments += ["-k", "not h2-dz-tz and not water-algebra and not cuda"]
        elif options.case != "all":
            arguments += ["-k", options.case + " and not cuda"]
        code = pytest.main(arguments, plugins=[Reports()])
    finally:
        CCSDTCBSCalculator.singlepoint = original
        stream.close()
        (options.output / "reports.json").write_text(json.dumps(reports))
        (options.output / "traces.json").write_text(json.dumps(traces))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
