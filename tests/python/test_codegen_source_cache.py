"""Regression tests for persistent generated-source caching."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "run_codegen.py"


def _run_codegen(
    *,
    source_root: Path,
    build: Path,
    cache: Path,
    generator: Path,
    dependency: Path,
) -> None:
    output = build / "generated.hpp"
    byproduct = build / "side.txt"
    env = os.environ.copy()
    env["GENERATIVEQC_CODEGEN_CACHE"] = str(cache)
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--depfile",
            str(output.with_suffix(".hpp.d")),
            "--target",
            str(output),
            "--byproduct",
            str(byproduct),
            "--dependency",
            str(dependency),
            "--source-root",
            str(source_root),
            str(generator),
            "--output",
            str(output),
            "--byproduct",
            str(byproduct),
            "--data",
            str(dependency),
        ],
        check=True,
        env=env,
    )


def _clear_generated(build: Path) -> None:
    for name in ("generated.hpp", "generated.hpp.d", "side.txt"):
        (build / name).unlink(missing_ok=True)


def test_codegen_cache_restores_and_invalidates_exact_dependencies(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build = tmp_path / "build"
    cache = tmp_path / "cache"
    source_root.mkdir()
    build.mkdir()
    helper = source_root / "helper.py"
    helper.write_text('VALUE = "alpha"\n', encoding="utf-8")
    data = source_root / "data.txt"
    data.write_text("one\n", encoding="utf-8")
    generator = source_root / "generator.py"
    generator.write_text(
        """\
import argparse
from pathlib import Path
import helper

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--byproduct", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
args = parser.parse_args()
counter = Path(__file__).with_name("counter.txt")
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(f"{helper.VALUE}:{args.data.read_text().strip()}\\n")
args.byproduct.write_text(f"side:{helper.VALUE}\\n")
""",
        encoding="utf-8",
    )
    counter = source_root / "counter.txt"

    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "1"

    _clear_generated(build)
    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "1"
    assert (build / "generated.hpp").read_text() == "alpha:one\n"
    assert (build / "side.txt").read_text() == "side:alpha\n"
    depfile = (build / "generated.hpp.d").read_text()
    assert str(helper) in depfile
    assert str(data) not in depfile

    helper.write_text('VALUE = "beta"\n', encoding="utf-8")
    _clear_generated(build)
    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "2"
    assert (build / "generated.hpp").read_text() == "beta:one\n"

    data.write_text("two\n", encoding="utf-8")
    _clear_generated(build)
    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "3"
    assert (build / "generated.hpp").read_text() == "beta:two\n"

    for artifact in cache.rglob("target-000.hpp"):
        artifact.write_text("corrupt", encoding="utf-8")
    _clear_generated(build)
    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "4"
    assert (build / "generated.hpp").read_text() == "beta:two\n"

    _clear_generated(build)
    _run_codegen(
        source_root=source_root,
        build=build,
        cache=cache,
        generator=generator,
        dependency=data,
    )
    assert counter.read_text() == "4"


def test_cmake_forwards_declared_codegen_dependencies_to_cache() -> None:
    helper = (ROOT / "cmake" / "GenerativeQCGenerated.cmake").read_text()
    assert "_generativeqc_codegen_byproducts" in helper
    assert "_generativeqc_codegen_dependencies" in helper
    assert '--byproduct "${_generativeqc_byproduct}"' in helper
    assert '--dependency "${_generativeqc_dependency}"' in helper
