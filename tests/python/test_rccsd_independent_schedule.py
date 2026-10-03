"""Expanded verification keeps its equations, bounded work, and old capacity ceiling."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor import Program, execute, input_tensor, multiply
from generativeqc_compiler.tensor.complexity import analyze_complexity
from generativeqc_compiler.tensor.types import TensorSpec

from tools import generate_rccsd_native as codegen
from tools.generativeqc_cc.oracle import dense_feeds, random_case


def test_dependency_order_retains_shared_outputs_and_logical_identity() -> None:
    scalar = TensorSpec((), role="input")
    x, y, dead = (input_tensor(name, scalar) for name in ("x", "y", "dead"))
    shared = multiply(x, y)
    first = multiply(shared, x)
    last = multiply(shared, first)
    program = Program({"first": first, "last": last}, definitions=(dead,))
    serialized, logical_hash = program.dumps(), program.logical_hash
    order = program.dependency_order
    assert order == (x, y, shared, first, last)
    assert len(order) == len(set(order))
    assert dead not in order
    assert program.dumps() == serialized and program.logical_hash == logical_hash
    replay = Program.loads(serialized)
    assert [node.op for node in replay.dependency_order] == [n.op for n in order]


def test_independent_reassociation_bounds_contraction_degree() -> None:
    expanded = codegen.build_ccsd_program(2, 3, form="expanded", diagnostics=False)
    lambdas = codegen.build_lambda_programs(2, 3, form="expanded")
    for program in (expanded, lambdas.residual_vjp.program):
        strict = codegen._prepare_production(
            program, "cpu", preserve_reduction_order=True
        )
        fast = codegen._reassociated_independent(program, "cpu")
        assert analyze_complexity(strict).max_work_degree == 8
        assert analyze_complexity(fast).max_work_degree <= 6


def _array(name: str, value: np.ndarray) -> str:
    flat = np.asarray(value, dtype=np.float64).ravel()
    return f"const double {name}[]={{" + ",".join(float(x).hex() for x in flat) + "};"


def _case(o: int, v: int) -> str:
    """Use unlowered expanded TensorIR as the independent numerical reference."""
    f, g, t1, t2 = random_case(o, v, 2200 + 10 * o + v)
    feeds = dense_feeds(f, g, t1, t2)
    programs = codegen.build_lambda_programs(o, v, form="expanded")
    feeds.update(
        bar_correlation_energy=np.asarray(-1.0),
        bar_singles_residual=t1,
        bar_doubles_residual=t2,
    )
    graphs = (
        (
            "replay",
            codegen.build_ccsd_program(o, v, form="expanded", diagnostics=False),
            "",
            (
                ("correlation_energy", "energy", "1"),
                ("singles_residual", "r1", "o*v"),
                ("doubles_residual", "r2", "o*o*v*v"),
            ),
        ),
        (
            "lambda_independent_rhs",
            programs.energy_vjp.program,
            "bar_correlation_energy,",
            (("bar_t1", "t1", "o*v"), ("bar_t2", "t2", "o*o*v*v")),
        ),
        (
            "lambda_independent_transpose",
            programs.residual_vjp.program,
            "bar_singles_residual,bar_doubles_residual,",
            (("bar_t1", "t1", "o*v"), ("bar_t2", "t2", "o*o*v*v")),
        ),
    )
    lines = [
        f"static void case_{o}_{v}(){{",
        f"const std::size_t o={o},v={v};",
        "Inputs inputs{};",
    ]
    for name, value in feeds.items():
        lines.append(_array(name, value))
        if not name.startswith("bar_"):
            lines.append(f"inputs.{name}={name};")
    for name, graph, seeds, fields in graphs:
        expected = execute(graph, feeds).outputs
        for output, _, _ in fields:
            lines.append(_array(f"{name}_{output}", expected[output]))
        for variant in ("strict_", "reassociated_", ""):
            prefix = f"{name}_{variant}"
            lines += [
                "{",
                f"const auto count={prefix}arena_elements(o,v);",
                # Guard both ends and run with exactly the admitted capacity.
                "std::vector<double> arena(count+2,1234567.0);",
                f"auto result=run_{prefix}cpu(o,v,inputs,{seeds}arena.data()+1,count);",
            ]
            for output, field, count in fields:
                value = f"&result.{field}" if field == "energy" else f"result.{field}"
                lines.append(f"check_close({value},{name}_{output},{count});")
            lines += [
                'require(arena.front()==1234567.0 && arena.back()==1234567.0,"arena canary");',
                "bool rejected=false;",
                f"try{{run_{prefix}cpu(o,v,inputs,{seeds}arena.data()+1,count-1);}}",
                "catch(const std::length_error&){rejected=true;}",
                'require(rejected,"one element short must fail");',
                "}",
            ]
    return "\n".join([*lines, "}"])


def test_compiled_variants_dispatch_capacity_and_unlowered_reference(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "generated_rccsd_cpu.hpp").write_text(codegen.cpu_header())
    prefix = r"""
#include "generated_rccsd_cpu.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace generativeqc::cc::generated;
static void require(bool ok,const char* reason){if(!ok) throw std::runtime_error(reason);}
static void check_close(const double* actual,const double* expected,std::size_t count){
  for(std::size_t i=0;i<count;++i)
    require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<=2e-11*(1+std::abs(expected[i])),"expanded oracle parity");
}
"""
    checks = []
    for name in ("replay", "lambda_independent_rhs", "lambda_independent_transpose"):
        checks += [
            "for(auto shape: {std::pair<std::size_t,std::size_t>{2,3},{20,8},{20,80},{20,1},{1,100},{100000,1}}){",
            "const auto [o,v]=shape;",
            f' require({name}_arena_elements(o,v)<={name}_strict_arena_elements(o,v),"capacity ceiling");',
            "}",
            f'require({name}_uses_reassociation(20,8),"normal shape uses reassociation");',
        ]
    for name in ("replay", "lambda_independent_transpose"):
        strict_hash = (
            "replay_equation_hash" if name == "replay" else name + "_program_hash"
        )
        checks += [
            f'require(!{name}_uses_reassociation(20,1),"extreme ratio keeps strict");',
            f'require(!{name}_uses_reassociation(100000,1),"optional intermediate overflow falls back");',
            f'require(std::strcmp({name}_selected_program_hash(20,1),{strict_hash})==0,"strict identity");',
            f'require(std::strcmp({name}_selected_program_hash(20,8),{strict_hash})!=0,"reassociated identity");',
        ]
    cases = ((1, 2), (2, 3), (3, 2), (20, 1))
    source = tmp_path / "independent.cpp"
    source.write_text(
        prefix
        + "\n".join(_case(o, v) for o, v in cases)
        + "\nint main(){\n"
        + "\n".join(checks)
        + "\n".join(f"case_{o}_{v}();" for o, v in cases)
        + "\n}\n"
    )
    executable = tmp_path / "independent"
    built = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=60, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
