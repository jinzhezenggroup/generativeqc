"""Shared invariant proof, pinned native storage and actual CPU solve ownership."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers

from tools import generate_rccsd_native as codegen
from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case

if TYPE_CHECKING:
    from generativeqc_compiler.tensor.program import Program

ROOT = Path(__file__).resolve().parents[2]


def _program(backend: str) -> Program:
    return codegen._prepare_production(codegen.iteration_program(2, 3), backend)


def test_shared_plan_retained_slots_exclude_every_dynamic_write() -> None:
    cpu, cuda = (_program(backend) for backend in ("cpu", "cuda"))
    plans = [codegen._iteration_reuse_plan(program) for program in (cpu, cuda)]
    assert plans[0].identity == plans[1].identity
    assert len(plans[0].invariant_nodes) > 0
    arenas = []
    for program, plan in zip((cpu, cuda), plans):
        nodes = codegen._execution_nodes(program)
        numbers = {node: number for number, node in enumerate(nodes)}
        arena = codegen._arena_plan(program, retained_nodes=plan.invariant_nodes)
        arenas.append(arena)
        pinned = {arena.node_slots[numbers[node]] for node in plan.invariant_nodes}
        assert len(pinned) == len(plan.invariant_nodes)
        assert all(
            arena.node_slots[numbers[node]] not in pinned for node in plan.dynamic_nodes
        )
        dependencies = dict(plan.input_dependencies)
        assert all(
            not {"t1", "t2"}.intersection(dependencies[node])
            for node in plan.invariant_nodes
        )
        # Reference-only transposes are retained even if their original DAG
        # position followed a same-shaped dynamic temporary.
        for o, v in ((1, 1), (1, 7), (7, 1), (2, 3), (20, 80)):
            extents = {"o": o, "v": v}
            sizes = [math.prod(extents[x] for x in shape) for shape in arena.slots]
            assert sum(sizes) >= sum(
                math.prod(extents[x] for x in shape)
                for shape in codegen._arena_plan(program).slots
            )
    assert arenas[0] == arenas[1]
    with pytest.raises(ValueError, match="conventional"):
        codegen._iteration_reuse_plan(
            codegen.iteration_program(2, 3, external_virtual_correction=True)
        )


def test_cuda_phase_launches_use_the_same_proof_and_keep_prepare_errors() -> None:
    program = _program("cuda")
    plan = codegen._iteration_reuse_plan(program)
    nodes = codegen._execution_nodes(program)
    numbers = {node: number for number, node in enumerate(nodes)}
    expected = {
        "prepare": {numbers[node] for node in plan.invariant_nodes},
        "dynamic": {numbers[node] for node in plan.dynamic_nodes},
    }
    for phase, output in (("prepare", "void"), ("dynamic", "DeviceIterationOutputs")):
        emitted = codegen._cuda_program(
            program,
            "iteration_reuse_" + phase,
            output,
            arena_field="iteration_reuse_arena",
            kernel_prefix="iteration",
            emit_kernels=False,
            reuse_plan=plan,
            reuse_phase=phase,
            reset_error=phase == "prepare",
        )
        actual = {int(x) for x in re.findall(r"iteration_node_(\d+)<<<", emitted)}
        assert actual == expected[phase]
        assert emitted.count("<<<") == len(actual)
        assert ("cudaMemsetAsync(s.error" in emitted) == (phase == "prepare")
        assert "s.iteration_reuse_arena" in emitted
    # Generated parity is intentionally not production CUDA promotion.
    owner = (ROOT / "src/cc/cuda_solver.cu").read_text()
    assert "run_iteration_reused_cuda" not in owner


def _array(value: np.ndarray) -> str:
    return "{" + ",".join(float(x).hex() for x in np.asarray(value).ravel()) + "}"


def _physical_case(
    o: int, v: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f, g, t1, t2 = random_case(o, v, 3100 + 10 * o + v)
    # Weakly coupled noncanonical reference gives a stable full native solve.
    f *= 0.04
    f += np.diag(np.r_[np.linspace(-1.2, -0.8, o), np.linspace(0.7, 1.1, v)])
    g *= 0.15
    return f, g, t1, t2


def _case_source(o: int, v: int) -> str:
    f, g, t1, t2 = _physical_case(o, v)
    feeds = dense_feeds(f, g, t1, t2)
    d1 = np.diag(f)[:o, None] - np.diag(f)[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    feeds.update(d1=d1, d2=d2)
    oracle = DeterminantOracle(f, g, o)
    plan = codegen._iteration_reuse_plan(_program("cpu"))
    retained = sum(
        math.prod({"o": o, "v": v}[codegen._dim(i)] for i in node.spec.indices)
        for node in plan.invariant_nodes
    )
    lines = [f"static void case_{o}_{v}(){{", f"Problem p; p.nocc={o};p.nvir={v};"]
    for name, value in feeds.items():
        field = "initial_" + name if name in ("t1", "t2") else name
        lines.append(f"p.{field}={_array(value)};")
    lines += [
        "auto in=bind(p); const auto o=p.nocc,v=p.nvir;",
        "const auto count=iteration_reuse_arena_elements(o,v);",
        "std::vector<double> reused(count+2,1234567),full(iteration_arena_elements(o,v));",
        "run_iteration_reuse_prepare_cpu(o,v,in,reused.data()+1,count);",
        f"const std::vector<double> pinned(reused.data()+1,reused.data()+1+{retained});",
    ]
    for scale in (1.0, 0.6, -0.4):
        x, y = scale * t1, scale * t2
        e, r1, r2 = oracle.evaluate_full(x, y)
        lines += [
            "{",
            f"const double x[]={_array(x)},y[]={_array(y)};in.t1=x;in.t2=y;",
            f"const double energy[]={_array(np.asarray(e))},r1[]={_array(r1)},r2[]={_array(r2)};",
            f"const double next1[]={_array(x + r1 / d1)},next2[]={_array(y + r2 / d2)};",
            "const auto a=run_iteration_reused_cpu(o,v,in,reused.data()+1,count);",
            "const auto b=run_iteration_cpu(o,v,in,full.data(),full.size());",
            "same(a,b,o*v,o*o*v*v);",
            "close(&a.energy,energy,1);close(a.r1,r1,o*v);close(a.r2,r2,o*o*v*v);",
            "close(a.next_t1,next1,o*v);close(a.next_t2,next2,o*o*v*v);",
            'require(std::equal(pinned.begin(),pinned.end(),reused.data()+1),"retained values were overwritten");',
            'require(reused.front()==1234567 && reused.back()==1234567,"pinned arena canary");',
            "}",
        ]
    lines += [
        "bool short_arena=false;",
        "try{run_iteration_reuse_prepare_cpu(o,v,in,reused.data()+1,count-1);}",
        "catch(const std::length_error&){short_arena=true;}",
        'require(short_arena,"prepare must reject short pinned arena");',
        "short_arena=false;",
        "try{run_iteration_reused_cpu(o,v,in,reused.data()+1,count-1);}",
        "catch(const std::length_error&){short_arena=true;}",
        'require(short_arena,"dynamic must reject short pinned arena");',
        # The owner never receives these bad inputs: validate_problem rejects
        # them. Exercise the generated preparation failure boundary directly.
        "in=bind(p); auto invalid=p.foo;invalid[0]=std::numeric_limits<double>::quiet_NaN();in.foo=invalid.data();",
        "bool published=false;try{run_iteration_reuse_prepare_cpu(o,v,in,reused.data()+1,count);published=true;}",
        "catch(const std::runtime_error&){}",
        'require(!published,"failed prepare must not publish success");',
        "in=bind(p);run_iteration_reuse_prepare_cpu(o,v,in,reused.data()+1,count);",
        "const auto retried=run_iteration_reused_cpu(o,v,in,reused.data()+1,count);",
        "const auto fresh=run_iteration_cpu(o,v,in,full.data(),full.size());same(retried,fresh,o*v,o*o*v*v);",
        "std::fill(p.initial_t1.begin(),p.initial_t1.end(),0);std::fill(p.initial_t2.begin(),p.initial_t2.end(),0);",
        "check_owner(p);",
        "}",
    ]
    return "\n".join(lines)


PREFIX = r"""
#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <vector>
using namespace generativeqc::cc;
using namespace generativeqc::cc::generated;
static std::size_t fail_allocation_bytes{};
static bool failed_allocation{};
void* operator new(std::size_t count){
 if(fail_allocation_bytes&&count==fail_allocation_bytes&&!failed_allocation){failed_allocation=true;throw std::bad_alloc();}
 if(void* p=std::malloc(count?count:1))return p;throw std::bad_alloc();
}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void close(const double* a,const double* b,std::size_t n){
  for(std::size_t i=0;i<n;++i)require(std::isfinite(a[i])&&std::abs(a[i]-b[i])<2e-11*(1+std::abs(b[i])),"determinant oracle parity");
}
static void same(IterationOutputs a,IterationOutputs b,std::size_t n1,std::size_t n2){
 require(a.energy==b.energy,"energy changed");
 require(!std::memcmp(a.r1,b.r1,n1*sizeof(double)),"R1 changed");
 require(!std::memcmp(a.r2,b.r2,n2*sizeof(double)),"R2 changed");
 require(!std::memcmp(a.next_t1,b.next_t1,n1*sizeof(double)),"trial T1 changed");
 require(!std::memcmp(a.next_t2,b.next_t2,n2*sizeof(double)),"trial T2 changed");
}
static Inputs bind(const Problem& p){
 return {p.foo.data(),p.fov.data(),p.fvv.data(),p.ovov.data(),p.ovvo.data(),p.oovv.data(),p.ovvv.data(),p.ovoo.data(),p.oooo.data(),p.vvvv.data(),p.d1.data(),p.d2.data(),p.initial_t1.data(),p.initial_t2.data()};
}
static std::size_t budget(const Problem& p,bool reuse,std::size_t history=0){
 const auto iteration=reuse?iteration_reuse_arena_elements(p.nocc,p.nvir):iteration_arena_elements(p.nocc,p.nvir);
 const auto elements=p.initial_t1.size()+p.initial_t2.size();
 const auto n=history+1,scratch=history?history*history+2*n*n+2*n:0;
 const auto host_vectors=history?4+2*history:2;
 return p.reference_retained_bytes+problem_host_bytes(p)+sizeof(double)*(iteration+replay_arena_elements(p.nocc,p.nvir)+host_vectors*elements+scratch);
}
static void print_result(const Problem& p,const SolverResult& r){
 std::cout<<std::setprecision(17)<<"{\"o\":"<<p.nocc<<",\"v\":"<<p.nvir<<",\"energy\":"<<r.correlation_energy<<",\"t1\":[";
 for(std::size_t i=0;i<r.t1.size();++i)std::cout<<(i?",":"")<<r.t1[i];
 std::cout<<"],\"t2\":[";for(std::size_t i=0;i<r.t2.size();++i)std::cout<<(i?",":"")<<r.t2[i];std::cout<<"]}\n";
}
static void check_owner(Problem p){
 SolverOptions options;options.diis_size=0;options.max_iterations=150;options.max_bytes=budget(p,true);
 const auto default_result=solve_cpu(p,options);require(!default_result.diagnostic.iteration_reuse,"unqualified schedule must remain opt-in");
 options.iteration_invariant_reuse=true;
 const auto reused=solve_cpu(p,options);const auto& d=reused.diagnostic;
 require(reused.converged(),"reused solve did not converge");
 require(d.iteration_reuse&&d.iteration_invariant_preparations==1,"one preparation per solve");
 require(d.iteration_graph_calls>2&&d.iteration_reused_evaluations==d.iteration_graph_calls,"current evaluation reuse");
 require(d.iteration_graph_calls==d.iterations&&d.update_calls+1==d.iterations,"no-DIIS evaluates once per iteration");
 require(d.iteration_invariant_operations==iteration_invariant_operation_count,"prepare work count");
 require(d.iteration_dynamic_operations==d.iteration_graph_calls*iteration_dynamic_operation_count,"dynamic work count");
 require(d.iteration_invariant_operations_saved==(d.iteration_graph_calls-1)*iteration_invariant_operation_count,"net saved work count");
 require(d.replay_graph_calls==1,"independent full convergence replay");
 require(d.numeric_capacity_bytes==options.max_bytes,"complete retained arena budget");
 print_result(p,reused);
 fail_allocation_bytes=sizeof(double)*iteration_reuse_arena_elements(p.nocc,p.nvir);failed_allocation=false;
 const auto allocation_fallback=solve_cpu(p,options);fail_allocation_bytes=0;
 require(failed_allocation&&!allocation_fallback.diagnostic.iteration_reuse,"optional allocation failure fallback");
 require(allocation_fallback.diagnostic.numeric_capacity_bytes==budget(p,false),"allocation fallback accounting");
 require(allocation_fallback.t1==reused.t1&&allocation_fallback.t2==reused.t2,"allocation fallback final-state parity");
 for(const auto limit:{budget(p,false),budget(p,true)-1}){
  options.max_bytes=limit;const auto fallback=solve_cpu(p,options);
  require(fallback.converged()&&!fallback.diagnostic.iteration_reuse,"budget fallback");
  require(fallback.diagnostic.iteration_invariant_preparations==0&&fallback.diagnostic.iteration_reused_evaluations==0,"fallback must not prepare");
  require(fallback.diagnostic.iteration_invariant_operations==fallback.diagnostic.iteration_graph_calls*iteration_invariant_operation_count,"uncached invariant work");
  require(fallback.diagnostic.iteration_dynamic_operations==fallback.diagnostic.iteration_graph_calls*iteration_dynamic_operation_count,"uncached dynamic work");
  require(fallback.correlation_energy==reused.correlation_energy&&fallback.t1==reused.t1&&fallback.t2==reused.t2,"full solver final-state parity");
  require(fallback.diagnostic.iterations==d.iterations&&fallback.diagnostic.iteration_graph_calls==d.iteration_graph_calls&&fallback.diagnostic.replay_graph_calls==d.replay_graph_calls,"same convergence/evaluation/replay path");
 }
 options.max_bytes=budget(p,false)-1;bool refused=false;
 try{solve_cpu(p,options);}catch(const std::length_error&){refused=true;}require(refused,"one-byte-short complete budget");
 // Same dimensions and storage addresses, changed reference values: a new
 // solve still prepares afresh. This also models geometry/basis/source changes.
 p.fov[0]+=0.001;
 options.max_bytes=budget(p,true);const auto changed=solve_cpu(p,options);
 require(changed.converged()&&changed.diagnostic.iteration_invariant_preparations==1,"new reference needs new prepare");
 options.max_bytes=budget(p,false);const auto changed_full=solve_cpu(p,options);
 require(changed.t1==changed_full.t1&&changed.t2==changed_full.t2&&changed.correlation_energy==changed_full.correlation_energy,"changed-reference fallback parity");
 require(changed.correlation_energy!=reused.correlation_energy,"changed source must affect solution");
 // DIIS replaces the dynamic amplitude vector while the immutable reference
 // epoch remains valid, including nontrivial history and trial evaluations.
 options.diis_size=6;options.max_bytes=budget(p,true,6);const auto diis_reused=solve_cpu(p,options);
 options.max_bytes=budget(p,false,6);const auto diis_full=solve_cpu(p,options);
 require(diis_reused.converged()&&diis_full.converged(),"DIIS convergence");
 require(diis_reused.diagnostic.iteration_invariant_preparations==1&&!diis_full.diagnostic.iteration_reuse,"DIIS reference lifetime");
 require(diis_reused.t1==diis_full.t1&&diis_reused.t2==diis_full.t2&&diis_reused.correlation_energy==diis_full.correlation_energy,"DIIS final-state parity");
 const auto& diis_d=diis_reused.diagnostic;
 require(diis_d.iteration_graph_calls==diis_full.diagnostic.iteration_graph_calls&&diis_d.iterations==diis_full.diagnostic.iterations,"same DIIS carried-output decisions");
 require(diis_d.iteration_graph_calls<2*diis_d.iterations-1,"DIIS must reuse its first accepted trial output");
 require(diis_d.iteration_reused_evaluations==diis_d.iteration_graph_calls&&diis_d.iteration_dynamic_operations==diis_d.iteration_graph_calls*iteration_dynamic_operation_count,"carried output must not count as a new graph evaluation");
 require(diis_d.iteration_invariant_operations_saved==(diis_d.iteration_graph_calls-1)*iteration_invariant_operation_count,"saved operations use actual graph calls");
 // The canonical denominator view carries an immutable spectrum/level shift
 // outside the logical d2 pointer; changing amplitudes must still reconstruct
 // each dynamic Jacobi denominator without storing a stale trial.
 auto canonical=p;std::vector<double> eps;
 for(std::size_t i=0;i<p.nocc;++i)eps.push_back(p.foo[i*p.nocc+i]);
 for(std::size_t a=0;a<p.nvir;++a)eps.push_back(p.fvv[a*p.nvir+a]);
 initialize_canonical_denominators(canonical,eps,options,true);
 options.max_bytes=budget(canonical,true,6);const auto derived_reused=solve_cpu(canonical,options);
 options.max_bytes=budget(canonical,false,6);const auto derived_full=solve_cpu(canonical,options);
 require(derived_reused.converged()&&derived_full.converged(),"canonical reuse convergence");
 require(derived_reused.diagnostic.iteration_invariant_preparations==1&&!derived_full.diagnostic.iteration_reuse,"canonical reference lifetime");
 require(derived_reused.diagnostic.derived_d2_iteration_evaluations==derived_reused.diagnostic.iteration_graph_calls*p.initial_t2.size(),"canonical dynamic Jacobi work");
 require(derived_reused.t1==derived_full.t1&&derived_reused.t2==derived_full.t2&&derived_reused.correlation_energy==derived_full.correlation_energy,"canonical final-state parity");
 require(derived_reused.diagnostic.iteration_graph_calls==derived_full.diagnostic.iteration_graph_calls,"canonical carried-output evaluation parity");
 // A zero-residual first trial is accepted unchanged. The next iteration
 // consumes its borrowed arena output without a graph call or preparation.
 auto unchanged=p;
 for(auto* values:{&unchanged.fov,&unchanged.ovov,&unchanged.ovvo,&unchanged.oovv,&unchanged.ovvv,&unchanged.ovoo,&unchanged.oooo,&unchanged.vvvv,&unchanged.initial_t1,&unchanged.initial_t2})
  std::fill(values->begin(),values->end(),0);
 options.max_bytes=budget(unchanged,true,6);const auto carried=solve_cpu(unchanged,options);
 const auto& carried_d=carried.diagnostic;
 require(carried.converged()&&carried_d.iterations==2&&carried_d.iteration_graph_calls==2&&carried_d.update_calls==1,"unchanged DIIS trial consumed once");
 require(carried_d.iteration_invariant_preparations==1&&carried_d.iteration_reused_evaluations==2&&carried_d.replay_graph_calls==1,"carried arena output preserves reference cache and replay");
 require(carried_d.iteration_invariant_operations_saved==iteration_invariant_operation_count&&carried_d.iteration_dynamic_operations==2*iteration_dynamic_operation_count,"carried output has zero additional generated work");
}
"""


def test_native_cpu_owner_matches_independent_oracle_and_budget_fallback(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    write_df_cpu_headers(tmp_path)
    cases = ((1, 2), (2, 2), (2, 3))
    source = tmp_path / "reuse.cpp"
    source.write_text(
        PREFIX
        + "\n".join(_case_source(o, v) for o, v in cases)
        + "\nint main(){"
        + "".join(f"case_{o}_{v}();" for o, v in cases)
        + "}\n"
    )
    output = tmp_path / "reuse"
    compile_owner(compiler, tmp_path, [ROOT / "src/cc/solver.cpp", source], output)
    run = subprocess.run(
        [str(output)], capture_output=True, text=True, timeout=60, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(rows) == len(cases)
    for row in rows:
        o, v = row["o"], row["v"]
        f, g, _, _ = _physical_case(o, v)
        t1 = np.asarray(row["t1"]).reshape(o, v)
        t2 = np.asarray(row["t2"]).reshape(o, o, v, v)
        energy, r1, r2 = DeterminantOracle(f, g, o).evaluate_full(t1, t2)
        assert abs(energy - row["energy"]) < 2e-11
        assert max(np.max(np.abs(r1)), np.max(np.abs(r2))) < 1e-9
