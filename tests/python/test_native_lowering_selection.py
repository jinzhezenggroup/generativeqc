"""Cross-language gates for canonical portfolio selection and native metadata."""

from __future__ import annotations

import shutil
import subprocess
import typing
from dataclasses import replace

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.common.lowering_contract import LoweringConstraints
from generativeqc_compiler.common.lowering_selection import select_lowering_binding
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from test_joint_lowering import COMPILATION, LIBRARY, TARGET, _cost, _offer, _request

if typing.TYPE_CHECKING:
    from pathlib import Path


def test_native_selector_matches_canonical_compiler(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    request = _request()
    strict = _offer(request, cost=_cost(request.precisions[0], kernel=100))
    mixed = _offer(
        request,
        LIBRARY,
        dtype="float32",
        cost=_cost(request.precisions[1], kernel=20, cast=15, audit=10, prepare=1000),
    )
    cases = [(request, (strict, mixed), 1), (request, (strict, mixed), 50)]
    cases.append(
        (request, (replace(strict, implementation="strict-other"), strict, mixed), 1)
    )
    cases.append((request, (strict, replace(mixed, cost=None)), 50))
    cases.append((request, (strict, replace(mixed, target=None)), 50))
    cases.append(
        (
            request,
            (
                strict,
                replace(mixed, status="unsupported", reason="unsupported arithmetic"),
            ),
            50,
        )
    )
    # Include every candidate's simultaneous resource ownership, not just the
    # provider workspace; exact-order capture cannot admit unordered execution.
    limited = replace(
        request, constraints=LoweringConstraints(additional_device_bytes=15)
    )
    cases.append(
        (
            limited,
            (
                replace(strict, request=limited),
                replace(
                    mixed,
                    request=limited,
                    provider_bytes=8,
                    execution=replace(mixed.execution, temporary_bytes=8),
                ),
            ),
            50,
        )
    )
    constrained = replace(
        request,
        constraints=LoweringConstraints(
            capture_required=True, determinism="exact-order"
        ),
    )
    cases.append(
        (
            constrained,
            (
                replace(
                    strict,
                    request=constrained,
                    execution=replace(
                        strict.execution, determinism="exact-order", capture_safe=True
                    ),
                ),
                replace(mixed, request=constrained),
            ),
            50,
        )
    )
    cases_with_incumbent = [(*case, None) for case in cases]
    unmeasured = (replace(strict, cost=None), replace(mixed, cost=None))
    cases_with_incumbent.extend(
        [
            (request, unmeasured, 50, 1),
            (request, (unmeasured[0], mixed), 50, 0),
            (request, (strict, mixed), 50, 0),
        ]
    )
    declarations, checks = [], []
    for i, (operation, candidates, replays, incumbent) in enumerate(
        cases_with_incumbent
    ):
        name = f"portfolio{i}"
        expected = select_lowering_binding(
            operation,
            TARGET,
            COMPILATION,
            candidates,
            expected_replays=replays,
            qualified_incumbent=candidates[incumbent].identity
            if incumbent is not None
            else None,
        )
        declarations.append(
            native_lowering_portfolio(
                operation, candidates, TARGET, COMPILATION, name=name
            )
        )
        checks += [
            "{",
            f"const auto decision=select_native_lowering({name}_request,{name}_candidates,{name}_target,{name}_compilation,{replays},{incumbent if incumbent is not None else 'std::nullopt'});",
            f'if({name}_candidates[decision.selected].identity!="{expected.selected.identity}") return {10 + i};',
            f"if(decision.retained_incumbent != {str(expected.retained_incumbent).lower()}) return {60 + i};",
            f"if(decision.fallbacks.size()!={len(expected.fallbacks)} || decision.rejections.size()!={len(expected.rejections)}) return {20 + i};",
            *(
                f'if({name}_candidates[decision.fallbacks[{j}]].identity!="{candidate.identity}") return {30 + i};'
                for j, candidate in enumerate(expected.fallbacks)
            ),
            "}",
        ]
    checks += [
        "auto reject=[](const auto& action){try{action();}catch(const std::exception&){return true;}return false;};",
        "if(!reject([&]{select_native_lowering(portfolio0_request,portfolio0_candidates,portfolio0_target,portfolio0_compilation,0);})) return 50;",
        'if(!reject([&]{select_native_lowering(portfolio0_request,portfolio0_candidates,"stale-target",portfolio0_compilation);} )) return 51;',
        "auto duplicate=portfolio0_candidates; duplicate[1]=duplicate[0];",
        "if(!reject([&]{select_native_lowering(portfolio0_request,duplicate,portfolio0_target,portfolio0_compilation);} )) return 52;",
        "auto unknown=portfolio0_candidates; unknown[0].cost.audit_ns=std::nullopt;",
        "if(!reject([&]{select_native_lowering(portfolio0_request,unknown,portfolio0_target,portfolio0_compilation);} )) return 53;",
        "auto overflow=portfolio0_candidates; overflow[0].cost.kernel_ns=UINT64_MAX;",
        "if(!reject([&]{select_native_lowering(portfolio0_request,overflow,portfolio0_target,portfolio0_compilation,2);} )) return 54;",
        "auto stale=portfolio0_candidates; stale[0].cost={}; stale[0].capabilities_available=false;",
        "if(!reject([&]{select_native_lowering(portfolio0_request,stale,portfolio0_target,portfolio0_compilation,1,0);} )) return 55;",
        "auto limited=portfolio0_request; limited.constraints.provider_bytes=0;",
        "stale=portfolio0_candidates; stale[0].cost={}; stale[0].provider_bytes=1;",
        "if(!reject([&]{select_native_lowering(limited,stale,portfolio0_target,portfolio0_compilation,1,0);} )) return 56;",
        "auto bad_precision=portfolio0_precisions; bad_precision[1].identity=bad_precision[0].identity;",
        "auto invalid=portfolio0_request; invalid.precisions=bad_precision;",
        "if(!reject([&]{select_native_lowering(invalid,portfolio0_candidates,portfolio0_target,portfolio0_compilation);} )) return 57;",
        "bad_precision=portfolio0_precisions; for(auto& p:bad_precision) if(!strict_requested_precision(invalid,p)) p.arithmetic.qualification={};",
        "if(!reject([&]{select_native_lowering(invalid,portfolio0_candidates,portfolio0_target,portfolio0_compilation);} )) return 58;",
        "invalid=portfolio0_request; invalid.inputs=5;",
        "if(!reject([&]{select_native_lowering(invalid,portfolio0_candidates,portfolio0_target,portfolio0_compilation);} )) return 59;",
    ]
    source = tmp_path / "selector.cpp"
    source.write_text(
        '#include "runtime/lowering_binding.hpp"\n'
        "using namespace generativeqc::runtime;\n"
        + "\n".join(declarations)
        + "\nint main(){\n"
        + "\n".join(checks)
        + "\n}\n"
    )
    binary = tmp_path / "selector"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
