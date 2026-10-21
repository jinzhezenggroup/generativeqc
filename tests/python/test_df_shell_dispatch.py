"""Exercise the actual lightweight dispatcher without a CUDA device or toolkit.

Only borrowed backend types and device queries are stubbed. Public shell views,
launch declarations, the dispatch ABI, generated policy/registry and production
control flow are compiled from the checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.df_shell_derivatives import SUPPORTED_SHELL_CLASSES
from generativeqc_compiler.integral.df_shell_units import emit_df_shell_units
from generativeqc_compiler.integral.df_tuning.manifest import emit_policy
from generativeqc_compiler.integral.df_tuning.policy import SCHEDULES

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dispatcher(tmp_path_factory: typing.Any, native_cxx: NativeCxx) -> typing.Any:
    directory = tmp_path_factory.mktemp("df_shell_dispatch")
    stub = directory / "scf/cuda"
    stub.mkdir(parents=True)
    (directory / "cuda_runtime.h").write_text("""
#pragma once
#include <cstddef>
#include <cstdint>
using cudaError_t=int;
using cudaStream_t=void*;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1;
enum cudaDeviceAttr {cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor};
int cudaGetDevice(int*);
int cudaDeviceGetAttribute(int*,cudaDeviceAttr,int);
""")
    (stub / "df_derivatives.cuh").write_text("""#pragma once
#include "cuda_runtime.h"
namespace generativeqc::scf {struct DfDerivativeBasisView {};}
""")
    (stub / "df_shell_diagnostics.cuh").write_text("""#pragma once
namespace generativeqc::scf {struct DfShellDiagnostics {};}
""")
    registry = next(emit_df_shell_units())
    (directory / registry[0]).write_text(registry[1])
    (directory / "generated_df_production.hpp").write_text(emit_policy())
    declarations = []
    for a, b, c in SUPPORTED_SHELL_CLASSES:
        declarations.append(
            f"extern const DfShellDispatch df_shell_{a}{b}{c}{{panel<{a},{b},{c}>,group<{a},{b},{c}>,packets<{a},{b},{c}>}};"
        )
    source = directory / "probe.cpp"
    source.write_text(
        r"""
#include <cstdio>
#include <cstdlib>
#include "scf/cuda/df_shell_dispatch.cuh"
static int architecture=120, device_error=0, fail_class=-1, count=0;
static int calls[80][3];
int cudaGetDevice(int* device) {*device=0;return device_error;}
int cudaDeviceGetAttribute(int* value,cudaDeviceAttr kind,int) {
  *value=kind==cudaDevAttrComputeCapabilityMajor?architecture/10:architecture%10;
  return device_error;
}
namespace generativeqc::scf {
template<unsigned A,unsigned B,unsigned C>
int record(const DfShellLaunch& context) {
  constexpr int code=20*A+5*B+C;
  calls[count][0]=code;calls[count][1]=context.variant;calls[count][2]=context.rys;++count;
  return code==fail_class?91:0;
}
template<unsigned A,unsigned B,unsigned C>
int panel(DfShellBasisView,DfShellBasisView,const DfShellLaunch& c) {return record<A,B,C>(c);}
template<unsigned A,unsigned B,unsigned C>
int group(DfShellBasisView,DfShellBasisView,DfShellBasisView,bool,const DfShellLaunch& c) {return record<A,B,C>(c);}
template<unsigned A,unsigned B,unsigned C>
int packets(std::span<const DfShellBasisView>,std::span<const DfShellBasisView>,const DfShellLaunch& c) {return record<A,B,C>(c);}
__DECLARATIONS__
}
int main(int argc,char** argv) {
  if(argc!=8) return 2;
  using namespace generativeqc::scf;
  int kind=std::atoi(argv[1]),full=std::atoi(argv[2]),variant=std::atoi(argv[3]);
  architecture=std::atoi(argv[4]);device_error=std::atoi(argv[5]);fail_class=std::atoi(argv[6]);
  bool empty=std::atoi(argv[7]);
  DfShellBasisView view{};int status;
  if(kind==0) status=launch_df_shell_derivative_panel(view,view,nullptr,0,0,nullptr,nullptr,nullptr,nullptr,full,variant,DfDerivativePairs::full);
  else if(kind==1) status=launch_df_shell_derivative_group(view,view,view,nullptr,0,0,nullptr,nullptr,nullptr,nullptr,full,variant,DfDerivativePairs::full,false);
  else status=launch_df_shell_derivative_packets(std::span(&view,empty?0:1),std::span(&view,1),nullptr,0,0,nullptr,nullptr,nullptr,nullptr,full,variant,DfDerivativePairs::full);
  std::printf("{\"status\":%d,\"calls\":[",status);
  for(int i=0;i<count;++i) std::printf("%s[%d,%d,%d]",i?",":"",calls[i][0],calls[i][1],calls[i][2]);
  std::printf("]}\n");
}
""".replace("__DECLARATIONS__", "\n".join(declarations))
    )
    executable = directory / "probe"
    native_cxx.build_executable(
        [source, ROOT / "src/scf/cuda/df_shell_derivatives.cu"],
        executable,
        compile_args=[
            "-std=c++20",
            "-O0",
            "-x",
            "c++",
            "-I",
            str(directory),
            "-I",
            str(ROOT / "src"),
        ],
        compile_timeout=90,
    )

    def run(
        kind: typing.Any = 0,
        full: typing.Any = True,
        variant: typing.Any = 0,
        architecture: typing.Any = 120,
        error: typing.Any = 0,
        fail: typing.Any = -1,
        empty: typing.Any = False,
        policy: typing.Any = "auto",
        schedule: typing.Any = None,
    ) -> typing.Any:
        env = dict(os.environ, GENERATIVEQC_DF_SHELL_POLICY=policy)
        env.pop("GENERATIVEQC_DF_SHELL_SCHEDULE", None)
        if schedule is not None:
            env["GENERATIVEQC_DF_SHELL_SCHEDULE"] = schedule
        result = subprocess.run(
            [
                str(executable),
                *map(
                    str,
                    (kind, int(full), variant, architecture, error, fail, int(empty)),
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        return json.loads(result.stdout)

    return run


@pytest.mark.parametrize("kind", range(3))
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("variant", range(3))
def test_dispatch_preserves_domains_order_and_explicit_schedules(
    dispatcher: typing.Any, kind: typing.Any, full: typing.Any, variant: typing.Any
) -> None:
    result = dispatcher(kind, full, variant, policy="legacy")
    domain = [
        20 * a + 5 * b + c
        for a, b, c in SUPPORTED_SHELL_CLASSES
        if full or (max(a, b, c) <= 1 and a + b + c > 0)
    ]
    assert result == {"status": 0, "calls": [[code, variant, 0] for code in domain]}


@pytest.mark.parametrize("kind", range(3))
def test_qualified_policy_and_unknown_target_fallback(
    dispatcher: typing.Any, kind: typing.Any
) -> None:
    manifest = json.loads(
        (
            ROOT
            / "python/generativeqc_compiler/integral/production_df_derivatives.json"
        ).read_text()
    )
    choices = {
        20 * int(row["class"][0]) + 5 * int(row["class"][1]) + int(row["class"][2]): row
        for row in manifest["architectures"]["sm_120"]["kernels"]
    }
    codes = [20 * a + 5 * b + c for a, b, c in SUPPORTED_SHELL_CLASSES]
    expected = [
        [
            code,
            SCHEDULES.index(choices[code]["schedule"]) if code in choices else 0,
            int(code in choices and choices[code]["lowering"] == "rys"),
        ]
        for code in codes
    ]
    assert dispatcher(kind)["calls"] == expected
    assert dispatcher(kind, policy="candidate")["calls"] == expected
    assert dispatcher(kind, architecture=80)["calls"] == [
        [code, 0, 0] for code in codes
    ]
    assert dispatcher(kind, schedule="auto")["calls"] == expected
    assert dispatcher(kind, variant=1, schedule="packed")["calls"] == [
        [row[0], 1, row[2]] for row in expected
    ]


@pytest.mark.parametrize("kind", range(3))
def test_errors_stop_dispatch_and_invalid_variant_is_rejected(
    dispatcher: typing.Any, kind: typing.Any
) -> None:
    assert dispatcher(kind, policy="invalid") == {"status": 1, "calls": []}
    assert dispatcher(kind, variant=3) == {"status": 1, "calls": []}
    assert dispatcher(kind, error=7) == {"status": 7, "calls": []}
    failed = dispatcher(kind, fail=21)
    assert failed["status"] == 91
    codes = [20 * a + 5 * b + c for a, b, c in SUPPORTED_SHELL_CLASSES]
    assert [row[0] for row in failed["calls"]] == codes[: codes.index(21) + 1]
    # Auxiliary-g failures stop at the same exact class, without aliasing through-f.
    failed_g = dispatcher(kind, fail=24)
    assert failed_g["status"] == 91
    assert [row[0] for row in failed_g["calls"]] == codes[: codes.index(24) + 1]


def test_empty_packet_keeps_early_return_contract(dispatcher: typing.Any) -> None:
    assert dispatcher(2, empty=True, error=7, policy="invalid") == {
        "status": 0,
        "calls": [],
    }
    assert dispatcher(2, empty=True, variant=3) == {"status": 1, "calls": []}
