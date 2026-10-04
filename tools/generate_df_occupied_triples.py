"""Emit bounded DF panels, W GEMMs and a fused occupied-tile (T) epilogue."""

from __future__ import annotations

import argparse
import sys
import typing
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.occupied_triples import (
    PERMUTATIONS,
    df_panel_program,
    energy_scalar_program,
    inverse,
    moment_program,
    v_scalar_program,
    w_fp32_candidate_program,
)
from generativeqc_compiler.cc.occupied_triples_fock import (
    moment_program as fock_moment_program,
)
from generativeqc_compiler.cc.occupied_triples_fock import (
    resolvent_scalar_program,
)
from generativeqc_compiler.cc.occupied_triples_response import (
    energy_scalar_vjp,
    gap_vjp,
    moment_vjp,
    panel_vjp,
    scaled_denominator_vjp,
)
from generativeqc_compiler.cc.triples import _LABELS, VP
from generativeqc_compiler.tensor import describe_precision
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

from tools.generate_rccsd_native import _cuda_program, _required_function

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node, Program


def _inputs(program: Program) -> tuple[str, ...]:
    return tuple(sorted(n.attrs["name"] for n in program.live_nodes if n.op == "input"))


def _call(program: Program, name: str, bindings: dict[str, str], output: str) -> str:
    names = _inputs(program)
    if set(names) != set(bindings):
        raise ValueError("scalar boundary bindings do not match the equation")
    return name + "(" + ",".join([*(bindings[key] for key in names), output]) + ");"


def _gemm(
    node: Node,
    bindings: dict[str, tuple[str, str | None]],
    alpha: str,
    beta: str,
    *,
    output_pointer: str = "output",
    output_leading_dimension: str | None = None,
) -> str:
    """Derive a column-major call from a TensorIR product and strided views.

    Views supply the address and physical leading dimension of each logical
    row-major input. Operand permutation, transposition, contraction dimensions
    and output layout come exclusively from the shared GEMM contract. A None
    leading dimension denotes a contiguous tensor; derive its matrix cut from
    this contraction, since one cube can be [ab,c] in W1 and [a,bc] in W2.
    """
    g = gemm_contract(node)
    if (
        g is None
        or g.batch_labels
        or g.coefficient != 1
        or g.dtype not in ("float32", "float64")
    ):
        raise ValueError("occupied triples require unbatched unit real products")
    a, b, c, m, n, k = (
        g.a_labels,
        g.b_labels,
        g.output_labels,
        g.m_labels,
        g.n_labels,
        g.k_labels,
    )
    # AD represents a signed seed as a one-input add. Fold only this exact
    # scalar multiplier into BLAS alpha, preserving all other input topology.
    scale = Fraction(1)
    operands = []
    for raw_operand in node.inputs:
        operand = raw_operand
        while operand.op == "add" and len(operand.inputs) == 1:
            scale *= Fraction(*operand.attrs["coefficients"][0])
            operand = operand.inputs[0]
        while operand.op == "cast":
            if operand.attrs["dtype"] != g.dtype:
                raise ValueError("occupied triples cast does not match GEMM precision")
            operand = operand.inputs[0]
        if operand.op != "input":
            raise ValueError(
                "occupied triples product needs a packed input or scaled seed"
            )
        operands.append(bindings[operand.attrs["name"]])
    aa, bb = operands
    if scale != 1:
        alpha = f"({alpha})*({scale.numerator}.0/{scale.denominator}.0)"
    if c != m + n:
        a, b, m, n, aa, bb = b, a, n, m, bb, aa
    if c != m + n:
        raise ValueError("occupied triples output needs an unqualified packing")
    dims = {
        label: {"occupied": "o", "virtual": "v", "auxiliary": "q", "batch": "capacity"}[
            index.space.kind
        ]
        for operand, labels in zip(node.inputs, node.attrs["labels"], strict=True)
        for index, label in zip(operand.spec.indices, labels, strict=True)
    }

    def extent(labels: tuple[int, ...]) -> str:
        return "checked_product({" + ",".join(dims[x] for x in labels) + "})"

    def trans(
        actual: tuple[int, ...], rows: tuple[int, ...], columns: tuple[int, ...]
    ) -> str:
        if actual == rows + columns:
            return "N"
        if actual == columns + rows:
            return "T"
        raise ValueError("occupied triples input needs an unqualified packing")

    lda = aa[1] or extent(k if a == m + k else m)
    ldb = bb[1] or extent(n if b == k + n else k)
    # Transpose the entire row-major product, reversing the two operands.
    return (
        f"gemm('{trans(b, k, n)}','{trans(a, m, k)}',"
        f"{extent(n)},{extent(m)},{extent(k)},{alpha},"
        f"{bb[0]},{ldb},{aa[0]},{lda},{beta},{output_pointer},{output_leading_dimension or extent(n)});"
    )


def response_blas_header() -> str:
    """Emit local reverse products on explicit physical strided views.

    W and panel outputs directly accumulate into their original input slices.
    V's ovov_ij is gathered before use because flattening its a/b axes would
    hide a physical stride; bar_ovov_ij is a bounded packed output for scatter.
    """
    fields = ("bov", "bvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
    lines: list[str] = [
        "struct ResponseOutputs { "
        + "; ".join("double* " + x + "{}" for x in fields)
        + "; };",
    ]
    program = panel_vjp(3, 4)
    lines += [
        "template<class Gemm> void pullback_panel(std::size_t o,std::size_t v,std::size_t q,",
        "  std::size_t i,const Inputs& in,const double* bar_panel,ResponseOutputs& out,Gemm&& gemm) {",
    ]
    bindings: dict[str, tuple[str, str | None]] = {
        "bar_panel": ("bar_panel", None),
        "bov_i": ("in.bov+i*v", "o*v"),
        "bvv": ("in.bvv", None),
    }
    for name, pointer, stride in (
        ("bar_bov_i", "out.bov+i*v", "o*v"),
        ("bar_bvv", "out.bvv", "v*v"),
    ):
        lines.append(
            _gemm(
                program.outputs[name],
                bindings,
                "1.0",
                "1.0",
                output_pointer=pointer,
                output_leading_dimension=stride,
            )
        )
    lines.append("}")
    program = moment_vjp(2, 3, "w")
    lines += [
        "template<class Gemm> void pullback_w(std::size_t o,std::size_t v,",
        "  std::size_t i,std::size_t j,std::size_t k,const Inputs& in,const double* panel,",
        "  const double* bar_w,double* bar_panel,ResponseOutputs& out,Gemm&& gemm) {",
    ]
    bindings = {
        "bar_w": ("bar_w", None),
        "panel": ("panel", None),
        "t2_kj": ("in.t2+(k*o+j)*v*v", None),
        "ovoo_ij": ("in.ovoo+(i*v*o+j)*o", "o*o"),
        "t2_mk": ("in.t2+k*v*v", "o*v*v"),
    }
    for name, pointer, stride in (
        ("bar_panel", "bar_panel", None),
        ("bar_t2_kj", "out.t2+(k*o+j)*v*v", "v"),
        ("bar_ovoo_ij", "out.ovoo+(i*v*o+j)*o", "o*o"),
        ("bar_t2_mk", "out.t2+k*v*v", "o*v*v"),
    ):
        lines.append(
            _gemm(
                program.outputs[name],
                bindings,
                "1.0",
                "1.0",
                output_pointer=pointer,
                output_leading_dimension=stride,
            )
        )
    lines.append("}")
    program = moment_vjp(2, 3, "v")
    lines += [
        "template<class Gemm> void pullback_v(std::size_t o,std::size_t v,",
        "  std::size_t i,std::size_t j,std::size_t k,const Inputs& in,const double* ovov_ij,",
        "  const double* bar_v,double* bar_ovov_ij,ResponseOutputs& out,Gemm&& gemm) {",
    ]
    bindings = {
        "bar_v": ("bar_v", None),
        "ovov_ij": ("ovov_ij", None),
        "t1_k": ("in.t1+k*v", None),
        "t2_ij": ("in.t2+(i*o+j)*v*v", None),
        "fov_k": ("in.fov+k*v", None),
    }
    for name, pointer in (
        ("bar_t1_k", "out.t1+k*v"),
        ("bar_fov_k", "out.fov+k*v"),
        ("bar_t2_ij", "out.t2+(i*o+j)*v*v"),
        ("bar_ovov_ij", "bar_ovov_ij"),
    ):
        lines.append(
            _gemm(
                program.outputs[name],
                bindings,
                "1.0",
                "0.0" if name == "bar_ovov_ij" else "1.0",
                output_pointer=pointer,
                output_leading_dimension="1",
            )
        )
    lines.append("}")
    return "\n".join(lines)


def _response_scalars() -> dict[str, Program]:
    """One requested derivative per scalar call lets the compiler prune work."""
    names = _inputs(energy_scalar_program())
    return {name: energy_scalar_vjp((name,)) for name in names}


def response_scalar_header() -> str:
    lines = []
    for name, program in _response_scalars().items():
        lines.append(
            emit_scalar_cpp(
                program, function_name="pullback_" + name, ordered_native_sums=True
            )
            # Clang/CuMetal annotates the global CUDA math overloads, whereas
            # libc++'s std wrappers can remain host-only. Keep the robust
            # scaled-bilinear algorithm and bind its calls to those overloads.
            .replace("std::frexp(", "::frexp(")
            .replace("std::scalbn(", "::scalbn(")
            .replace("std::fma(", "::fma(")
            .replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool ")
        )
    lines.append(
        emit_scalar_cpp(
            scaled_denominator_vjp(),
            function_name="pullback_scaled_gap",
            ordered_native_sums=True,
        ).replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool ")
    )
    program = gap_vjp(3)
    lines += [
        "struct GapOutputs { const double *bar_eps_i{},*bar_eps_j{},*bar_eps_k{},*bar_eps_v{}; };",
        _required_function(program, "gap_response_arena_elements"),
        f"inline constexpr std::size_t gap_response_operations={sum(n.op != 'input' for n in program.live_nodes)};",
    ]
    return "\n".join(lines)


def _response_point(program: Program, name: str, result: str) -> list[str]:
    """Bind a derivative's live inputs to this exact energy point's coordinates."""
    bindings = {}
    lines = [
        "const std::size_t occupied[3]={i,j,k};",
        "const double gap=in.eps_o[i]+in.eps_o[j]+in.eps_o[k]-in.eps_v[a]-in.eps_v[b]-in.eps_v[c];",
        "const double denominator=gap*degeneracy;",
        "if(!isfinite(gap)||gap>=0.0||fabs(gap)<=threshold||!isfinite(denominator)){atomicCAS(error,0,1);}",
    ]
    for key in _inputs(program):
        if key == "bar_energy":
            bindings[key] = "1.0"
        elif key == "denominator":
            bindings[key] = "denominator"
        elif key.startswith("w_"):
            _, occ, vir = key.split("_")
            index = _LABELS.index(occ)
            x, y, z = ("abc"[axis] for axis in VP[vir])
            bindings[key] = f"moments[{index}*v3+({x}*v+{y})*v+{z}]"
        elif key.startswith("v_"):
            occ = key[2:]
            I, J, K = (f"occupied[{axis}]" for axis in VP[occ])
            lines += [
                f"double {key}=0.0;",
                _call(
                    v_scalar_program(),
                    "v_element",
                    {
                        "ovov": f"in.ovov[((({I})*v+a)*o+({J}))*v+b]",
                        "t1": f"in.t1[({K})*v+c]",
                        "t2": f"in.t2[((({I})*o+({J}))*v+a)*v+b]",
                        "fov": f"in.fov[({K})*v+c]",
                    },
                    key,
                ),
            ]
            bindings[key] = key
        else:
            raise ValueError("unknown occupied scalar response input")
    call = _call(program, "pullback_" + name, bindings, result).removesuffix(";")
    lines.append(f"if(!{call}) atomicCAS(error,0,2);")
    return lines


def response_cuda_source() -> str:
    programs = _response_scalars()
    lines = [
        "__global__ void response_w_kernel(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double degeneracy,double threshold,Inputs in,const double* moments,double* bar_w,int* error){",
        " const auto v3=v*v*v;",
        " for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<6*v3;flat+=std::size_t(blockDim.x)*gridDim.x){",
        "  const auto cube=flat/v3, local=flat%v3; const std::size_t abc[3]={local/(v*v),(local/v)%v,local%v};",
        "  double sum=0.0; switch(cube){",
    ]
    for index, occ in enumerate(_LABELS):
        lines.append(f"case {index}: {{")
        for vir in _LABELS:
            a, b, c = inverse(VP[vir])
            name = f"w_{occ}_{vir}"
            lines += [
                f"{{ const std::size_t a=abc[{a}],b=abc[{b}],c=abc[{c}]; double value=0.0;"
            ]
            lines += _response_point(programs[name], name, "value")
            lines += ["sum+=value; }"]
        lines.append("break; }")
    lines += [
        "} bar_w[flat]=generativeqc_tensor::finite(sum,error,3); } }",
        "__global__ void response_v_kernel(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " unsigned permutation,double degeneracy,double threshold,Inputs in,const double* moments,double* bar_v,int* error){",
        " const auto v3=v*v*v;",
        " for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x){",
        " const std::size_t a=flat/(v*v),b=(flat/v)%v,c=flat%v; double value=0.0; switch(permutation){",
    ]
    for index, occ in enumerate(_LABELS):
        name = "v_" + occ
        lines += [
            f"case {index}: {{",
            *_response_point(programs[name], name, "value"),
            "break; }",
        ]
    lines += [
        "} bar_v[flat]=generativeqc_tensor::finite(value,error,4); } }",
        "__global__ void response_gap_kernel(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double degeneracy,double threshold,Inputs in,const double* moments,double* bar_gap,int* error){",
        " const auto v3=v*v*v;",
        " for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x){",
        " const std::size_t a=flat/(v*v),b=(flat/v)%v,c=flat%v; double value=0.0, scaled=0.0;",
        *_response_point(programs["denominator"], "denominator", "value"),
    ]
    call = _call(
        scaled_denominator_vjp(),
        "pullback_scaled_gap",
        {"bar_denominator": "value", "multiplicity": "degeneracy"},
        "scaled",
    ).removesuffix(";")
    lines += [
        f"if(!{call}) atomicCAS(error,0,5);",
        "bar_gap[flat]=generativeqc_tensor::finite(scaled,error,6); } }",
    ]
    for key in ("w", "v", "gap"):
        extra = "unsigned permutation," if key == "v" else ""
        arg = "permutation," if key == "v" else ""
        lines += [
            f"void response_{key}_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
            f" {extra}double degeneracy,double threshold,const Inputs& in,const double* moments,unsigned blocks,double* output,int* error,cudaStream_t stream){{",
            f"response_{key}_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,{arg}degeneracy,threshold,in,moments,output,error);",
            "generativeqc_tensor::cuda_check(cudaGetLastError()); }",
        ]
    program = gap_vjp(3)
    lines += [
        _cuda_program(
            program,
            "gap_response",
            "GapOutputs",
            state_type="GapCudaState",
            input_overrides={"bar_gap": "s.bar_gap"},
            output_fields=tuple(program.outputs),
            reset_error=False,
        ),
        "GapOutputs gap_response_cuda(GapCudaState& s){return run_gap_response(s);}",
    ]
    return "\n".join(lines)


def fock_header() -> str:
    """Lower occupied-resolvent scalar algebra and its two marginal products."""
    scalar = resolvent_scalar_program()
    lines = [
        f'inline constexpr const char* resolvent_hash="{scalar.logical_hash}";',
        emit_scalar_cpp(
            scalar, function_name="resolvent_element", ordered_native_sums=True
        ).replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool "),
    ]
    for block in ("oo", "vv"):
        program = fock_moment_program(3, 2, block)
        names = _inputs(program)
        lines += [
            f"template<class Gemm> void fock_{block}(std::size_t v,std::size_t capacity,",
            ",".join(f"const double* {name}" for name in names)
            + ",double weight,double* output,Gemm&& gemm){",
        ]
        for index, node in enumerate(program.outputs.values()):
            lines.append(
                _gemm(
                    node,
                    {name: (name, None) for name in names},
                    "-0.5*weight" if block == "oo" else "0.5*weight",
                    "0.0" if block == "oo" and index == 0 else "1.0",
                )
            )
        lines.append("}")
    return "\n".join(lines)


def fock_cuda_source() -> str:
    """Bind six W cubes and pointwise V values without an ovvv source."""
    program = resolvent_scalar_program()
    lines = [
        "__global__ void resolvent_kernel(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double threshold,Inputs in,const double* moments,double* x,double* y,int* error){",
        " const auto v3=v*v*v; const std::size_t occupied[3]={i,j,k};",
        " for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x){",
        " const std::size_t a=flat/(v*v),b=(flat/v)%v,c=flat%v;",
        " const double gap=in.eps_o[i]+in.eps_o[j]+in.eps_o[k]-in.eps_v[a]-in.eps_v[b]-in.eps_v[c];",
        " if(!isfinite(gap)||gap>=0.0||fabs(gap)<=threshold){atomicCAS(error,0,1);x[flat]=y[flat]=0.0;continue;}",
    ]
    bindings = {"gap": "gap"}
    for name in _inputs(program):
        if name == "gap":
            continue
        kind, occ, vir = name.split("_")
        a, b, c = ("abc"[axis] for axis in VP[vir])
        if kind == "w":
            bindings[name] = f"moments[{_LABELS.index(occ)}*v3+({a}*v+{b})*v+{c}]"
        else:
            I, J, K = (f"occupied[{axis}]" for axis in VP[occ])
            lines += [
                f"double {name}=0.0;",
                _call(
                    v_scalar_program(),
                    "v_element",
                    {
                        "ovov": f"in.ovov[((({I})*v+{a})*o+({J}))*v+{b}]",
                        "t1": f"in.t1[({K})*v+{c}]",
                        "t2": f"in.t2[((({I})*o+({J}))*v+{a})*v+{b}]",
                        "fov": f"in.fov[({K})*v+{c}]",
                    },
                    name,
                ),
            ]
            bindings[name] = name
    call = _call(program, "resolvent_element", bindings, "xx,yy").removesuffix(";")
    lines += [
        "double xx=0.0,yy=0.0;",
        f"if(!{call}) atomicCAS(error,0,2);",
        "x[flat]=generativeqc_tensor::finite(xx,error,3); y[flat]=generativeqc_tensor::finite(yy,error,3); } }",
        "void resolvent_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double threshold,const Inputs& in,const double* moments,unsigned blocks,double* x,double* y,int* error,cudaStream_t stream){",
        "resolvent_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,threshold,in,moments,x,y,error);",
        "generativeqc_tensor::cuda_check(cudaGetLastError()); }",
    ]
    return "\n".join(lines)


def header() -> str:
    panel = df_panel_program(3, 4)
    moments = moment_program(2, 3)
    mixed_moments = w_fp32_candidate_program(2, 3)
    mixed_schedule = describe_precision(mixed_moments)
    scalar, v_scalar = energy_scalar_program(), v_scalar_program()
    w = moments.outputs["w"]
    mixed_w = mixed_moments.outputs["w"]
    if w.op != "add" or len(w.inputs) != 2:
        raise ValueError("occupied W seed must contain two audited products")
    if mixed_w.op != "add" or len(mixed_w.inputs) != 2:
        raise ValueError("mixed occupied W seed must retain the FP64 W sum")
    mixed_products = []
    for value in mixed_w.inputs:
        if (
            value.op != "cast"
            or value.spec.dtype != "float64"
            or len(value.inputs) != 1
            or value.inputs[0].op != "einsum"
            or value.inputs[0].spec.dtype != "float32"
        ):
            raise ValueError("mixed occupied W seed must cast two FP32 reductions to FP64")
        mixed_products.append(value.inputs[0])
    if mixed_w.attrs["coefficients"] != w.attrs["coefficients"]:
        raise ValueError("mixed occupied W coefficients differ from the strict equation")
    weights = w.attrs["coefficients"]
    coefficient = lambda pair: f"({pair[0]}.0/{pair[1]}.0)"
    lines = [
        "// Generated from occupied_triples TensorIR; do not edit.",
        "#pragma once",
        "#include <cstddef>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "#ifdef __CUDACC__",
        "#define GQC_DF_TRIPLES_HD __host__ __device__",
        "#else",
        "#define GQC_DF_TRIPLES_HD",
        "#endif",
        "namespace generativeqc::cc::triples::generated_df {",
        "using posthf::checked_add; using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> xs) {",
        "  std::size_t n=1; for (auto x:xs) n=checked_mul(n,x); return n; }",
        f'inline constexpr const char* panel_hash="{panel.logical_hash}";',
        f'inline constexpr const char* moment_hash="{moments.logical_hash}";',
        f'inline constexpr const char* w_fp32_precision_schedule_identity="{mixed_schedule.identity}";',
        f'inline constexpr const char* w_fp32_precision_request_identity="{mixed_schedule.request_identity}";',
        f'inline constexpr const char* epilogue_hash="{scalar.logical_hash}";',
        "inline constexpr unsigned permutations[6][3]={"
        + ",".join("{" + ",".join(map(str, p)) + "}" for p in PERMUTATIONS)
        + "};",
        "struct Inputs { const double *bov{},*bvv{},*ovoo{},*ovov{},*fov{},*t1{},*t2{},*eps_o{},*eps_v{}; };",
        "template<class Gemm> void build_panel(std::size_t o,std::size_t v,std::size_t q,",
        "  std::size_t i,const Inputs& in,double* output,Gemm&& gemm) {",
        _gemm(
            panel.outputs["panel"],
            {"bov_i": ("in.bov+i*v", "o*v"), "bvv": ("in.bvv", "v*v")},
            "1.0",
            "0.0",
        ),
        "}",
        "template<class Gemm> void build_w(std::size_t o,std::size_t v,",
        "  std::size_t i,std::size_t j,std::size_t k,const Inputs& in,",
        "  const double* panel,double* output,Gemm&& gemm) {",
        _gemm(
            w.inputs[0],
            {"panel": ("panel", "v"), "t2_kj": ("in.t2+(k*o+j)*v*v", "v")},
            coefficient(weights[0]),
            "0.0",
        ),
        _gemm(
            w.inputs[1],
            {
                "ovoo_ij": ("in.ovoo+(i*v*o+j)*o", "o*o"),
                "t2_mk": ("in.t2+k*v*v", "o*v*v"),
            },
            coefficient(weights[1]),
            "1.0",
        ),
        "}",
        "template<class Gemm,class Accumulate> void build_w_fp32(std::size_t o,std::size_t v,",
        "  std::size_t i,std::size_t j,std::size_t k,const float* ovoo,const float* t2,",
        "  const float* panel,float* scratch,double* output,Gemm&& gemm,Accumulate&& accumulate) {",
        _gemm(
            mixed_products[0],
            {"panel": ("panel", "v"), "t2_kj": ("t2+(k*o+j)*v*v", "v")},
            "1.0",
            "0.0",
            output_pointer="scratch",
        ),
        f"accumulate(scratch,checked_product({{v,v,v}}),{coefficient(weights[0])},0.0,output);",
        _gemm(
            mixed_products[1],
            {
                "ovoo_ij": ("ovoo+(i*v*o+j)*o", "o*o"),
                "t2_mk": ("t2+k*v*v", "o*v*v"),
            },
            "1.0",
            "0.0",
            output_pointer="scratch",
        ),
        f"accumulate(scratch,checked_product({{v,v,v}}),{coefficient(weights[1])},1.0,output);",
        "}",
    ]
    for program, name in ((v_scalar, "v_element"), (scalar, "energy_element")):
        lines.append(
            emit_scalar_cpp(
                program,
                function_name=name,
                caller_owned_checks=True,
                ordered_native_sums=True,
            ).replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool ")
        )
    lines += [
        response_blas_header(),
        response_scalar_header(),
        fock_header(),
        "}  // namespace generativeqc::cc::triples::generated_df",
        "#undef GQC_DF_TRIPLES_HD",
        "",
    ]
    return "\n".join(lines)


def cuda_header() -> str:
    return """// Generated occupied-tile epilogue interface; do not edit.
#pragma once
#include <cuda_runtime.h>
#include "generated_df_occupied_triples.hpp"
namespace generativeqc::cc::triples::generated_df {
void energy_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                 double degeneracy,double threshold,const Inputs& in,const double* moments,
                 unsigned blocks,double* partials,int* error,cudaStream_t stream);
void response_w_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
void response_v_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     unsigned permutation,double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
void response_gap_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
struct GapCudaState { std::size_t o{},v{}; const double* bar_gap{}; double* response_arena{};
                      int* error{}; cudaStream_t stream{}; };
GapOutputs gap_response_cuda(GapCudaState& state);
void resolvent_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                    double threshold,const Inputs& in,const double* moments,unsigned blocks,
                    double* x,double* y,int* error,cudaStream_t stream);
}
"""


def cuda_source() -> str:
    scalar, v_scalar = energy_scalar_program(), v_scalar_program()
    bindings = {"denominator": "denominator"}
    lines = [
        '#include "generated_df_occupied_triples_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::cc::triples::generated_df {",
        "__global__ void tile_kernel(std::size_t o,std::size_t v,std::size_t i,",
        "  std::size_t j,std::size_t k,double degeneracy,double threshold,Inputs in,",
        "  const double* moments,double* partials,int* error) {",
        "  const auto v3=v*v*v; const std::size_t occupied[3]={i,j,k};",
        "  double accumulated=0.0;",
        "  for (std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;",
        "       flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x) {",
        "    const std::size_t a=flat/(v*v),b=(flat/v)%v,c=flat%v;",
        "    const double gap=in.eps_o[i]+in.eps_o[j]+in.eps_o[k]-in.eps_v[a]-in.eps_v[b]-in.eps_v[c];",
        "    const double denominator=gap*degeneracy;",
        "    if (!isfinite(gap) || gap>=0.0 || fabs(gap)<=threshold || !isfinite(denominator)) {",
        "      atomicCAS(error,0,1); continue; }",
    ]
    for index, occ in enumerate(_LABELS):
        p = PERMUTATIONS[index]
        I, J, K = (f"occupied[{axis}]" for axis in p)
        v_name = f"v_{occ}"
        lines += [
            f"    double {v_name}=0.0;",
            "    "
            + _call(
                v_scalar,
                "v_element",
                {
                    "ovov": f"in.ovov[((({I})*v+a)*o+({J}))*v+b]",
                    "t1": f"in.t1[({K})*v+c]",
                    "t2": f"in.t2[((({I})*o+({J}))*v+a)*v+b]",
                    "fov": f"in.fov[({K})*v+c]",
                },
                v_name,
            ),
        ]
        bindings[v_name] = v_name
        for vir in _LABELS:
            x, y, z = ("abc"[axis] for axis in VP[vir])
            bindings[f"w_{occ}_{vir}"] = f"moments[{index}*v3+({x}*v+{y})*v+{z}]"
    lines += [
        "    double value=0.0;",
        "    " + _call(scalar, "energy_element", bindings, "value"),
        "    accumulated+=generativeqc_tensor::finite(value,error,1);",
        "  }",
        "  __shared__ double sums[256]; sums[threadIdx.x]=accumulated; __syncthreads();",
        "  for (unsigned stride=blockDim.x/2;stride;stride/=2) {",
        "    if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];",
        "    __syncthreads(); }",
        "  if (threadIdx.x==0) partials[blockIdx.x]=generativeqc_tensor::finite(sums[0],error,2);",
        "}",
        "void energy_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        "  double degeneracy,double threshold,const Inputs& in,const double* moments,",
        "  unsigned blocks,double* partials,int* error,cudaStream_t stream) {",
        "  tile_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,degeneracy,threshold,in,moments,partials,error);",
        "  generativeqc_tensor::cuda_check(cudaGetLastError());",
        "}",
        response_cuda_source(),
        fock_cuda_source(),
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, emitter in (
        (".hpp", header),
        ("_cuda.cuh", cuda_header),
        ("_cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_occupied_triples" + suffix)).write_text(
            emitter()
        )


if __name__ == "__main__":
    main()
