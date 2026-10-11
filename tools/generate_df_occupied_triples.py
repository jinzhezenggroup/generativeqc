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
from generativeqc_compiler.cc.occupied_triples_lowering import emit_w_portfolio
from generativeqc_compiler.cc.occupied_triples_response import (
    energy_scalar_vjp,
    fused_tile_program,
    gap_vjp,
    moment_vjp,
    panel_vjp,
    scaled_denominator_vjp,
)
from generativeqc_compiler.cc.triples import _LABELS, VP
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import describe_precision
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract
from generativeqc_compiler.tensor.indexed_cuda_reduction import (
    emit_indexed_reduction_cuda,
    plan_indexed_reduction,
)
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import contraction_initializer
from generativeqc_compiler.tensor.optimize import prepare_for_backend
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

from tools.generate_rccsd_native import _cuda_program, _dim, _required_function, _size

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

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
    bindings: Mapping[str, tuple[str, str | None]],
    alpha: str,
    beta: str,
    *,
    output_pointer: str = "output",
    output_leading_dimension: str | None = None,
    adapter: TensorLoweringAdapter | None = None,
    descriptors: list[str] | None = None,
    paired_bindings: Mapping[str, tuple[str, str | None]] | None = None,
    paired_output_pointer: str | None = None,
    paired_slot: int | None = None,
) -> str:
    """Derive a column-major call from a TensorIR product and strided views.

    Views supply the address and physical leading dimension of each logical
    row-major input. Operand permutation, transposition, contraction dimensions
    and output layout come exclusively from the shared GEMM contract. A None
    leading dimension denotes a contiguous tensor; derive its matrix cut from
    this contraction, since one cube can be [ab,c] in W1 and [a,bc] in W2.
    A paired call reuses an existing identical typed descriptor with two views;
    it cannot introduce a new equation, precision or physical leading dimension.
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
    paired_operands = []
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
        if paired_bindings is not None:
            paired_operands.append(paired_bindings[operand.attrs["name"]])
    aa, bb = operands
    if scale != 1:
        alpha = f"({alpha})*({scale.numerator}.0/{scale.denominator}.0)"
    if c != m + n:
        a, b, m, n, aa, bb = b, a, n, m, bb, aa
        paired_operands.reverse()
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
    if adapter is not None:
        if descriptors is None or c != g.m_labels + g.n_labels:
            raise ValueError("typed triples projection needs original input order")
        slot = len(descriptors)
        dimension = lambda index: {"occupied": "o", "virtual": "v", "auxiliary": "q"}[
            index.space.kind
        ]
        descriptor = contraction_initializer(
            adapter,
            node,
            dimension,
            transpose=(trans(a, m, k), trans(b, k, n)),
            extents=("1", extent(m), extent(n), extent(k)),
            coefficient=alpha,
            row_axes=(
                len(m) if a == m + k else len(k),
                len(k) if b == k + n else len(n),
                len(m),
            ),
            leading_dimensions=(lda, ldb, output_leading_dimension or extent(n)),
            beta=beta,
        )
        if paired_bindings is not None:
            if (
                paired_output_pointer is None
                or paired_slot is None
                or not 0 <= paired_slot < len(descriptors)
            ):
                raise ValueError(
                    "paired triples need an existing descriptor and second output"
                )
            if descriptors[paired_slot] != descriptor:
                raise ValueError(
                    "paired triples differ from the prepared scientific descriptor"
                )
            other_left, other_right = paired_operands
            if (other_left[1] or extent(k if a == m + k else m)) != lda or (
                other_right[1] or extent(n if b == k + n else k)
            ) != ldb:
                raise ValueError(
                    "paired triples views require different physical strides"
                )
            return (
                f"table.execute_independent_pair({paired_slot},o,v,q,context.stream,"
                f"{aa[0]},{other_left[0]},{bb[0]},{other_right[0]},"
                f"{output_pointer},{paired_output_pointer},context.error,"
                "pointer_storage,plan_.pair_pointer_bytes());"
            )
        if paired_output_pointer is not None or paired_slot is not None:
            raise ValueError("paired triples metadata requires paired input views")
        descriptors.append(descriptor)
        return f"table.execute({slot},o,v,q,context.stream,{aa[0]},{bb[0]},{output_pointer},context.error);"
    if (
        paired_bindings is not None
        or paired_output_pointer is not None
        or paired_slot is not None
    ):
        raise ValueError("paired triples require typed prepared execution")
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
    scalars = _response_scalars()
    fused = fused_tile_program()
    fusion_identity = canonical_hash(
        {
            "schema": "generativeqc.occupied-triples-scalar-fusion.v1",
            "equation": fused.logical_hash,
            "threads": 256,
            "coordinate_order": _LABELS,
            "denominators": "six-ordered-inverse-permutations",
            "seed_storage": "six-w-and-six-v-cubes",
            "arithmetic": "fp64-ordered-native-sums-no-reassociation",
            "scalar_emission": "output-dependency-order",
            "energy_reduction": "unchanged-grid-stride-and-256-thread-tree",
            "gap_cotangents": "unrequested",
        }
    )
    lines += [
        f'inline constexpr const char* scalar_fusion_equation_identity="{fused.logical_hash}";',
        f'inline constexpr const char* scalar_fusion_identity="{fusion_identity}";',
        emit_scalar_cpp(
            fused,
            function_name="fused_response_element",
            ordered_native_sums=True,
            output_dependency_order=True,
        ).replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool "),
    ]
    # Count logical scalar arithmetic and source reads, not hardware instructions
    # or cache traffic. Native owners multiply these receipts by executed tiles.
    separate = [
        energy_scalar_program(),
        *[program for name, program in scalars.items() if name != "denominator"],
    ]
    for label, programs in (("fused", [fused]), ("unfused", separate)):
        inputs = arithmetic = 0
        for program in programs:
            lowered = prepare_for_backend(program, "scalar")
            for node in lowered.live_nodes:
                if node.op == "add":
                    arithmetic += (
                        len(node.inputs)
                        - 1
                        + sum(
                            coefficient not in ((1, 1), (-1, 1))
                            for coefficient in node.attrs["coefficients"]
                        )
                    )
                elif node.op in ("multiply", "divide"):
                    arithmetic += 1
                elif node.op not in ("input", "constant"):
                    raise ValueError(
                        "unexpected primitive in scalar fusion work receipt"
                    )
            for name in _inputs(program):
                if name.startswith("v_"):
                    arithmetic += 3  # Two products and their sum in v_scalar_program.
                    inputs += 4
                elif name.startswith("w_"):
                    inputs += 1
                elif name.startswith("denominator"):
                    arithmetic += 6
                    inputs += 6
        lines += [
            f"inline constexpr std::size_t scalar_{label}_value_reads={inputs};",
            f"inline constexpr std::size_t scalar_{label}_arithmetic_ops={arithmetic};",
        ]
    for name, program in scalars.items():
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
    without_gap = tuple(name for name in scalars if name != "denominator")
    _, parallel_header, _ = _parallel_gap_response()
    lines += [
        f"inline constexpr std::size_t response_scalar_outputs_with_gap={len(scalars)};",
        f"inline constexpr std::size_t response_scalar_outputs_without_gap={len(without_gap)};",
        f'inline constexpr const char* response_without_gap_identity="{energy_scalar_vjp(without_gap).logical_hash}";',
        "struct GapOutputs { const double *bar_eps_i{},*bar_eps_j{},*bar_eps_k{},*bar_eps_v{}; };",
        _required_function(program, "gap_response_serial_arena_elements"),
        f"inline constexpr std::size_t gap_response_operations={sum(n.op != 'input' for n in program.live_nodes)};",
        parallel_header,
        "inline bool gap_response_parallel_selected(std::size_t v,bool enabled){return enabled && v>=4;}",
        "inline std::size_t gap_response_arena_elements(std::size_t o,std::size_t v,bool parallel=false){",
        "return gap_response_parallel_selected(v,parallel)?gap_response_parallel_arena_elements(o,v):gap_response_serial_arena_elements(o,v);}",
        "inline std::size_t gap_response_kernel_count(std::size_t o,std::size_t v,bool parallel){",
        "return gap_response_parallel_selected(v,parallel)?gap_response_parallel_kernel_count(o,v):gap_response_operations;}",
    ]
    output_roots = set(program.outputs.values())
    original_counts = {
        "materialized_elements": [
            _size(node.spec)
            for node in program.live_nodes
            if node.op != "input" and node not in output_roots
        ],
        "value_reads": [
            _size(child.spec) for node in program.live_nodes for child in node.inputs
        ],
        "value_writes": [
            _size(node.spec) for node in program.live_nodes if node.op != "input"
        ],
        "reduction_summands": [
            _size(node.inputs[0].spec)
            for node in program.live_nodes
            if node.op == "reduce"
        ],
    }
    for name, terms in original_counts.items():
        expression = "0"
        for term in terms:
            expression = f"checked_add({expression},{term})"
        lines.append(
            f"inline std::size_t gap_response_serial_{name}(std::size_t o,std::size_t v){{return {expression};}}"
        )
        lines.append(
            f"inline std::size_t gap_response_{name}(std::size_t o,std::size_t v,bool parallel){{"
            f"return gap_response_parallel_selected(v,parallel)?gap_response_parallel_{name}(o,v):gap_response_serial_{name}(o,v);}}"
        )
    return "\n".join(lines)


def _parallel_gap_response() -> tuple[str, str, str]:
    """Stream the audited gap AD graph; scheduling introduces no CC equation."""
    return emit_indexed_reduction_cuda(
        plan_indexed_reduction(gap_vjp(3)),
        "gap_response_parallel",
        dimension=_dim,
        parameters=("o", "v"),
        input_bindings={"bar_gap": "state.bar_gap"},
        state_type="GapCudaState",
        output_type="GapOutputs",
    )


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
    parallel_source, _, _ = _parallel_gap_response()
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
        parallel_source,
        "GapOutputs gap_response_cuda(GapCudaState& s){",
        "return gap_response_parallel_selected(s.v,s.parallel_reduction)?run_gap_response_parallel(s):run_gap_response(s);}",
    ]
    return "\n".join(lines) + "\n" + fused_response_cuda_source()


def fused_response_cuda_source() -> str:
    """One bounded traversal publishes primal energy plus packed W/V seeds."""
    program = fused_tile_program()
    outputs = tuple(sorted(program.outputs))
    lines = [
        "__global__ void fused_response_kernel(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double degeneracy,double threshold,Inputs in,const double* moments,double* bar_w,double* bar_v,double* partials,int* error){",
        " const auto v3=v*v*v; const std::size_t occupied[3]={i,j,k}; double accumulated=0.0;",
        " for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x){",
        " const std::size_t coordinates[3]={flat/(v*v),(flat/v)%v,flat%v};",
    ]
    bindings = {}
    for name in _inputs(program):
        if name == "bar_energy":
            bindings[name] = "1.0"
        elif name.startswith("denominator_"):
            virtual = name.split("_")[1]
            a, b, c = (f"coordinates[{axis}]" for axis in VP[virtual])
            lines += [
                f"const double gap_{virtual}=in.eps_o[i]+in.eps_o[j]+in.eps_o[k]-in.eps_v[{a}]-in.eps_v[{b}]-in.eps_v[{c}];",
                f"const double {name}=gap_{virtual}*degeneracy;",
                f"if(!isfinite(gap_{virtual})||gap_{virtual}>=0.0||fabs(gap_{virtual})<=threshold||!isfinite({name})) atomicCAS(error,0,1);",
            ]
            bindings[name] = name
        else:
            kind, occupied, virtual = name.split("_")
            a, b, c = (f"coordinates[{axis}]" for axis in VP[virtual])
            if kind == "w":
                bindings[name] = (
                    f"moments[{_LABELS.index(occupied)}*v3+({a}*v+{b})*v+{c}]"
                )
            elif kind == "v":
                I, J, K = (f"occupied[{axis}]" for axis in VP[occupied])
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
            else:
                raise ValueError("unexpected fused scalar boundary input")
    lines += [f"double {name}=0.0;" for name in outputs]
    call = _call(
        program, "fused_response_element", bindings, ",".join(outputs)
    ).removesuffix(";")
    lines.append(f"if(!{call}) atomicCAS(error,0,2);")
    for index, label in enumerate(_LABELS):
        lines += [
            f"bar_w[{index}*v3+flat]=generativeqc_tensor::finite(bar_w_{label},error,3);",
            f"bar_v[{index}*v3+flat]=generativeqc_tensor::finite(bar_v_{label},error,4);",
        ]
    lines += [
        "accumulated+=generativeqc_tensor::finite(energy,error,1); }",
        "__shared__ double sums[256]; sums[threadIdx.x]=accumulated; __syncthreads();",
        "for(unsigned stride=blockDim.x/2;stride;stride/=2){",
        "if(threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride]; __syncthreads(); }",
        "if(threadIdx.x==0) partials[blockIdx.x]=generativeqc_tensor::finite(sums[0],error,2); }",
        "void fused_response_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        " double degeneracy,double threshold,const Inputs& in,const double* moments,unsigned blocks,",
        " double* bar_w,double* bar_v,double* partials,int* error,cudaStream_t stream){",
        "fused_response_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,degeneracy,threshold,in,moments,bar_w,bar_v,partials,error);",
        "generativeqc_tensor::cuda_check(cudaGetLastError()); }",
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


def native_execution_header() -> str:
    """Emit typed, prepared execution of the existing strict/mixed W region."""
    panel, moments, mixed = (
        df_panel_program(3, 4),
        moment_program(2, 3),
        w_fp32_candidate_program(2, 3),
    )
    panel_adapter, strict_adapter, mixed_adapter = map(
        TensorLoweringAdapter, (panel, moments, mixed)
    )
    w, mixed_w = moments.outputs["w"], mixed.outputs["w"]
    products = [value.inputs[0] for value in mixed_w.inputs]
    coefficient = lambda pair: f"({pair[0]}.0/{pair[1]}.0)"
    weights = w.attrs["coefficients"]
    panel_descriptors: list[str] = []
    strict_descriptors: list[str] = []
    mixed_descriptors: list[str] = []
    panel_call = _gemm(
        panel.outputs["panel"],
        {"bov_i": ("in.bov+i*v", "o*v"), "bvv": ("in.bvv", "v*v")},
        "1.0",
        "0.0",
        adapter=panel_adapter,
        descriptors=panel_descriptors,
    )
    strict_calls, mixed_calls = [], []
    for slot, (left, right) in enumerate(
        (
            (
                {"panel": ("panel", "v"), "t2_kj": ("in.t2+(k*o+j)*v*v", "v")},
                {"panel": ("panel", "v"), "t2_kj": ("t2+(k*o+j)*v*v", "v")},
            ),
            (
                {
                    "ovoo_ij": ("in.ovoo+(i*v*o+j)*o", "o*o"),
                    "t2_mk": ("in.t2+k*v*v", "o*v*v"),
                },
                {
                    "ovoo_ij": ("ovoo+(i*v*o+j)*o", "o*o"),
                    "t2_mk": ("t2+k*v*v", "o*v*v"),
                },
            ),
        )
    ):
        strict_calls.append(
            _gemm(
                w.inputs[slot],
                left,
                coefficient(weights[slot]),
                f"{slot}.0",
                adapter=strict_adapter,
                descriptors=strict_descriptors,
            )
        )
        mixed_calls.append(
            _gemm(
                products[slot],
                right,
                "1.0",
                "0.0",
                output_pointer="scratch",
                adapter=mixed_adapter,
                descriptors=mixed_descriptors,
            )
        )
        mixed_calls.append(
            f"generativeqc_tensor::accumulate_fp32_into_fp64(context,scratch,output,v3,{coefficient(weights[slot])},{slot}.0,22);"
        )
    paired_calls = []
    for slot, (first_views, second_views) in enumerate(
        (
            (
                {
                    "panel": ("panel", "v"),
                    "t2_kj": ("in.t2+(occupied_third*o+occupied_second)*v*v", "v"),
                },
                {
                    "panel": ("panel", "v"),
                    "t2_kj": ("in.t2+(occupied_second*o+occupied_third)*v*v", "v"),
                },
            ),
            (
                {
                    "ovoo_ij": (
                        "in.ovoo+(occupied_first*v*o+occupied_second)*o",
                        "o*o",
                    ),
                    "t2_mk": ("in.t2+occupied_third*v*v", "o*v*v"),
                },
                {
                    "ovoo_ij": ("in.ovoo+(occupied_first*v*o+occupied_third)*o", "o*o"),
                    "t2_mk": ("in.t2+occupied_second*v*v", "o*v*v"),
                },
            ),
        )
    ):
        paired_calls.append(
            _gemm(
                w.inputs[slot],
                first_views,
                coefficient(weights[slot]),
                f"{slot}.0",
                output_pointer="first_output",
                adapter=strict_adapter,
                descriptors=strict_descriptors,
                paired_bindings=second_views,
                paired_output_pointer="second_output",
                paired_slot=slot,
            )
        )
    code_identity = canonical_hash(
        {
            "descriptors": [panel_descriptors, strict_descriptors, mixed_descriptors],
            "calls": [panel_call, strict_calls, mixed_calls, paired_calls],
            "schema": "prepared-w-v3-independent-pairs",
        }
    )
    return (
        "\n".join(
            [
                "#ifdef __CUDACC__",
                emit_w_portfolio(code_identity),
                r"""
struct WPlan {
  std::size_t selected{};
  runtime::NativeLoweringPrecision precision;
  bool retained_incumbent{};
  std::string_view schedule_identity;
  tensor::ContractionProviderReservation optional_reservation;
  std::size_t pair_pointer_bytes() const {
    return precision.arithmetic.is_strict_fp64() &&
        algorithm()==tensor::ContractionAlgorithm::PedanticBlas
        ? tensor::PreparedContractions::independent_pair_storage_bytes() : 0;
  }
  tensor::ContractionAlgorithm algorithm() const {
    const auto provider=w_lowering_candidates[selected].provider;
    if(provider=="cublas") return tensor::ContractionAlgorithm::PedanticBlas;
    if(provider=="generated.cuda") return tensor::ContractionAlgorithm::GeneratedOrdered;
    if(provider=="cutensor") return tensor::ContractionAlgorithm::CutensorAffine;
    throw std::logic_error("unbound triples W provider");
  }
  std::size_t provider_bytes() const {
    if(algorithm()!=tensor::ContractionAlgorithm::CutensorAffine)
      return w_lowering_candidates[selected].provider_bytes;
    // One strict panel and two W plans coexist; workspace is owned by those
    // prepared plans outside the method arena, and must be admitted with them.
    return checked_mul(3,checked_add(optional_reservation.workspace_bytes,
                                    optional_reservation.provider_bytes));
  }
  std::size_t optional_host_bytes() const {
    return algorithm()==tensor::ContractionAlgorithm::CutensorAffine
        ? checked_mul(3,optional_reservation.host_bytes) : 0;
  }
  std::size_t storage_bytes(std::size_t o,std::size_t v,std::size_t panels) const {
    if(precision.arithmetic.storage_dtype==runtime::PrecisionDtype::Fp64) return 0;
    const auto v3=checked_product({v,v,v});
    return checked_mul(checked_add(checked_add(checked_product({o,v,o,o}),checked_product({o,o,v,v})),
                                  checked_mul(checked_add(panels,1),v3)),sizeof(float));
  }
};
inline WPlan prepare_w_plan(runtime::PrecisionDirective admitted,bool library_available=true) {
  const bool strict=admitted.is_strict_fp64();
  if(admitted.math_mode!=runtime::kStrictPrecisionMathMode ||
      (!strict && (admitted.storage_dtype!=runtime::PrecisionDtype::Fp32 ||
                   admitted.compute_dtype!=runtime::PrecisionDtype::Fp32 ||
                   admitted.accumulation_dtype!=runtime::PrecisionDtype::Fp32 ||
                   admitted.qualification!="issue1764/df-triples-w-fp32-candidate-v1")))
    throw std::invalid_argument("unqualified triples W arithmetic");
  const auto reservation=tensor::qualified_cutensor_reservation();
  const auto version=tensor::cutensor_provider_version();
  auto offers=w_lowering_candidates;
  std::optional<std::size_t> incumbent;
  for(std::size_t i=0;i<offers.size();++i) {
    auto& offer=offers[i];
    const bool offer_strict=w_lowering_precisions[offer.precision].arithmetic.is_strict_fp64();
    if(strict && !offer_strict) offer.rejection="scientific owner did not admit this precision";
    if(!library_available && offer.provider!="generated.cuda")
      offer.rejection="optional provider preparation unavailable";
    if(offer.provider=="cutensor") {
      if(!reservation.host_bytes || version<20800 || version/10000!=2)
        offer.rejection="qualified cuTENSOR resource profile unavailable";
      offer.workspace_bytes=checked_mul(3,reservation.workspace_bytes);
      offer.provider_bytes=checked_mul(3,reservation.provider_bytes);
      offer.host_bytes=checked_mul(3,reservation.host_bytes);
    }
    if(offer_strict==strict && offer.provider==(library_available?"cublas":"generated.cuda"))
      incumbent=i;
#if defined(GENERATIVEQC_TEST_HOOKS)
    // Synthetic complete costs exercise joint selection only in qualification
    // builds. They are not measured performance evidence or production policy.
    if(library_available && reservation.host_bytes) {
      offer.cost.source="test-only-provider-ranking";
      offer.cost.prepare_ns=0;offer.cost.cast_ns=0;offer.cost.pack_ns=0;
      offer.cost.refinement_ns=0;offer.cost.audit_ns=0;offer.cost.fallback_ns=0;
      offer.cost.kernel_ns=(offer.provider=="cutensor"?1:100)+(offer_strict==strict?0:10);
    }
#endif
  }
  if(!incumbent) throw std::logic_error("qualified W incumbent is missing");
  const auto selected=runtime::select_native_lowering(w_lowering_request,offers,w_lowering_target,
                                                     w_lowering_compilation,1,incumbent);
  return {selected.selected,w_lowering_precisions[offers[selected.selected].precision],selected.retained_incumbent,
          w_lowering_precisions[offers[selected.selected].precision].arithmetic.is_strict_fp64()
              ? "@STRICT_SCHEDULE@" : "@MIXED_SCHEDULE@",reservation};
}

// Finite resource fallback order: retain precision with generated execution,
// then restore strict precision. No method owner chooses a vendor or dtype.
inline std::optional<WPlan> lower_resource_w_plan(const WPlan& plan) {
  if(plan.algorithm()!=tensor::ContractionAlgorithm::GeneratedOrdered)
    return prepare_w_plan(plan.precision.arithmetic,false);
  if(!plan.precision.arithmetic.is_strict_fp64()) return prepare_w_plan(runtime::strict_fp64_precision(),false);
  return std::nullopt;
}

/** Complete prepared W region. Native methods supply canonical inputs and
 * borrowed storage; casts, matrix algorithms and FP64 publication live here.
 * Tables/context are prepared once; repeated occupied tiles only bind views.
 * The caller's tensor Context and arena must outlive this object. */
class WExecution {
 public:
  static std::size_t host_bytes(const WPlan& plan) {
    return checked_add(plan.optional_host_bytes(),sizeof(WExecution)+
        tensor::PreparedContractions::storage_bytes(1)+tensor::PreparedContractions::storage_bytes(2));
  }
  WExecution(WPlan plan,std::size_t o_,std::size_t v_,std::size_t q_,generativeqc_tensor::Context& context,
             unsigned char* storage,std::size_t panels,std::size_t& fp64_calls,std::size_t& fp32_calls,
             std::size_t& summands,std::size_t& casts)
      : o(o_),v(v_),q(q_),v3(checked_product({v,v,v})),plan_(plan),casts_(&casts) {
    if(plan_.algorithm()!=tensor::ContractionAlgorithm::PedanticBlas)
      provider_.prepare_generated(context.stream);
    else if(!provider_.prepare(context.stream)) {
      provider_.prepare_generated(context.stream);
      plan_=prepare_w_plan(plan.precision.arithmetic,false);
    }
    const auto bind=[&] {
      const auto algorithm=plan_.algorithm();
""",
                "    panel_table_.add(o,v,q,{"
                + ",".join(panel_descriptors)
                + "},provider_,fp64_calls,summands,{algorithm},plan_.optional_reservation);",
                "    if(plan_.precision.arithmetic.storage_dtype==runtime::PrecisionDtype::Fp64) {",
                "      w_table_.add(o,v,q,{"
                + ",".join(strict_descriptors)
                + "},provider_,fp64_calls,summands,{algorithm,algorithm},plan_.optional_reservation);",
                "    } else {",
                "      w_table_.add(o,v,q,{"
                + ",".join(mixed_descriptors)
                + "},provider_,fp32_calls,summands,{algorithm,algorithm},plan_.optional_reservation);",
                r"""
      }
    };
    try { bind(); }
    catch(const tensor::ContractionPreparationUnavailable&) {
      // Same-precision generated storage is a subset of the already admitted
      // layout. Checked release must succeed before the one permitted retry.
      w_table_.release();panel_table_.release();
      plan_=prepare_w_plan(plan_.precision.arithmetic,false);
      bind();
    }
    if(plan_.precision.arithmetic.is_strict_fp64()) launch_=&WExecution::strict_w;
    else {
      auto* cursor=reinterpret_cast<float*>(storage);
      ovoo_=cursor;cursor+=checked_product({o,v,o,o});
      t2_=cursor;cursor+=checked_product({o,o,v,v});
      panels_=cursor;cursor+=checked_mul(panels,v3);
      scratch_=cursor;
      launch_=&WExecution::mixed_w;
    }
  }
  const WPlan& plan() const noexcept { return plan_; }
  std::size_t provider_bytes() const {
    return checked_add(provider_.retained_bytes(),checked_add(panel_table_.optional_resources().provider_bytes,
                                                             w_table_.optional_resources().provider_bytes));
  }
  int provider_version() const {
    return plan_.algorithm()==tensor::ContractionAlgorithm::CutensorAffine
        ? int(tensor::cutensor_provider_version()) : provider_.provider_version();
  }
  int runtime_version() const noexcept { return provider_.runtime_version(); }
  void initialize(generativeqc_tensor::Context& context,const Inputs& in) {
    if(!ovoo_) return;
    const auto a=checked_product({o,v,o,o}),b=checked_product({o,o,v,v});
    generativeqc_tensor::convert_fp64_to_fp32(context,in.ovoo,ovoo_,a,20);
    generativeqc_tensor::convert_fp64_to_fp32(context,in.t2,t2_,b,21);
    *casts_=checked_add(*casts_,checked_add(a,b));
  }
  void build_panel(generativeqc_tensor::Context& context,const Inputs& in,std::size_t i,
                   double* output,std::size_t slot) {
    auto& table=panel_table_;
""",
                panel_call,
                r"""
    if(panels_) {
      generativeqc_tensor::convert_fp64_to_fp32(context,output,panels_+slot*v3,v3,23);
      *casts_=checked_add(*casts_,v3);
    }
  }
  void build_w(generativeqc_tensor::Context& context,const Inputs& in,
               std::size_t i,std::size_t j,std::size_t k,const double* panel,double* output,std::size_t slot) {
    (this->*launch_)(context,in,i,j,k,panel,output,slot);
  }
  /** Group independent seeds without changing either original W request.
   * Refusal happens before work; dispatch/arithmetic errors cannot replay a
   * partially evaluated seed. Both prepared cuts must support the same region. */
  bool build_w_pair(generativeqc_tensor::Context& context,const Inputs& in,
      std::size_t occupied_first,std::size_t occupied_second,std::size_t occupied_third,
      const double* panel,double* first_output,double* second_output,void* pointer_storage) {
    if(!plan_.pair_pointer_bytes() || !pointer_storage || occupied_second==occupied_third ||
       second_output!=first_output+v3) return false;
    auto& table=w_table_;
    if(!table.independent_pair_supported(0,o,v,q,context.stream) ||
       !table.independent_pair_supported(1,o,v,q,context.stream)) return false;
""",
                *paired_calls,
                r"""
    return true;
  }
 private:
  void strict_w(generativeqc_tensor::Context& context,const Inputs& in,
                std::size_t i,std::size_t j,std::size_t k,const double* panel,double* output,std::size_t) {
    auto& table=w_table_;
""",
                *strict_calls,
                r"""
  }
  void mixed_w(generativeqc_tensor::Context& context,const Inputs&,
               std::size_t i,std::size_t j,std::size_t k,const double*,double* output,std::size_t slot) {
    auto& table=w_table_;
    const auto* ovoo=ovoo_;const auto* t2=t2_;const auto* panel=panels_+slot*v3;
    auto* scratch=scratch_;
""",
                *mixed_calls,
                r"""
    *casts_=checked_add(*casts_,checked_mul(2,v3));
  }
  using Launcher=void(WExecution::*)(generativeqc_tensor::Context&,const Inputs&,
      std::size_t,std::size_t,std::size_t,const double*,double*,std::size_t);
  std::size_t o,v,q,v3;
  WPlan plan_;
  tensor::CudaContractionContext provider_;
  tensor::PreparedContractions panel_table_,w_table_;
  float *ovoo_{},*t2_{},*panels_{},*scratch_{};
  std::size_t* casts_{};
  Launcher launch_{};
};
#endif
""",
            ]
        )
        .replace("@STRICT_SCHEDULE@", describe_precision(moments).identity)
        .replace("@MIXED_SCHEDULE@", describe_precision(mixed).identity)
    )


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
            raise ValueError(
                "mixed occupied W seed must cast two FP32 reductions to FP64"
            )
        mixed_products.append(value.inputs[0])
    if mixed_w.attrs["coefficients"] != w.attrs["coefficients"]:
        raise ValueError(
            "mixed occupied W coefficients differ from the strict equation"
        )
    weights = w.attrs["coefficients"]
    coefficient = lambda pair: f"({pair[0]}.0/{pair[1]}.0)"
    lines = [
        "// Generated from occupied_triples TensorIR; do not edit.",
        "#pragma once",
        "#include <algorithm>",
        "#include <cstddef>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "#ifdef __CUDACC__",
        '#include "tensor/cuda_contraction.cuh"',
        '#include "runtime/lowering_binding.hpp"',
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
        "/** Alias only equal physical occupied tuples; retain every projected energy contribution. */",
        "struct MomentSourceMap { unsigned index[6]{0,1,2,3,4,5}; };",
        "inline MomentSourceMap occupied_moment_sources(std::size_t occupied_first,",
        "  std::size_t occupied_second,std::size_t occupied_third) {",
        "  const std::size_t occupied[3]={occupied_first,occupied_second,occupied_third};",
        "  MomentSourceMap sources;",
        "  for (unsigned source=0;source<6;++source)",
        "    for (unsigned candidate=0;candidate<source;++candidate) {",
        "      bool equal=true;",
        "      for (unsigned axis=0;axis<3;++axis)",
        "        equal=equal && occupied[permutations[source][axis]]==occupied[permutations[candidate][axis]];",
        "      if (equal) { sources.index[source]=candidate; break; }",
        "    }",
        "  return sources;",
        "}",
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
        native_execution_header(),
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
void energy_distinct_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                         double degeneracy,double threshold,const Inputs& in,const double* moments,
                         const MomentSourceMap& sources,unsigned blocks,double* partials,
                         int* error,cudaStream_t stream);
void response_w_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
void fused_response_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                         double degeneracy,double threshold,const Inputs& in,const double* moments,
                         unsigned blocks,double* bar_w,double* bar_v,double* partials,int* error,cudaStream_t stream);
void response_v_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     unsigned permutation,double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
void response_gap_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                     double degeneracy,double threshold,const Inputs& in,const double* moments,
                     unsigned blocks,double* output,int* error,cudaStream_t stream);
struct GapCudaState { std::size_t o{},v{}; const double* bar_gap{}; double* response_arena{};
                      int* error{}; cudaStream_t stream{}; bool parallel_reduction{}; };
GapOutputs gap_response_cuda(GapCudaState& state);
void resolvent_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                    double threshold,const Inputs& in,const double* moments,unsigned blocks,
                    double* x,double* y,int* error,cudaStream_t stream);
}
"""


def _energy_tile_kernel(name: str, *, remapped: bool = False) -> str:
    """Emit the same six projected terms, optionally aliasing equal W seeds.

    The map changes only occupied-source addresses. Virtual permutations,
    denominator multiplicities and finite checks remain in the original algebra.
    Response kernels retain independent cotangents and never use this map.
    """
    scalar, v_scalar = energy_scalar_program(), v_scalar_program()
    bindings = {"denominator": "denominator"}
    lines = [
        f"__global__ void {name}(std::size_t o,std::size_t v,std::size_t i,",
        "  std::size_t j,std::size_t k,double degeneracy,double threshold,Inputs in,",
        "  const double* moments,MomentSourceMap sources,double* partials,int* error) {"
        if remapped
        else "  const double* moments,double* partials,int* error) {",
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
            source = f"sources.index[{index}]" if remapped else str(index)
            bindings[f"w_{occ}_{vir}"] = f"moments[{source}*v3+({x}*v+{y})*v+{z}]"
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
    ]
    return "\n".join(lines)


def cuda_source() -> str:
    lines = [
        '#include "generated_df_occupied_triples_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::cc::triples::generated_df {",
        _energy_tile_kernel("tile_kernel"),
        _energy_tile_kernel("distinct_tile_kernel", remapped=True),
        "void energy_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        "  double degeneracy,double threshold,const Inputs& in,const double* moments,",
        "  unsigned blocks,double* partials,int* error,cudaStream_t stream) {",
        "  tile_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,degeneracy,threshold,in,moments,partials,error);",
        "  generativeqc_tensor::cuda_check(cudaGetLastError());",
        "}",
        "void energy_distinct_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        "  double degeneracy,double threshold,const Inputs& in,const double* moments,",
        "  const MomentSourceMap& sources,unsigned blocks,double* partials,int* error,cudaStream_t stream) {",
        "  distinct_tile_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,degeneracy,threshold,in,moments,sources,partials,error);",
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
