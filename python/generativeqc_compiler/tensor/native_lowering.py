"""Project canonical TensorIR lowering requests into native AOT descriptors.

Runtime extents are supplied by the existing native emitter; this adapter owns
no orbital/method policy and does not construct an alternative scientific IR.
Representative request hashes identify the AOT template, while native binding
compatibility additionally includes every resolved operand extent and stride.
"""

from __future__ import annotations

import json
import typing
from dataclasses import replace
from math import prod

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .checked_contraction_pair import CheckedTransposePair
    from .ir import Node
    from .lowering import TensorLoweringAdapter
    from .program import Program
    from .types import Index


def emit_contraction_region_portfolio(
    request: LoweringRequest, source_identity: str, *, name: str
) -> str:
    """Offer homogeneous prepared implementations for one compiler region.

    Native preparation resolves resource/version availability. This factory
    neither invents precision variants nor implements casts/refinement; regions
    requiring those obligations need a composite executor such as triples W.
    """
    from generativeqc_compiler.common.backend import TargetInfo
    from generativeqc_compiler.common.lowering_contract import CandidateExecution
    from generativeqc_compiler.common.lowering_provider import (
        LoweringCandidate,
        ProviderDescriptor,
    )
    from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
    from generativeqc_compiler.common.schedule import ScheduleTopology
    from generativeqc_compiler.common.specialization import (
        CompilationIdentity,
        TargetCapabilities,
    )

    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    providers = tuple(
        ProviderDescriptor(provider, kind, "prepared-affine-region", version=version)
        for provider, kind, version in (
            ("cublas", "library", "runtime-bound-pedantic"),
            ("generated.cuda", "generated", source_identity),
            ("cutensor", "library", "runtime-bound-qualified-2.x"),
            ("cublaslt", "library", "runtime-bound-qualified-matmul"),
        )
    )
    candidates = []
    batch_scaled = "batch_scale_mode" in dict(request.semantics)
    checked = "scalar_update_hash" in dict(request.semantics)
    for precision in request.precisions:
        d = precision.directive
        if (
            precision.casts
            or precision.refinement
            or precision.audit
            or len(
                {
                    *precision.input_dtypes,
                    precision.publication_dtype,
                    d.storage_dtype,
                    d.compute_dtype,
                    d.accumulation_dtype,
                }
            )
            != 1
        ):
            raise ValueError("contraction region requires homogeneous arithmetic")
        for provider in providers:
            candidates.append(
                LoweringCandidate(
                    request,
                    "region-" + provider.name,
                    (provider,),
                    "ready",
                    d.math_mode,
                    execution=CandidateExecution(
                        precision,
                        (
                            "ordered-checked-scalar"
                            if checked and provider.name == "generated.cuda"
                            else "ordered-checked-scalar-unimplemented"
                            if checked
                            else "batch-scaled-fused"
                            if batch_scaled and provider.name == "generated.cuda"
                            else "batch-scaled-contraction-and-publication"
                            if batch_scaled
                            else "prepared-affine-region"
                        ),
                        request.operands,
                        ScheduleTopology(
                            materialization="compiler-owned-liveness",
                            reduction=(
                                "increasing-logical-index"
                                if checked and provider.name == "generated.cuda"
                                else "provider-reproducible"
                            ),
                        ),
                        determinism=(
                            "exact-order"
                            if checked and provider.name == "generated.cuda"
                            else "reproducible"
                        ),
                        capture_safe=False,
                    ),
                    target=target,
                )
            )
    assert request.scientific_identity is not None
    return native_lowering_portfolio(
        request,
        candidates,
        target,
        CompilationIdentity(request.scientific_identity, source_identity),
        name=name,
    )


def projected_contraction_request(
    adapter: TensorLoweringAdapter,
    node: Node,
    *,
    fixed_modes: tuple[int, ...] = (),
    operand_order: tuple[int, int] = (0, 1),
) -> LoweringRequest:
    """Project compiler-owned row traversal into the common semantic request.

    Fixing modes describes one slice of the existing einsum, not a new algebra.
    The enclosing compiler schedule must traverse every fixed reduction mode
    and accumulate its contributions. Only leading packed axes can be removed;
    this cannot silently materialize a strided interior slice. All providers
    consume the same projected request, with the parent operation retained.
    """
    request = adapter.request(node, backend="cuda")
    if (
        node.op != "einsum"
        or len(node.inputs) != 2
        or operand_order not in ((0, 1), (1, 0))
    ):
        raise ValueError("native projection requires two ordered einsum operands")
    if not fixed_modes and operand_order == (0, 1):
        return request
    fixed = set(fixed_modes)
    if len(fixed) != len(fixed_modes) or not fixed <= {
        mode for operand in request.operands for mode in operand.modes
    }:
        raise ValueError("invalid fixed contraction modes")
    operands = []
    for index in (*operand_order, 2):
        operand = request.operands[index]
        kept = [axis for axis, mode in enumerate(operand.modes) if mode not in fixed]
        if kept and kept != list(range(kept[0], len(operand.modes))):
            raise ValueError("native row projection requires leading fixed axes")
        operands.append(
            replace(
                operand,
                modes=tuple(operand.modes[i] for i in kept),
                shape=tuple(operand.shape[i] for i in kept),
                strides=None
                if operand.strides is None
                else tuple(operand.strides[i] for i in kept),
            )
        )
    output = operands[-1]
    semantics = dict(request.semantics)
    extents = {
        mode: n for op in operands for mode, n in zip(op.modes, op.shape, strict=True)
    }
    semantics.update(
        {
            "parent_semantic_identity": request.semantic_identity,
            "fixed_modes": json.dumps(sorted(fixed)),
            "operand_order": json.dumps(operand_order),
            "output_elements": prod(output.shape),
            "reduction_extent": prod(
                n for mode, n in extents.items() if mode not in output.modes
            ),
        }
    )
    return replace(
        request,
        operands=tuple(operands),
        shape=output.shape,
        input_dtypes=tuple(request.input_dtypes[i] for i in operand_order),
        precisions=tuple(
            replace(p, input_dtypes=tuple(p.input_dtypes[i] for i in operand_order))
            for p in request.precisions
        ),
        semantics=tuple(semantics.items()),
    )


def affine_contraction_initializer(
    adapter: TensorLoweringAdapter,
    node: Node,
    dimension: Callable[[Index], str],
    *,
    coefficient: str,
    beta: str = "0.0",
    fixed_modes: tuple[int, ...] = (),
    operand_order: tuple[int, int] = (0, 1),
) -> str:
    """Project the original binary axes without requiring a GEMM factorization.

    Zero matrix dimensions deliberately prevent accidental matrix execution.
    General providers validate the same descriptor's affine semantic fields;
    the matrix provider additionally requires its existing physical recipe.
    """
    return contraction_initializer(
        adapter,
        node,
        dimension,
        transpose=("N", "N"),
        extents=("1", "0", "0", "0"),
        coefficient=coefficient,
        beta=beta,
        fixed_modes=fixed_modes,
        operand_order=operand_order,
    )


def contraction_initializer(
    adapter: TensorLoweringAdapter,
    node: Node,
    dimension: Callable[[Index], str],
    *,
    transpose: tuple[str, str],
    extents: tuple[str, str, str, str],
    coefficient: str,
    row_axes: tuple[int, int, int] | None = None,
    leading_dimensions: tuple[str, str, str] | None = None,
    beta: str = "0.0",
    accumulation: Node | None = None,
    batch_scale: Node | None = None,
    checked_update: Program | None = None,
    checked_publication: Program | None = None,
    checked_right_symmetrization: tuple[str, str, str] | None = None,
    checked_pair: CheckedTransposePair | None = None,
    fixed_modes: tuple[int, ...] = (),
    operand_order: tuple[int, int] = (0, 1),
) -> str:
    """Emit a typed descriptor for an already recognized binary matrix einsum.

    Optional row cuts/strides describe unbatched affine views; beta can also
    update a dense output. `accumulation` additionally binds the existing SSA
    einsum-plus-input region and its donated seed, rather than hiding beta in a
    binary operation identity. Native validation checks the physical recipe against
    the original semantic modes before execution.

    A checked symmetric RHS additionally binds the original sum/transpose region
    and scalar input roles. It borrows one square matrix twice, preserving the
    original helper's half-factor order instead of materializing a new matrix.

    Matrix recognition belongs to the existing physical lowerer. The descriptor
    retains original mode labels and operand shapes, so provider execution can
    validate its matrix recipe against that same semantic request. No provider
    or execution precision choice is made by a scientific/method owner here.
    """
    if node.op != "einsum" or len(node.inputs) != 2:
        raise ValueError("native contraction projection requires binary einsum")
    if accumulation is not None and batch_scale is not None:
        raise ValueError("native region cannot combine donation and batch weighting")
    if batch_scale is not None:
        from .batch_scaled_contraction import batch_scaled_contraction_request

        if fixed_modes or operand_order != (0, 1) or beta != "0.0":
            raise ValueError(
                "native batch scale cannot project, reorder or donate operands"
            )
        if node not in batch_scale.inputs:
            raise ValueError(
                "native batch scale must retain its contraction intermediate"
            )
        request = batch_scaled_contraction_request(adapter, batch_scale, backend="cuda")
    elif accumulation is None:
        request = projected_contraction_request(
            adapter, node, fixed_modes=fixed_modes, operand_order=operand_order
        )
    else:
        from .contraction_update import contraction_update_request

        if fixed_modes or operand_order != (0, 1):
            raise ValueError("native update cannot project or reorder donated operands")
        if node not in accumulation.inputs or beta != "1.0":
            raise ValueError("native update must preserve its unit seed contribution")
        request = contraction_update_request(adapter, accumulation, backend="cuda")
    pair_role = 0
    if checked_pair is not None:
        from .checked_contraction_pair import checked_transpose_pair_request

        pair_role = checked_pair.role(node)
        if (
            checked_update is None
            or checked_publication is not None
            or checked_right_symmetrization is not None
            or accumulation is not None
            or batch_scale is not None
            or fixed_modes
            or operand_order != (0, 1)
            or beta != "0.0"
            or coefficient != "1.0"
            or transpose != (("N" if pair_role == 1 else "T"), "N")
            or row_axes is not None
        ):
            raise ValueError(
                "checked pair requires its complete fresh transpose recipe"
            )
        request = checked_transpose_pair_request(
            adapter, checked_pair, checked_update, role=pair_role, backend="cuda"
        )
    elif checked_right_symmetrization:
        from .checked_contraction import checked_symmetric_right_request

        if (
            checked_update is None
            or checked_publication is not None
            or accumulation is not None
            or batch_scale is not None
            or fixed_modes
            or operand_order != (0, 1)
            or beta != "0.0"
            or coefficient != "0.5"
            or transpose != ("N", "N")
        ):
            raise ValueError(
                "checked symmetric RHS requires its complete fresh half-scaled recipe"
            )
        request = checked_symmetric_right_request(
            adapter,
            node,
            checked_update,
            scalar_inputs=checked_right_symmetrization,
            backend="cuda",
        )
    elif checked_update is not None:
        from .checked_contraction import checked_contraction_request

        if accumulation is not None or beta != "0.0" or coefficient != "1.0":
            raise ValueError("checked scalar contraction requires a fresh unit result")
        request = checked_contraction_request(
            request, checked_update, checked_publication, contraction=node
        )
    elif checked_publication is not None:
        raise ValueError("checked publication requires its scalar update")
    precision = request.precisions[0]
    directive = precision.directive
    if (
        len(request.precisions) != 1
        or precision.casts
        or precision.refinement
        or precision.audit
        or len({*precision.input_dtypes, precision.publication_dtype}) != 1
        or directive.storage_dtype != directive.compute_dtype
        or directive.compute_dtype != directive.accumulation_dtype
    ):
        raise ValueError(
            "native dense candidate does not implement requested precision"
        )

    def dtype(name: str) -> str:
        return (
            "generativeqc::runtime::PrecisionDtype::"
            + {
                "float64": "Fp64",
                "float32": "Fp32",
            }[name]
        )

    operands = []
    if (row_axes is None) != (leading_dimensions is None):
        raise ValueError("matrix view cuts and strides must be supplied together")
    # An update's third read is the donated output itself. Keep its original
    # three physical matrix views while binding the complete SSA region identity.
    layouts = (*request.operands[:2], request.operands[-1])
    for i, (value, layout) in enumerate(
        zip(
            (*(node.inputs[i] for i in operand_order), node),
            layouts,
            strict=True,
        )
    ):
        modes = ",".join(map(str, layout.modes))
        original_labels = (
            node.attrs["labels"][operand_order[i]] if i < 2 else node.attrs["output"]
        )
        shape = ",".join(
            dimension(index)
            for mode, index in zip(original_labels, value.spec.indices, strict=True)
            if mode not in fixed_modes
        )
        view = "matrix_view" if row_axes is not None else "dense"
        extra = ""
        if row_axes is not None:
            assert leading_dimensions is not None
            extra = f",{row_axes[i]},{leading_dimensions[i]}"
        operands.append(
            f"generativeqc::tensor::ContractionOperand::{view}("
            f"{{{modes}}},{{{shape}}},{dtype(value.spec.dtype)}{extra})"
        )
    arithmetic = ",".join(
        (
            dtype(directive.storage_dtype),
            dtype(directive.compute_dtype),
            dtype(directive.accumulation_dtype),
            json.dumps(directive.qualification or ""),
            json.dumps(directive.math_mode),
        )
    )
    return (
        "generativeqc::tensor::ContractionRequest{"
        f'"{request.scientific_identity}","{request.semantic_identity}",'
        f'"{precision.identity}",'
        "{" + ",".join(operands) + "},"
        "{"
        + arithmetic
        + "},"
        + dtype(precision.publication_dtype)
        + f",'{transpose[0]}','{transpose[1]}',"
        + ",".join((*extents, coefficient))
        + (
            ",{" + ",".join(leading_dimensions or ()) + "}," + beta
            if leading_dimensions is not None
            or beta != "0.0"
            or checked_update is not None
            else ""
        )
        + (
            ","
            + json.dumps(checked_update.logical_hash)
            + ","
            + json.dumps(
                checked_publication.logical_hash if checked_publication else ""
            )
            if checked_update is not None
            else ""
        )
        + (",true" if checked_right_symmetrization else "")
        + (f",false,{pair_role}" if pair_role else "")
        + "}"
    )
