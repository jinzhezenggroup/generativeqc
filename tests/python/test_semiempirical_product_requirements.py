"""CPU-only product/derivative requirement contracts for issue #875."""

from dataclasses import replace

import pytest

from generativeqc_compiler.method.semiempirical import (
    InvalidSemiempiricalMethod,
    ParameterResource,
    ParameterSetRef,
    PrimitiveNode,
    ProductSpec,
    SemiempiricalMethodIR,
    StateField,
    semiempirical_from_xtb,
)
from generativeqc_compiler.method.semiempirical_requirements import (
    DerivativeMechanism,
    semiempirical_product_requirements,
)
from generativeqc_compiler.method.xtb import (
    GFN2_PARAMETER_SET,
    XtbMethodSpec,
    resolve_xtb_method,
)


def _generic_method(family: str = "dftb") -> SemiempiricalMethodIR:
    """Representation fixture only: no claim of DFTB or NDDO execution."""
    dftb = family == "dftb"
    state = StateField(
        "charge" if dftb else "density_matrix", "atom" if dftb else "ao-pair"
    )
    resource = ParameterResource(
        "radial:slater-koster" if dftb else "integral:nddo-one-center",
        "sha256:bound-v1",
    )
    nodes = (
        PrimitiveNode(
            "integrals",
            "integral",
            "dftb.slater-koster" if dftb else "nddo.one-center",
            produces=("one-electron-operator",),
            parameter_resources=(resource.role,),
            derivative_capabilities=("integral-adjoint",),
        ),
        PrimitiveNode(
            "interaction",
            "interaction",
            "dftb.gamma-scc" if dftb else "nddo.two-center",
            requires=("integrals",),
            state_reads=(state.name,),
            state_writes=(state.name,),
            derivative_capabilities=("state-adjoint",),
            fixed_point=True,
        ),
    )
    return SemiempiricalMethodIR(
        family=family,
        model="test-only",
        reference="restricted",
        parameter_set=ParameterSetRef(
            identifier="audited-schematic",
            schema="fixture-v1",
            source_identity="sha256:fixture",
            supported_atomic_numbers=(8, 1),
            resources=(resource,),
        ),
        state_fields=(state,),
        primitives=nodes,
        requested_products=(
            ProductSpec("energy", 0),
            ProductSpec("nuclear-gradient", 1),
        ),
    )


@pytest.mark.parametrize("name", ["GFN1-xTB", "GFN2-xTB"])
def test_real_gfn_graph_binds_parameter_and_state_requirements(name: str) -> None:
    legacy = resolve_xtb_method(
        name, requested_products=("energy", "nuclear-gradient")
    )
    common = semiempirical_from_xtb(legacy)
    energy = semiempirical_product_requirements(common, ProductSpec("energy", 0))
    gradient = semiempirical_product_requirements(
        common, ProductSpec("nuclear-gradient", 1)
    )
    assert energy.method_identity == common.identity
    assert gradient.method_identity == common.identity
    assert energy.has_closed_derivative_contract  # no derivative requested
    assert gradient.unresolved_global_obligations[-1] == "total-gradient-assembly"
    assert "stationary-state-response" in gradient.unresolved_global_obligations
    assert not gradient.has_closed_derivative_contract
    with pytest.raises(InvalidSemiempiricalMethod, match="unresolved derivative"):
        gradient.require_closed_derivative_contract()

    original = {node.node_id: node for node in common.primitives}
    parameters = {p.role: p for p in common.parameter_set.resources}
    states = {s.name: s for s in common.state_fields}
    assert {n.node_id for n in gradient.nodes} == set(original)
    for bound in gradient.nodes:
        primitive = original[bound.node_id]
        assert bound.depends_on == primitive.requires
        assert bound.equation_identity == primitive.equation_identity
        assert bound.parameter_resources == tuple(
            parameters[k] for k in primitive.parameter_resources
        )
        assert bound.state_reads == tuple(states[k] for k in primitive.state_reads)
        assert bound.state_writes == tuple(states[k] for k in primitive.state_writes)
        assert bound.fixed_point == primitive.fixed_point
        if "integral-adjoint" in primitive.derivative_capabilities:
            assert DerivativeMechanism.INTEGRAL_ADJOINT in (
                bound.derivative_mechanisms
            )
    assert (
        gradient.supported_atomic_numbers
        == legacy.parameter_set.supported_atomic_numbers
    )
    assert "runtime_executable" not in str(gradient.to_payload())


def test_gfn_alias_identity_and_parameter_revision_do_not_drift() -> None:
    left = semiempirical_from_xtb(
        resolve_xtb_method(XtbMethodSpec("alias-a", "gfn2", GFN2_PARAMETER_SET))
    )
    right = semiempirical_from_xtb(
        resolve_xtb_method(XtbMethodSpec("alias-b", "gfn2", GFN2_PARAMETER_SET))
    )
    product = ProductSpec("energy", 0)
    assert left.identity == right.identity
    assert semiempirical_product_requirements(
        left, product
    ).identity == semiempirical_product_requirements(right, product).identity

    changed = semiempirical_from_xtb(
        resolve_xtb_method(
            XtbMethodSpec(
                "alias-a", "gfn2",
                replace(GFN2_PARAMETER_SET, revision="new-audited-content"),
            )
        )
    )
    assert left.identity != changed.identity
    assert semiempirical_product_requirements(
        left, product
    ).identity != semiempirical_product_requirements(changed, product).identity


@pytest.mark.parametrize("family", ["dftb", "nddo"])
def test_common_requirements_have_no_gfn_specific_branches(family: str) -> None:
    method = _generic_method(family)
    reordered = replace(method, primitives=tuple(reversed(method.primitives)))
    product = ProductSpec("nuclear-gradient", 1)
    first = semiempirical_product_requirements(method, product)
    second = semiempirical_product_requirements(reordered, product)
    assert first.identity == second.identity
    assert first.nodes[0].parameter_resources[0] == method.parameter_set.resources[0]
    assert first.nodes[1].state_writes == method.state_fields
    assert first.nodes[0].derivative_mechanisms == (
        DerivativeMechanism.INTEGRAL_ADJOINT,
    )
    assert not first.has_closed_derivative_contract
    assert first.unresolved_global_obligations == (
        "stationary-state-response",
        "total-gradient-assembly",
    )


def test_semantic_resource_and_state_changes_invalidate_requirement_identity() -> None:
    method = _generic_method()
    product = ProductSpec("energy", 0)
    baseline = semiempirical_product_requirements(method, product)
    changed_parameters = replace(
        method,
        parameter_set=replace(
            method.parameter_set,
            resources=(
                replace(method.parameter_set.resources[0], identity="sha256:v2"),
            ),
        ),
    )
    changed_state = replace(
        method,
        state_fields=(replace(method.state_fields[0], scope="shell"),),
    )
    assert baseline.identity != semiempirical_product_requirements(
        changed_parameters, product
    ).identity
    assert baseline.identity != semiempirical_product_requirements(
        changed_state, product
    ).identity


def test_product_request_and_derivative_semantics_fail_closed() -> None:
    method = _generic_method()
    with pytest.raises(InvalidSemiempiricalMethod, match="not requested"):
        semiempirical_product_requirements(method, ProductSpec("dipole", 0))

    unsupported = replace(
        method, requested_products=(ProductSpec("hessian", 2),)
    )
    with pytest.raises(InvalidSemiempiricalMethod, match="unsupported"):
        semiempirical_product_requirements(unsupported, ProductSpec("hessian", 2))

    bad_order = replace(
        method, requested_products=(ProductSpec("nuclear-gradient", 0),)
    )
    with pytest.raises(InvalidSemiempiricalMethod, match="unsupported"):
        semiempirical_product_requirements(
            bad_order, ProductSpec("nuclear-gradient", 0)
        )

    no_energy = replace(
        method, requested_products=(ProductSpec("nuclear-gradient", 1),)
    )
    with pytest.raises(InvalidSemiempiricalMethod, match="requires the total energy"):
        semiempirical_product_requirements(
            no_energy, ProductSpec("nuclear-gradient", 1)
        )

    with pytest.raises(TypeError, match="typed ProductSpec"):
        semiempirical_product_requirements(method, "energy")  # type: ignore[arg-type]


def test_unrecognized_derivative_and_ambiguous_writers_fail_closed() -> None:
    method = _generic_method()
    unknown = replace(
        method,
        primitives=(
            replace(method.primitives[0], derivative_capabilities=("magic-gradient",)),
            method.primitives[1],
        ),
    )
    with pytest.raises(InvalidSemiempiricalMethod, match="unrecognized derivative"):
        semiempirical_product_requirements(unknown, ProductSpec("energy", 0))

    # Two different primitives cannot each claim to write the same fixed-point
    # state without an explicit owner/composition node.
    conflicting = replace(
        method,
        primitives=(
            replace(method.primitives[0], state_writes=("charge",)),
            method.primitives[1],
        ),
    )
    with pytest.raises(InvalidSemiempiricalMethod, match="ambiguous writer"):
        semiempirical_product_requirements(conflicting, ProductSpec("energy", 0))


def test_local_derivative_tags_never_imply_a_complete_total_force() -> None:
    method = _generic_method()
    all_local = replace(
        method,
        primitives=tuple(
            replace(n, derivative_capabilities=("nuclear-gradient",))
            for n in method.primitives
        ),
    )
    requirements = semiempirical_product_requirements(
        all_local, ProductSpec("nuclear-gradient", 1)
    )
    assert requirements.unresolved_local_derivative_nodes == ()
    assert not requirements.has_closed_derivative_contract
    with pytest.raises(InvalidSemiempiricalMethod, match="total-gradient-assembly"):
        requirements.require_closed_derivative_contract()
