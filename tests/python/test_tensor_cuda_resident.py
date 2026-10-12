"""CPU proofs of the resident ABI contract emitted on top of a verified plan.

These checks need no GPU and no NVCC: they assert the generated translation
unit keeps the ordinary host-staged entry points intact, pins exactly the
planned input/output offsets, and propagates a non-zero ordinary status
instead of swallowing it. Real-device numerical parity is a separate opt-in
suite.
"""

import typing
from pathlib import Path

import pytest

from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    input_tensor,
    multiply,
)
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.cuda_resident import (
    DeviceTensor,
    _check_lease,
    _resident_dependency_logical_name,
)
from generativeqc_compiler.tensor.cuda_resident_emit import resident_source

TARGET = cuda_target_info("sm_80")


def doubled_pair_program() -> typing.Any:
    """Two inputs, two outputs: one elementwise and one symmmetric double."""
    occupied = IndexSpace("occupied", "occupied", 2)
    virtual = IndexSpace("virtual", "virtual", 3)
    i, j = Index("i", occupied), Index("j", occupied)
    a, b = Index("a", virtual), Index("b", virtual)
    common = {
        "representation": "restricted_spatial",
        "role": "parameter",
        "differentiable": True,
    }
    singles = TensorSpec((i, a), **common)
    doubles = TensorSpec((i, j, a, b), symmetries=(Symmetry((1, 0, 3, 2)),), **common)
    t1 = input_tensor("t1", singles)
    t2 = input_tensor("t2", doubles)
    return Program({"squared": add(multiply(t1, t1)), "copy": add(t2)})


def test_resident_source_keeps_the_ordinary_abi() -> None:
    """A resident artifact must remain a valid host-staged artifact."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for ordinary in (
        "tensor_plan_identity",
        "tensor_create",
        "tensor_destroy",
        "tensor_run",
        "tensor_probe",
    ):
        assert ordinary in source, ordinary
    assert "tensor_static_initialize" not in source

    external = resident_source(plan, embed_static_data=False)
    assert "tensor_static_bytes" in external
    assert "tensor_static_initialize" in external
    assert "tensor static data is not initialized" in external


def test_resident_source_declares_the_resident_abi() -> None:
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for entry in (
        "resident_plan_identity",
        "resident_abi",
        "resident_upload",
        "resident_run",
        "resident_download",
    ):
        assert entry in source, entry
    assert "return 1;" in source


def test_resident_spans_match_the_pinned_plan_offsets() -> None:
    """Spans are compiled from pinned offsets; the planner must pin them."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    assert all(plan.steps[i].last_use == len(plan.steps) for i in plan.inputs)
    source = resident_source(plan)
    for i in plan.inputs:
        step = plan.steps[i]
        assert f"{{{step.offset}ULL,{step.node.spec.size * 8}ULL}}" in source
    for _, i in plan.outputs:
        step = plan.steps[i]
        assert f"{{{step.offset}ULL,{step.node.spec.size * 8}ULL}}" in source


def test_resident_run_propagates_the_ordinary_status() -> None:
    """A native non-finite/division error must not be reported as success.

    The resident path inlines the generated launch sequence directly
    (without the ordinary H2D/D2H staging copies), so it does not shell
    out to ``tensor_run``.  The inline error boundary is identical: it
    reads ``arithmetic_error`` from the device, throws when non-zero, and
    the ``catch`` block synchronises the stream before returning through
    ``generativeqc_tensor::error_text``.
    """
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    assert "arithmetic_error" in source
    assert "generativeqc_tensor::error_text(error, size, e.what()); return 1;" in source
    assert "try {" in source
    assert "throw std::runtime_error" in source


def test_resident_validation_covers_every_input_symmetry() -> None:
    """Uploads must be checked like the ordinary path, before any kernel."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for slot in range(len(plan.inputs)):
        assert f"resident_validate_{slot}" in source
    # The doubles input declares simultaneous pair exchange; its validation
    # must compare against the permuted partner with the ordinary tolerance.
    assert "1e-11 + 1e-10 * fabs(peer)" in source


def test_resident_requires_pinned_materialized_outputs() -> None:
    """Resident spans are only valid while the planner pins outputs.

    This is the invariant the whole design rests on: every output step must
    be materialized and live to the end of the program, so its arena offset
    is stable and can be re-used by a later run. ``resident_source`` rejects
    a plan that violates it instead of emitting a stale span.
    """
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    for _, i in plan.outputs:
        step = plan.steps[i]
        assert not step.virtual
        assert step.last_use == len(plan.steps)
    # The emitter agrees with the planner rather than re-deriving it.
    assert resident_source(plan)


def test_resident_source_rejects_an_unpinned_output() -> None:
    """The emitter must fail closed when an output is not pinned for the
    full plan lifetime — this is the emitter's own check, not the planner's.
    A future planner change that recycles an output's arena offset must be
    caught here, before a compiled resident binary reads overwritten storage."""
    from types import SimpleNamespace

    import pytest as _pytest

    class _MockStep:
        virtual = False
        last_use = 0  # not pinned

    plan = SimpleNamespace(
        steps=[_MockStep],
        inputs=[],
        outputs=[("unpinned", 0)],
    )
    with _pytest.raises(ValueError, match="pinned"):
        resident_source(plan)


def test_resident_extension_is_part_of_the_generated_source() -> None:
    """Any plan-specific post-run action must be visible in the identity."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    clean = resident_source(plan)
    extended = resident_source(plan, extension="// plan-specific tables\n")
    assert clean != extended
    assert "// plan-specific tables" in extended


def test_resident_run_without_extension_has_no_undefined_hook() -> None:
    """The default source must not reference an undefined post-run symbol."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    assert "generativeqc_resident_action" not in source
    assert "GENERATIVEQC_RESIDENT_POST_RUN" not in source


def test_resident_extension_post_run_is_wired_only_when_declared() -> None:
    """A declared post-run action is emitted and its status propagated."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    declared = resident_source(
        plan,
        extension="__GENERATIVEQC_RESIDENT_POST_RUN_DECL__\n"
        "int generativeqc_resident_post_run(void*, int, Metrics*, char*, size_t) "
        "{ return 0; }\n",
    )
    assert (
        "generativeqc_resident_post_run(pointer, profile, result, error, size)"
        in declared
    )
    assert "if (status) return status;" in declared


# ── lease invalidation and identity normalisation regressions ──


def test_lease_is_invalidated_by_generation_or_readiness() -> None:
    """A DeviceTensor lease must be rejected after its owner's generation
    advances or readiness is cleared — otherwise a stale download could
    read arena data that was already overwritten by a later run()."""
    import pytest as _pytest

    class _FakeOwner:
        _pointer = object()
        _ready = True
        _generation = 0

    owner = _FakeOwner()
    owner.plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    # generation mismatch

    lease = DeviceTensor(owner, "squared", 99)
    with _pytest.raises(RuntimeError, match="stale"):
        lease._step()
    # readiness cleared
    owner._generation = 1
    lease2 = DeviceTensor(owner, "squared", 1)
    lease2._step()  # ok
    owner._ready = False
    with _pytest.raises(RuntimeError, match="stale|closed"):
        lease2._step()


def test_download_before_run_or_after_invalidation_is_rejected() -> None:
    """``download(None, name=...)`` and stale-lease downloads must both be
    rejected — the arena is not safe to read before ``run()`` succeeds or
    after a later ``upload``/failed run invalidated ``_ready``."""
    import pytest as _pytest

    class _FakeOwner:
        _pointer = object()
        _ready = False
        _generation = 0
        _lock = None  # unused by the readiness path we test

    owner = _FakeOwner()
    owner.plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)

    # download(None, name=...) before run: _ready check in download must fire
    # before _lock is acquired — verify at _check_lease level
    with _pytest.raises(RuntimeError, match="stale"):
        _check_lease(owner, DeviceTensor(owner, "squared", 99))  # generation mismatch
    owner._ready = True
    _check_lease(owner, DeviceTensor(owner, "squared", 0))  # ok
    owner._ready = False
    with _pytest.raises(RuntimeError, match="stale|closed"):
        _check_lease(owner, DeviceTensor(owner, "squared", 0))
    # download(None, ...) rejects None via TypeError
    with _pytest.raises(TypeError, match="DeviceTensor"):
        _check_lease(owner, None)


def test_resident_dependency_keys_agree_for_checkout_and_installed_wheel(
    tmp_path: Path,
) -> None:
    """Actual owned paths get the same cache key regardless of install prefix."""
    checkout = tmp_path / "checkout"
    package = tmp_path / "wheel" / "generativeqc_compiler"
    for checkout_name, installed_name, expected in (
        (
            "python/generativeqc_compiler/tensor/cuda_resident_emit.py",
            "tensor/cuda_resident_emit.py",
            "python/generativeqc_compiler/tensor/cuda_resident_emit.py",
        ),
        (
            "src/tensor/cuda_resident.cuh",
            "assets/src/tensor/cuda_resident.cuh",
            "src/tensor/cuda_resident.cuh",
        ),
    ):
        original = checkout / checkout_name
        installed = package / installed_name
        for path in (original, installed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("owned dependency")
        assert (
            _resident_dependency_logical_name(
                original, package_root=package, checkout_root=checkout
            )
            == expected
        )
        assert (
            _resident_dependency_logical_name(
                installed, package_root=package, checkout_root=checkout
            )
            == expected
        )


def test_resident_dependency_names_match_real_checkout_or_wheel_assets() -> None:
    """The identity test runs on a CPU host; no NVCC or GPU is needed."""
    from generativeqc_compiler.common.paths import PACKAGE, asset_path

    assert _resident_dependency_logical_name(
        PACKAGE / "tensor/cuda_resident.py"
    ) == "python/generativeqc_compiler/tensor/cuda_resident.py"
    assert _resident_dependency_logical_name(
        asset_path("src/tensor/cuda_resident.cuh")
    ) == "src/tensor/cuda_resident.cuh"


def test_compile_resident_source_identity_rejects_external_dependencies(
    tmp_path: Path,
) -> None:
    """Refuse paths outside either owner, including prefixes and symlinks.

    Previously this test only inspected the spelling of compile_resident.
    Invoke its actual key-building policy so accidental weakening is detected.
    """
    checkout = tmp_path / "checkout"
    package = tmp_path / "wheel" / "generativeqc_compiler"
    checkout.mkdir()
    package.mkdir(parents=True)
    external = tmp_path / "checkout-sibling" / "unowned.hpp"
    external.parent.mkdir()
    external.write_text("external")
    kwargs = {"package_root": package, "checkout_root": checkout}
    for path in (
        external,
        checkout / ".." / "checkout-sibling" / "unowned.hpp",
    ):
        with pytest.raises(ValueError, match="outside both"):
            _resident_dependency_logical_name(path, **kwargs)

    escape = checkout / "escape.hpp"
    escape.symlink_to(external)
    with pytest.raises(ValueError, match="outside both"):
        _resident_dependency_logical_name(escape, **kwargs)


def test_installed_resident_rejects_unowned_dependency_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import paths

    package = tmp_path / "wheel" / "generativeqc_compiler"
    package.mkdir(parents=True)
    external = tmp_path / "external.hpp"
    external.write_text("external")

    def unavailable() -> Path:
        raise ValueError("no source checkout")

    monkeypatch.setattr(paths, "source_root", unavailable)
    with pytest.raises(ValueError, match="outside both"):
        _resident_dependency_logical_name(external, package_root=package)
