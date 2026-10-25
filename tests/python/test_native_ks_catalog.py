"""Shared native catalog programs, immutable identities and explicit routes."""

import ctypes
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import Calculator, GridSpec, _native
from generativeqc.ks import KsOptions, native_ks_options, resolve_ks_options
from generativeqc_compiler.method import (
    ExactExchangePrimitive,
    compile_ks_execution_plan,
    resolve_method,
)
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks

from tools.generate_cli_method_catalog import rows


@pytest.mark.parametrize(
    "name",
    [
        "PBE-D3(BJ)",
        "PBE-D3(0)",
        "PBE-D3(BJ)-ATM",
        "PBE0-D4(BJ-EEQ-ATM)",
        "B3LYP-D4(BJ-EEQ-ATM)",
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_native_composition_owns_correction_and_result_lifetime(
    library: Any,
    name: str,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from generativeqc import dispersion

    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a Slurm-allocated CUDA device")
        assert os.environ.get("SLURM_JOB_ID"), (
            "CUDA qualification must run through srun"
        )
    graph = resolve_method(name)
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(
        method=graph, device=device, ks_options=KsOptions(grid=GridSpec(8, 4, 8))
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("complete native program delegated correction execution to Python")

    monkeypatch.setattr(dispersion, "D3CorrectionBatch", forbidden)
    monkeypatch.setattr(dispersion, "D4CorrectionBatch", forbidden)
    with calculator.prepare_batch([atoms], warm_start=False) as batch:
        full = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        assert full.dispersion is not None and full.dispersion.gradient is not None
        assert batch.dispersion_diagnostic.method_ir_identity == graph.identity
        energy = batch.execute(strict=True, properties=("energy",)).items[0]
        assert energy.energy == pytest.approx(full.energy, abs=1e-12)
        assert energy.dispersion is not None and energy.dispersion.gradient is None
        assert energy.forces is None
        bad = batch.execute([np.full((2, 3), np.nan)], properties=("energy",)).items[0]
        assert not bad.succeeded and bad.dispersion is None
        # A failed replay must also revoke the separately queried receipt.
        query = library.generativeqc_ks_batch_dispersion_result_v1
        out = _native.D4BatchItemResultDescriptor(
            ctypes.sizeof(_native.D4BatchItemResultDescriptor), _native.ABI_VERSION
        )
        assert (
            query(batch._batch, 0, ctypes.byref(out)) == _native.STATUS_NOT_IMPLEMENTED
        )
        recovered = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        np.testing.assert_allclose(recovered.forces, full.forces, atol=1e-11, rtol=0)


def test_complete_native_budget_refusal_never_retries_external_owner(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import dispersion

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a native capacity refusal retried external correction execution")

    monkeypatch.setattr(dispersion, "D3CorrectionBatch", forbidden)
    calculator = Calculator(
        method=resolve_method("PBE-D3(BJ)"),
        ks_options=KsOptions(grid=GridSpec(8, 4, 8)),
        dispersion_memory_budget_bytes=1,
    )
    with pytest.raises(RuntimeError):
        calculator.singlepoint(
            [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))], properties=("energy",)
        )


@pytest.mark.parametrize(
    "name,electronic",
    [("PBE-D3(BJ)-ATM", "pbe-rks"), ("PBE0-D4(BJ-EEQ-ATM)", "pbe0-rks")],
)
def test_complete_native_triatomic_keeps_nonzero_atm(
    library: Any, name: str, electronic: str
) -> None:
    """Check parameter transport/composition where the three-body term is nonzero.

    Standalone correction providers have separate independent reference gates;
    here their full gradient must enter the native molecular endpoint once.
    """
    import numpy as np
    from generativeqc import evaluate_d3_correction, evaluate_d4_correction

    numbers = [8, 1, 1]
    positions = np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (-0.45, 1.75, 0.0)])
    atoms = list(zip(numbers, positions, strict=True))
    options = KsOptions(grid=GridSpec(12, 6, 12))
    plain = Calculator(method=electronic, ks_options=options).singlepoint(
        atoms, properties=("energy", "forces")
    )
    evaluate = evaluate_d4_correction if "D4" in name else evaluate_d3_correction
    correction = evaluate(name, numbers, positions)
    if "D4" in name:
        assert abs(correction.atm_energy) > 1e-12
    else:
        pair = evaluate_d3_correction("PBE-D3(BJ)", numbers, positions)
        assert abs(correction.energy - pair.energy) > 1e-12
    with Calculator(method=resolve_method(name), ks_options=options).prepare_batch(
        [atoms]
    ) as batch:
        complete = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        assert complete.dispersion.energy == pytest.approx(correction.energy, abs=2e-13)
        assert complete.energy == pytest.approx(
            plain.energy + correction.energy, abs=2e-11
        )
        np.testing.assert_allclose(
            complete.forces, plain.forces - correction.gradient, atol=3e-10, rtol=0
        )


@pytest.fixture(scope="module")
def library() -> Any:
    selected = _native.load_library()
    assert hasattr(selected, "generativeqc_ks_catalog_plan_v1")
    return selected


def program(descriptor: Any) -> tuple:
    """Detach the scientific wire while deliberately ignoring pointer identity."""
    return (
        descriptor.scf_domain,
        descriptor.spin_channels,
        descriptor.semilocal_range_omega,
        tuple(
            (
                descriptor.semilocal_components[index].component_id.decode(),
                descriptor.semilocal_components[index].coefficient,
            )
            for index in range(descriptor.semilocal_component_count)
        ),
        tuple(
            (
                descriptor.exchange_terms[index].operator_kind,
                descriptor.exchange_terms[index].coefficient,
                descriptor.exchange_terms[index].omega,
                descriptor.exchange_terms[index].fock_coefficient,
            )
            for index in range(descriptor.exchange_term_count)
        ),
        descriptor.has_nonlocal_correlation,
        descriptor.nonlocal_variant,
        descriptor.nonlocal_b,
        descriptor.nonlocal_c,
        descriptor.nonlocal_coefficient,
        descriptor.nonlocal_maximum_bytes,
    )


def test_entire_catalog_against_independent_compiler_programs(library: Any) -> None:
    query = library.generativeqc_ks_catalog_plan_v1
    query.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(_native.KsOptionsDescriptor),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query.restype = ctypes.c_int
    qualified = 0
    for row in rows():
        for backend, eligible in (
            (_native.BACKEND_CPU_REFERENCE, row.cpu),
            (_native.BACKEND_CUDA, row.cuda),
        ):
            output = _native.KsOptionsDescriptor(
                ctypes.sizeof(_native.KsOptionsDescriptor), _native.ABI_VERSION
            )
            before = bytes(output)
            selected = ctypes.c_uint32(9)
            name = row.name.encode()
            assert (
                query(
                    name,
                    len(name),
                    None,
                    0,
                    backend,
                    ctypes.byref(output),
                    ctypes.byref(selected),
                )
                == 0
            ), row.name
            # The prefix-only API must not publish a truncated electronic
            # program for a corrected method. Named preparation owns those.
            electronic_only = eligible and not row.dispersion[0]
            assert selected.value == int(electronic_only), (row.name, backend)
            if not electronic_only:
                assert bytes(output) == before
                continue
            spin = "unpolarized" if row.spin == 1 else "polarized"
            if row.name.startswith(("libxc:", "libxc-uks:")):
                method = resolve_bulk_ks(
                    row.name.split(":", 1)[1].upper(), spin=spin
                ).method
            else:
                method = resolve_method(row.source, spin=spin)
            reference = compile_ks_execution_plan(method)
            expected_exchange = tuple(
                (
                    {"full-range": 1, "short-range": 2, "long-range": 3}[term.operator],
                    float(term.coefficient),
                    float(term.omega),
                    float(term.fock_coefficient),
                )
                for term in reference.exchange
            )
            assert program(output) == (
                row.domain.encode(),
                row.spin,
                float(reference.semilocal.functional.range_omega),
                tuple(
                    (name, float(coefficient))
                    for name, coefficient in reference.semilocal.functional.components
                ),
                expected_exchange,
                int(reference.nonlocal_correlation is not None),
                *row.nonlocal_correlation,
                0,
            ), (row.name, backend)
            qualified += 1
    assert qualified > 20


@pytest.mark.parametrize(
    "name",
    [
        "lda-rks",
        "pbe-rks",
        "pbe-uks",
        "pbe0-rks",
        "pbe0-uks",
        "b3lyp-rks",
        "b3lyp-uks",
        "wb97m-v-rks",
        "wb97m-v-uks",
    ],
)
def test_named_runtime_skips_python_scientific_lowering(
    library: Any, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import ks

    options = resolve_ks_options(name, KsOptions(grid=GridSpec(8, 6, 12)))
    portable = native_ks_options(options)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("catalog-selected program fell back to Python scientific lowering")

    monkeypatch.setattr(ks, "_native_execution_plan", forbidden)
    selected = native_ks_options(options, library=library)
    assert selected._catalog_library_owner is library
    assert program(selected) == program(portable)
    assert selected.radial_points == portable.radial_points
    assert selected.angular_polar == portable.angular_polar
    assert selected.angular_azimuth == portable.angular_azimuth


def test_same_name_custom_graph_remains_exact_explicit_compiler_work(
    library: Any,
) -> None:
    graph = resolve_method("PBE0")
    custom = replace(
        graph,
        primitives=tuple(
            replace(node, coefficient=node.coefficient / 2)
            if isinstance(node, ExactExchangePrimitive)
            else node
            for node in graph.primitives
        ),
    )
    assert custom.identifier == graph.identifier and custom.identity != graph.identity
    options = resolve_ks_options(
        "pbe0-rks", KsOptions(composition=custom, grid=GridSpec(8, 6, 12))
    )
    selected = native_ks_options(options, library=library)
    portable = native_ks_options(options)
    assert not hasattr(selected, "_catalog_library_owner")
    assert program(selected) == program(portable)


def test_older_library_keeps_explicit_portable_plan() -> None:
    options = resolve_ks_options("pbe0-uks", KsOptions(grid=GridSpec(8, 6, 12)))
    assert program(native_ks_options(options, library=SimpleNamespace())) == program(
        native_ks_options(options)
    )


def test_native_refusal_never_retries_compiler_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import ks

    options = resolve_ks_options("pbe-rks")

    class Refusal:
        def __call__(self, *_args: Any) -> int:
            return _native.STATUS_INVALID_ARGUMENT

    def forbidden(*_args: Any) -> None:
        pytest.fail("native refusal retried compiler lowering")

    monkeypatch.setattr(ks, "_native_execution_plan", forbidden)
    with pytest.raises(RuntimeError, match="selected catalog refusal"):
        native_ks_options(
            options,
            library=SimpleNamespace(
                generativeqc_ks_catalog_plan_v1=Refusal(),
                generativeqc_status_message=lambda _status: b"selected catalog refusal",
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "struct_size",
        "abi_version",
        "tile_points",
        "application_energy_budget",
        "selection",
    ],
)
def test_selected_publication_cannot_change_caller_controls(
    library: Any, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from generativeqc import ks

    class CorruptPublication:
        def __call__(self, *args: Any) -> int:
            status = library.generativeqc_ks_catalog_plan_v1(*args)
            assert status == 0
            if field == "selection":
                ctypes.cast(
                    args[-1], ctypes.POINTER(ctypes.c_uint32)
                ).contents.value = 2
            else:
                descriptor = ctypes.cast(
                    args[-2],
                    ctypes.POINTER(_native.ApplicationPrecisionKsOptionsDescriptor),
                ).contents
                setattr(descriptor, field, getattr(descriptor, field) + 1)
            return status

    def forbidden(*_args: Any) -> None:
        pytest.fail("inconsistent publication retried compiler lowering")

    options = resolve_ks_options(
        "pbe-rks", KsOptions(_precision_budget=(1e-7, 2e-7, 3e-7))
    )
    monkeypatch.setattr(ks, "_native_execution_plan", forbidden)
    with pytest.raises(RuntimeError, match="native KS catalog"):
        native_ks_options(
            options,
            library=SimpleNamespace(
                generativeqc_ks_catalog_plan_v1=CorruptPublication()
            ),
        )


def test_unselected_publication_preserves_the_complete_extension() -> None:
    class CorruptExtension:
        def __call__(self, *args: Any) -> int:
            ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_uint32)).contents.value = 0
            descriptor = ctypes.cast(
                args[-2],
                ctypes.POINTER(_native.ApplicationPrecisionKsOptionsDescriptor),
            ).contents
            descriptor.application_energy_budget = 1.0
            return 0

    options = resolve_ks_options(
        "pbe-rks", KsOptions(_precision_budget=(1e-7, 2e-7, 3e-7))
    )
    with pytest.raises(RuntimeError, match="unselected native KS catalog plan changed"):
        native_ks_options(
            options,
            library=SimpleNamespace(generativeqc_ks_catalog_plan_v1=CorruptExtension()),
        )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks", "pbe0-rks", "b3lyp-rks"])
def test_real_calculator_and_budgeted_prepare_use_selected_owner(
    library: Any, method: str
) -> None:
    from generativeqc import ResourceBudget

    calculator = Calculator(
        method=method,
        resource_budget=ResourceBudget(host_bytes=512 << 20),
        ks_options=KsOptions(grid=GridSpec(8, 6, 12)),
        initial_guess=None,
    )
    descriptor = calculator._method_descriptor()
    assert descriptor.ks_options.contents.semilocal_component_count > 0
    molecule = (("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7)))
    multiplicity = 2 if method.endswith("uks") else 1
    charge = 1 if multiplicity == 2 else 0
    with calculator.prepare_batch(
        [molecule], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        result = batch.execute(properties=("energy",)).items[0]
        assert result.converged


def test_precision_extension_and_controls_survive_native_selection(
    library: Any,
) -> None:
    options = resolve_ks_options(
        "pbe-rks",
        KsOptions(
            _precision_budget=(1e-7, 2e-7, 3e-7),
            tile_points=37,
        ),
    )
    descriptor = native_ks_options(
        options, library=library, backend=_native.BACKEND_CUDA
    )
    owner = descriptor._extension_owner
    assert owner._catalog_library_owner is library
    assert (
        owner.application_energy_budget,
        owner.application_max_force_budget,
        owner.application_rms_force_budget,
    ) == (1e-7, 2e-7, 3e-7)
    assert owner.tile_points == 37
