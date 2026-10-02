"""Explicit preliminary guesses preserve target identity and warm-state ownership."""

from __future__ import annotations

import ctypes
import json
import typing
from dataclasses import replace
from types import SimpleNamespace

import pytest
from generativeqc import Calculator, GridSpec, InitialGuessSpec, KsOptions, _native
from generativeqc.initial_guess import (
    require_initial_guess_library,
    with_initial_guess_resources,
)
from generativeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def test_immutable_typed_policy_and_native_snapshot() -> None:
    hf = InitialGuessSpec("hf")
    lda = InitialGuessSpec("lda")
    assert hf.grid is None and lda.grid == GridSpec(8, 6, 12)
    assert hf.max_iterations == 32
    for policy, kind in ((hf, 1), (lda, 2)):
        native = policy.native()
        assert native.struct_size == ctypes.sizeof(native)
        assert native.abi_version == _native.ABI_VERSION
        assert native.kind == kind
        assert native.maximum_numeric_bytes == 256 << 20
        assert json.loads(json.dumps(policy.to_payload()))["kind"] == policy.kind
    assert hf.native().radial_points == 0
    assert lda.native().radial_points == 8


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "auto"},
        {"kind": "pbe0"},
        {"max_iterations": 0},
        {"max_iterations": 65},
        {"max_iterations": True},
        {"diis_history": 17},
        {"energy_tolerance": float("nan")},
        {"density_tolerance": float("inf")},
        {"energy_tolerance": False},
        {"maximum_numeric_bytes": 0},
        {"maximum_numeric_bytes": 2**63},
        {"kind": "hf", "grid": GridSpec(8, 6, 12)},
        {"kind": "lda", "grid": False},
        {"kind": "lda", "grid": GridSpec(48, 16, 32)},
        {"kind": "lda", "grid": GridSpec(8, 6, 12, element_radii=((8, 2.0),))},
    ],
)
def test_invalid_or_unqualified_policy_is_rejected(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        InitialGuessSpec(**options)


def test_missing_native_feature_is_not_silently_ignored() -> None:
    with pytest.raises(NotImplementedError, match="lacks preliminary"):
        require_initial_guess_library(SimpleNamespace())


def request(name: str, size: int) -> ResourceRequest:
    identity = ResourceIdentity(
        "target-PBE0",
        name,
        "cpu",
        "fp64",
        '{"nao": 13}',
        ("energy",),
        '{"max_iterations": 100}',
    )
    return ResourceRequest(
        name,
        identity,
        (
            ResourceCandidate(
                name, "resident", (ResourceEstimate(name, size, "pageable", 0, 0),)
            ),
        ),
    )


def test_global_budget_includes_both_live_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import resources_hf

    seen = []

    def guess_request(systems: typing.Any, **options: typing.Any) -> ResourceRequest:
        seen.append((systems, options))
        return request("preliminary", 80)

    monkeypatch.setattr(resources_hf, "hf_resource_request", guess_request)
    calc = SimpleNamespace(
        _initial_guess=InitialGuessSpec("hf"),
        _basis="sto-3g",
        _representation_name="cartesian",
        _library=None,
    )
    target = request("target", 100)
    combined = with_initial_guess_resources(target, calc, [WATER], [0], [1])
    assert target.identity.method == combined.identity.method
    assert target.identity != combined.identity
    assert seen[0][1]["method"] == "rhf"
    assert len(combined.candidates[0].estimates) == 2
    assert sum(e.bytes for e in combined.candidates[0].estimates) == 180
    plan_resources((target,), ResourceBudget(host_bytes=150)).require_feasible()
    with pytest.raises(MemoryError):
        plan_resources((combined,), ResourceBudget(host_bytes=150)).require_feasible()
    calc._initial_guess = None
    assert with_initial_guess_resources(target, calc, [WATER], [0], [1]) is target


@pytest.fixture
def native() -> None:
    library = _native.load_library(device="cpu")
    if not hasattr(library, "generativeqc_initial_guess_options_version"):
        pytest.skip("preliminary SCF native build required")
    require_initial_guess_library(library)


def calculator(method: str, policy: InitialGuessSpec | None = None) -> Calculator:
    return Calculator(
        method,
        "sto-3g",
        device="cpu",
        initial_guess=policy,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16))
        if method.endswith("-rks")
        else None,
    )


@pytest.mark.parametrize(
    "method", ["rhf", "pbe-rks", "pbe0-rks", "r2scan-rks", "b3lyp-rks"]
)
@pytest.mark.parametrize("kind", ["hf", "lda"])
def test_complete_native_energy_and_fallbacks(
    native: None, method: str, kind: str
) -> None:
    baseline = calculator(method).singlepoint(WATER, properties=("energy",))
    assert baseline.initial_guess is None
    policy = InitialGuessSpec(kind)
    calc = calculator(method, policy)
    assert calc.initial_guess is policy
    assert calc.capabilities.supported_properties == frozenset({"energy"})
    seeded = calc.singlepoint(WATER, properties=("energy",))
    assert seeded.converged and seeded.energy == pytest.approx(
        baseline.energy, abs=1e-8
    )
    assert seeded.initial_guess["outcome"] == "used"
    assert seeded.initial_guess["preliminary_iterations"] > 0
    assert seeded.initial_guess["work_counters_complete"]
    assert seeded.initial_guess["target_attempts"] == 1
    for alternate, outcome in (
        (replace(policy, maximum_numeric_bytes=1), "budget_skipped"),
        (replace(policy, max_iterations=1), "preparation_failed"),
    ):
        result = calculator(method, alternate).singlepoint(
            WATER, properties=("energy",)
        )
        assert result.converged and result.energy == pytest.approx(
            baseline.energy, abs=1e-8
        )
        assert result.iterations == baseline.iterations
        assert result.initial_guess["outcome"] == outcome
    with pytest.raises((ValueError, NotImplementedError)):
        calc.singlepoint(WATER, properties=("energy", "forces"))


@pytest.mark.parametrize("kind", ["hf", "lda"])
def test_existing_and_imported_warm_density_is_authoritative(
    native: None, tmp_path: typing.Any, kind: str
) -> None:
    checkpoint = tmp_path / "core-checkpoint.bin"
    with calculator("rhf").prepare_batch([WATER]) as source:
        reference = source.execute(properties=("energy",), strict=True).items[0]
        source.save_checkpoint(checkpoint)
    with calculator("rhf", InitialGuessSpec(kind)).prepare_batch([WATER]) as target:
        target.load_checkpoint(checkpoint)
        imported = target.execute(strict=True).items[0]
        assert imported.warm_start_used
        assert imported.energy == pytest.approx(reference.energy, abs=1e-8)
        assert imported.initial_guess["outcome"] == "existing_density"
        assert imported.initial_guess["preliminary_iterations"] == 0
        target.clear_warm_starts()
        cold = target.execute(strict=True).items[0]
        assert cold.initial_guess["outcome"] == "used"
        target.set_warm_start_updates(False)
        warm = target.execute(strict=True).items[0]
        assert warm.initial_guess["outcome"] == "existing_density"
        # A malformed replay cannot replace the last good state or leave a
        # completed-run preparation diagnostic attached to the failed item.
        bad = [[(float("nan"), 0.0, 0.0), WATER[1][1], WATER[2][1]]]
        failed = target.execute(bad).items[0]
        assert not failed.succeeded and failed.initial_guess is None
        recovered = target.execute(strict=True).items[0]
        assert recovered.initial_guess["outcome"] == "existing_density"
        assert recovered.energy == pytest.approx(reference.energy, abs=1e-8)


def test_runtime_policy_change_requires_repreparation(native: None) -> None:
    calc = calculator("rhf", InitialGuessSpec("hf"))
    with calc.prepare_batch([WATER]) as batch:
        calc._initial_guess = InitialGuessSpec("lda")
        with pytest.raises(RuntimeError, match="policy changed"):
            batch.execute()


@pytest.mark.parametrize("kind", ["hf", "lda"])
def test_linked_global_budget_observes_preparation(native: None, kind: str) -> None:
    policy = InitialGuessSpec(kind)
    probe = calculator("pbe0-rks", policy).estimate_resources([WATER])
    calc = Calculator(
        "pbe0-rks",
        "sto-3g",
        device="cpu",
        initial_guess=policy,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        resource_budget=ResourceBudget(host_bytes=probe.peak_bytes["host"]),
    )
    with calc.prepare_batch([WATER], warm_start=False) as batch:
        result = batch.execute(strict=True).items[0]
        assert result.initial_guess["outcome"] == "used"
        observation = batch.resource_diagnostics["observation"]
        assert observation["samples"] > 0
        assert (
            0 < observation["sampled_item_peak_host_bytes"] <= probe.peak_bytes["host"]
        )
        assert not observation["complete_plan_peak"]


@pytest.mark.parametrize("kind", ["hf", "lda"])
@pytest.mark.parametrize(
    "basis,representation,atoms,charge",
    [
        ("sto-3g", "cartesian", [("O", (0, 0, 0)), ("H", (0, 0, 1.8))], -1),
        ("def2-svp", "spherical", WATER, 0),
    ],
)
def test_charged_and_polarization_basis_endpoints(
    native: None,
    kind: str,
    basis: str,
    representation: str,
    atoms: typing.Any,
    charge: int,
) -> None:
    settings = {
        "method": "pbe0-rks",
        "basis": basis,
        "basis_representation": representation,
        "device": "cpu",
        "ks_options": KsOptions(grid=GridSpec(16, 8, 16)),
    }
    baseline = Calculator(**settings).singlepoint(
        atoms, charge=charge, properties=("energy",)
    )
    actual = Calculator(**settings, initial_guess=InitialGuessSpec(kind)).singlepoint(
        atoms, charge=charge, properties=("energy",)
    )
    assert actual.converged and baseline.converged
    assert actual.energy == pytest.approx(baseline.energy, abs=1e-8)
    assert actual.initial_guess["outcome"] == "used"


def test_native_descriptor_and_diagnostic_abi_guards(native: None) -> None:
    calc = calculator("rhf", InitialGuessSpec("hf"))
    with calc.prepare_batch([WATER]) as batch:
        library = batch._library
        query = library.generativeqc_batch_get_initial_guess_diagnostic
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_native.InitialGuessDiagnosticDescriptor),
        ]
        query.restype = ctypes.c_int
        out = _native.InitialGuessDiagnosticDescriptor(0, _native.ABI_VERSION)
        assert query(batch._batch, 0, ctypes.byref(out)) == _native.STATUS_ABI_MISMATCH
        out.struct_size = ctypes.sizeof(out)
        assert (
            query(batch._batch, 0, ctypes.byref(out)) == _native.STATUS_NOT_IMPLEMENTED
        )
        batch.execute(strict=True)
        assert query(batch._batch, 0, ctypes.byref(out)) == _native.STATUS_SUCCESS
        assert out.requested_kind == 1 and out.outcome == 2
        assert (
            query(batch._batch, 1, ctypes.byref(out)) == _native.STATUS_INVALID_ARGUMENT
        )
        # Rejected replay invalidates the new record before output validation.
        assert (
            library.generativeqc_batch_execute(batch._batch, None, 0, None, 0)
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert (
            query(batch._batch, 0, ctypes.byref(out)) == _native.STATUS_NOT_IMPLEMENTED
        )


def test_native_rejects_short_nested_options_before_preparation(native: None) -> None:
    calc = calculator("rhf", InitialGuessSpec("hf"))
    library = calc._library
    context = ctypes.c_void_p()
    descriptor = calc._context_descriptor()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(descriptor), ctypes.byref(context)
        ),
    )
    system = ctypes.c_void_p()
    calculation = ctypes.c_void_p()
    try:
        from generativeqc import Atom

        system = calc._create_native_system(
            context, tuple(Atom.from_value(a) for a in WATER), 0, 1
        )
        method = calc._method_descriptor()
        method.initial_guess.contents.struct_size = 8
        status = library.generativeqc_calculation_prepare(
            context, system, ctypes.byref(method), ctypes.byref(calculation)
        )
        assert status == _native.STATUS_ABI_MISMATCH and not calculation.value
    finally:
        if calculation.value:
            library.generativeqc_calculation_destroy(calculation)
        if system.value:
            library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


def preliminary_decline_basis() -> typing.Any:
    from generativeqc.basis import (
        BasisProvenance,
        BasisSet,
        BasisShell,
        ElementBasis,
    )

    s = BasisShell(0, ("1",), (("1",),))
    g = BasisShell(4, ("0.1",), (("1",),))
    return BasisSet(
        "preliminary-decline-He-Li",
        (ElementBasis(2, (s, g)), ElementBasis(3, (s,))),
        BasisProvenance("test-local", "1", "CC0", "0" * 64),
        representation="spherical",
    )


def test_lda_decline_planning_preserves_target_and_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import resources_ks
    from generativeqc.resources_hf import hf_resource_request

    systems = [[(2, (0, 0, 0))]]
    basis = preliminary_decline_basis()
    target = hf_resource_request(systems, basis=basis, basis_representation="spherical")
    calc = SimpleNamespace(
        _initial_guess=InitialGuessSpec("lda"),
        _basis=basis,
        _representation_name="spherical",
        _library=None,
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("deterministically declined LDA must not construct a KS owner")

    monkeypatch.setattr(resources_ks, "ks_resource_request", forbidden)
    actual = with_initial_guess_resources(target, calc, systems, None, None)
    assert actual.candidates == target.candidates
    assert actual.scope_exclusions == target.scope_exclusions
    assert actual.identity.topology == target.identity.topology
    assert actual.identity != target.identity
    assert json.loads(actual.identity.schedule)["initial_guess"] == json.loads(
        json.dumps(calc._initial_guess.to_payload())
    )


def test_mixed_lda_decline_charges_only_eligible_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import resources_ks
    from generativeqc.resources_hf import hf_resource_request

    he = [(2, (0, 0, 0))]
    li = [(3, (1, 0, 0))]
    systems = [he, li, he]
    basis = preliminary_decline_basis()
    target = hf_resource_request(
        systems, basis=basis, basis_representation="spherical", charges=[0, 1, 0]
    )
    calc = SimpleNamespace(
        _initial_guess=InitialGuessSpec("lda"),
        _basis=basis,
        _representation_name="spherical",
        _library=None,
    )
    original = resources_ks.ks_resource_request
    seen = []

    def recorded(systems: typing.Any, **options: typing.Any) -> typing.Any:
        result = original(systems, **options)
        seen.append((systems, options, result))
        return result

    monkeypatch.setattr(resources_ks, "ks_resource_request", recorded)
    actual = with_initial_guess_resources(target, calc, systems, [0, 1, 0], [1] * 3)
    selected, options, preliminary = seen[0]
    assert selected == (li,)
    assert options["charges"] == (1,) and options["multiplicities"] == (1,)
    extra = tuple(
        replace(e, name=f"preliminary {e.name}")
        for e in preliminary.candidates[0].estimates
    )
    assert actual.candidates[0].estimates == (*target.candidates[0].estimates, *extra)
    assert actual.identity.topology == target.identity.topology

    def genuine_failure(*args: typing.Any, **kwargs: typing.Any) -> None:
        raise NotImplementedError("independent eligible-provider failure")

    monkeypatch.setattr(resources_ks, "ks_resource_request", genuine_failure)
    with pytest.raises(
        NotImplementedError, match="independent eligible-provider failure"
    ):
        with_initial_guess_resources(target, calc, systems, [0, 1, 0], [1] * 3)


def test_native_lda_decline_with_global_budget(native: None) -> None:
    from generativeqc import ResourceBudget

    systems = [[(2, (0, 0, 0))]]
    options = {
        "method": "rhf",
        "basis": preliminary_decline_basis(),
        "basis_representation": "spherical",
        "device": "cpu",
    }
    direct = Calculator(**options).singlepoint(systems[0], properties=("energy",))
    policy = InitialGuessSpec("lda")
    peak = (
        Calculator(**options, initial_guess=policy)
        .estimate_resources(systems)
        .peak_bytes["host"]
    )
    calc = Calculator(
        **options, initial_guess=policy, resource_budget=ResourceBudget(host_bytes=peak)
    )
    actual = calc.singlepoint(systems[0], properties=("energy",))
    assert actual.converged and actual.energy == pytest.approx(direct.energy, abs=1e-10)
    assert actual.initial_guess["outcome"] == "preparation_failed"
    assert actual.initial_guess["preliminary_iterations"] == 0
    assert actual.initial_guess["preparation_numeric_capacity"] == 0


@pytest.mark.parametrize("short_enums", [False, True])
def test_public_kind_is_fixed_width_against_native_parser(
    native: None, tmp_path: typing.Any, short_enums: bool
) -> None:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    compiler = shutil.which(os.environ.get("CXX", "c++"))
    ccache = shutil.which(os.environ.get("CCACHE", "ccache"))
    if compiler is None or ccache is None:
        pytest.skip("C++ compiler and ccache required for client ABI probe")
    root = Path(__file__).resolve().parents[2]
    library = Path(_native.load_library(device="cpu")._name).resolve()
    source = tmp_path / "initial_guess_client.cpp"
    source.write_text(r"""
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "scf/preliminary_guess.hpp"
static_assert(std::is_same_v<generativeqc_initial_guess_kind, std::int32_t>);
int main() {
  for (const auto kind : {GENERATIVEQC_INITIAL_GUESS_HF, GENERATIVEQC_INITIAL_GUESS_LDA}) {
    generativeqc_initial_guess_options value;
    std::memset(&value, 0xa5, sizeof(value));
    value.struct_size = sizeof(value);
    value.abi_version = GENERATIVEQC_ABI_VERSION;
    value.kind = kind;
    value.max_iterations = value.diis_history = 0;
    value.energy_tolerance = value.density_tolerance = 0;
    value.maximum_numeric_bytes = 0;
    value.radial_points = value.angular_polar = value.angular_azimuth = 0;
    const auto parsed = generativeqc::scf::initial_guess::preliminary_options(&value);
    if (!parsed || static_cast<std::int32_t>(parsed->kind) != kind) return 1;
  }
}
""")
    output = tmp_path / "initial_guess_client"
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    object_file = tmp_path / "initial_guess_client.o"
    command = [ccache, compiler, "-std=c++20", "-O0", "-c"]
    if short_enums:
        command.append("-fshort-enums")
    command += [
        "-I",
        str(root / "include"),
        "-I",
        str(root / "src"),
        str(source),
        "-o",
        str(object_file),
    ]
    compiled = subprocess.run(command, check=False, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run(
        [
            compiler,
            str(object_file),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(output)], check=True, capture_output=True, text=True)
