"""Independent grid semantics and catalog ownership for the CPU facade."""

import ctypes as ct
from typing import Any, NoReturn

import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _named_ks, _native, ks
from generativeqc_compiler.dft.grid import GridPolicy
from generativeqc_compiler.method import resolve_method


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> ct.CDLL:
    """Require the source-built projection; never qualify the old resolver here."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_ks_catalog_execution_policy_v1")
    monkeypatch.setattr(_native, "load_library", lambda **_: loaded)
    return loaded


@pytest.mark.parametrize(
    ("selector", "options", "primary", "secondary"),
    [
        ("pbe-rks", None, GridSpec.treutler(3), None),
        ("pbe-d4-rks", KsOptions(grid_accuracy="tight"), GridSpec.treutler(4), None),
        ("wb97m-v", None, GridSpec.treutler(3), None),
        (
            "wb97m-v-rks",
            KsOptions(grid_accuracy="tight"),
            GridSpec.treutler(4),
            GridSpec.treutler(3),
        ),
        (
            "wb97m-v-rks",
            KsOptions(grid=GridSpec.treutler(0)),
            GridSpec.treutler(0),
            GridSpec.treutler(3),
        ),
        ("wb97m-v-rks", KsOptions(grid=GridSpec(8, 4, 8)), GridSpec(8, 4, 8), None),
        (
            "wb97m-v-rks",
            KsOptions(grid_accuracy="tight", nonlocal_grid=GridSpec.treutler(2)),
            GridSpec.treutler(4),
            GridSpec.treutler(2),
        ),
    ],
)
def test_authenticated_named_constructor_owns_grid_and_domain_metadata(
    library: ct.CDLL,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    options: KsOptions | None,
    primary: GridSpec,
    secondary: GridSpec | None,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("native named grid policy reentered the compiler resolver")

    expected_domain = ks.scf_domain_for_method(selector)
    monkeypatch.setattr(GridPolicy, "resolve", forbidden)
    monkeypatch.setattr(ks, "_scf_domain_for_ir", forbidden)
    calculator = Calculator(method=selector, ks_options=options, initial_guess=None)
    assert calculator._library is library
    assert calculator._ks_options.grid == primary
    assert calculator._ks_options.nonlocal_grid == secondary
    assert calculator._ks_options.scf_domain == expected_domain
    if options is not None and options.grid is not None:
        assert calculator._ks_options.grid is options.grid
    if options is not None and options.nonlocal_grid is not None:
        assert calculator._ks_options.nonlocal_grid is options.nonlocal_grid


def test_explicit_composition_does_not_borrow_named_grid_policy(
    library: ct.CDLL, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("an explicit graph used the registered selector's grid policy")

    monkeypatch.setattr(_named_ks, "select_execution_policy", forbidden)
    calculator = Calculator(
        method="pbe-rks", ks_options=KsOptions(composition=resolve_method("PBE"))
    )
    assert calculator._library is library
    assert calculator._ks_options.grid == GridSpec.treutler(3)


def test_failed_catalog_query_never_retries_compiler_policy(
    library: ct.CDLL, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any) -> int:
        return _native.STATUS_INVALID_ARGUMENT

    def forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("refused native grid policy reentered the compiler resolver")

    monkeypatch.setattr(library, "generativeqc_ks_catalog_execution_policy_v1", refuse)
    monkeypatch.setattr(GridPolicy, "resolve", forbidden)
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 1:"):
        Calculator(method="pbe-rks")


def test_execution_policy_rejects_a_different_exact_identity(library: ct.CDLL) -> None:
    result_type = _named_ks._ExecutionPolicyResult
    query = library.generativeqc_ks_catalog_execution_policy_v1
    query.argtypes = [
        ct.c_char_p,
        ct.c_size_t,
        ct.c_char_p,
        ct.c_size_t,
        ct.c_uint32,
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(result_type),
    ]
    query.restype = ct.c_int
    selector = b"wb97m-v"
    different_identity = resolve_method("PBE").identity.encode("ascii")
    output = result_type(ct.sizeof(result_type), _native.ABI_VERSION, 8, 7)
    before = bytes(output)
    assert (
        query(
            selector,
            len(selector),
            different_identity,
            len(different_identity),
            1,
            0,
            0,
            ct.byref(output),
        )
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert bytes(output) == before


@pytest.mark.parametrize(
    ("selector", "has_nonlocal"), [("pbe-rks", False), ("wb97m-v", True)]
)
def test_ks_wire_nonlocal_budget_uses_the_bound_flag(
    library: ct.CDLL,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    has_nonlocal: bool,
) -> None:
    calculator = Calculator(
        method=selector, ks_options=KsOptions(grid_accuracy="tight")
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("bound nonlocal metadata requested a duplicate execution plan")

    monkeypatch.setattr(KsOptions, "execution_plan", property(forbidden))
    options = calculator._ks_options
    descriptor = ks.native_ks_options(
        options, library=library, backend=_native.BACKEND_CPU_REFERENCE
    )
    assert options.has_nonlocal_correlation is has_nonlocal
    assert descriptor.has_nonlocal_correlation == has_nonlocal
    assert descriptor.nonlocal_maximum_bytes == (
        options.nonlocal_memory_budget_bytes if has_nonlocal else 0
    )


def test_explicit_wrong_scf_domain_is_not_replaced(library: ct.CDLL) -> None:
    with pytest.raises(NotImplementedError, match="tail/spin domain does not match"):
        Calculator(
            method="pbe-rks",
            ks_options=KsOptions(scf_domain=ks.SPLIT_HYBRID_SCF_DOMAIN),
        )
