"""Close the linked bare/total Hessian identity boundary before SCF."""

import ctypes as ct
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from generativeqc import Calculator, _native, _native_cpu_hessian
from generativeqc_compiler.method import resolve_method


@pytest.fixture
def linked_library() -> ct.CDLL:
    """Run against the installed host derivative owner, not a mock registry."""
    library = _native.load_library(device="cpu")
    assert hasattr(library, "generativeqc_ks_hessian_cpu_model_match_v1")
    version = library.generativeqc_ks_hessian_cpu_version_v1
    version.argtypes, version.restype = [], ct.c_uint32
    if version() != 1:
        pytest.skip("optional native host Hessian owner is not linked")
    return library


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_pair_requires_the_same_bare_electronic_closure(
    linked_library: ct.CDLL, spin: str
) -> None:
    match = linked_library.generativeqc_ks_hessian_cpu_model_match_v1
    match.argtypes, match.restype = [ct.c_char_p, ct.c_char_p], ct.c_uint32

    def digest(method: str) -> bytes:
        return resolve_method(method, spin=spin).identity.encode("ascii")

    pbe = digest("PBE")
    pbe_bj = digest("PBE-D3(BJ)")
    b3lyp = digest("B3LYP")
    assert _native_cpu_hessian.linked_method(linked_library, pbe_bj.decode("ascii"))
    assert _native_cpu_hessian.linked_method(linked_library, b3lyp.decode("ascii"))
    assert match(pbe, pbe) == 2
    assert match(pbe, pbe_bj) == 2
    assert match(b3lyp, pbe_bj) == 1
    assert match(pbe, digest("PBE-D3(0)")) == 1
    assert match(pbe_bj, pbe_bj) == 0
    assert match(None, pbe_bj) == 0


def test_preflight_uses_native_total_pair_without_rechecking_individual_links(
    linked_library: ct.CDLL, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrected = Calculator(method="pbe-d3(bj)-rks")

    def forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("native Hessian preflight tried independent linked-method bits")

    monkeypatch.setattr(_native_cpu_hessian, "linked_method", forbidden)
    _native_cpu_hessian.preflight(corrected)

    stub = SimpleNamespace(
        _device_name="cpu",
        _library=linked_library,
        ks_options=SimpleNamespace(method_ir=resolve_method("B3LYP")),
        method_ir=corrected.method_ir,
        _dispersion_method_ir=corrected.method_ir,
    )
    with pytest.raises(NotImplementedError, match="exact total native correction"):
        _native_cpu_hessian.preflight(stub)
