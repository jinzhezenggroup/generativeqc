"""The installed response adapter must retain the NativeAO representation."""

from types import SimpleNamespace

import pytest

from generativeqc.rks_response import _RKSIntegralSourceView


@pytest.mark.parametrize(
    "representation,sizes",
    [("cartesian", (1, 3, 6, 10)), ("real_spherical", (1, 3, 5, 7))],
)
def test_rks_topology_preserves_native_shell_component_counts(representation, sizes):
    basis = SimpleNamespace(
        atoms=(object(),),
        shells=tuple(SimpleNamespace(angular_momentum=l) for l in range(4)),
        nao=sum(sizes),
        representation=representation,
        _handle=1,
    )
    view = _RKSIntegralSourceView(basis)
    assert view.shell_sizes == sizes
    assert view.nbf == basis.nao
    assert view.representation == representation
    assert view.basis is basis
    view._check_open()
    basis._handle = 0
    with pytest.raises(RuntimeError, match="closed"):
        view._check_open()


def test_rks_topology_still_rejects_inconsistent_native_ao_count():
    basis = SimpleNamespace(
        atoms=(object(),),
        shells=(SimpleNamespace(angular_momentum=2),),
        nao=6,
        representation="real_spherical",
        _handle=1,
    )
    with pytest.raises(ValueError, match="does not match NativeAO"):
        _RKSIntegralSourceView(basis)
