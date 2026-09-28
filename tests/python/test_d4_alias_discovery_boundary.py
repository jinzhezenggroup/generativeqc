"""D4 discovery must not promote ordinary aliases into duplicate catalog rows."""

import pytest
from vibeqc.ks import native_dft_carrier, public_dft_selectors, resolve_ks_method
from vibeqc_compiler.method import resolve_method


@pytest.mark.parametrize(
    ("suffix", "spin"), (("rks", "unpolarized"), ("uks", "polarized"))
)
def test_ordinary_alias_resolution_remains_distinct_from_discovery(
    suffix: str, spin: str
) -> None:
    selector = f"pbe1pbe-{suffix}"
    method, _ = resolve_ks_method(selector)
    assert method.identifier == "PBE1PBE"
    assert method.identity == resolve_method("PBE0", spin=spin).identity
    assert native_dft_carrier(selector) == f"pbe-{suffix}"
    assert selector not in public_dft_selectors()
    assert f"pbe0-{suffix}" in public_dft_selectors()


@pytest.mark.parametrize("suffix", ("rks", "uks"))
def test_parameterized_d4_aliases_remain_discoverable(suffix: str) -> None:
    selectors = public_dft_selectors()
    for stem in ("pbe0", "pbeh"):
        selector = f"{stem}-d4-{suffix}"
        assert selector in selectors
        resolve_ks_method(selector)
        assert native_dft_carrier(selector) == f"pbe-{suffix}"
