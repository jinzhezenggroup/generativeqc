"""Execute exact-key, immutable-copy, resource and failure cache contracts."""

import numpy as np
import pytest
from generativeqc._geometry_response_cache import ExactGeometryResponseCache


def test_exact_density_and_geometry_response_storage() -> None:
    cache = ExactGeometryResponseCache()
    density = np.eye(3)
    parts = {"xc_ao": np.arange(6.0).reshape(2, 3), "nuclear": np.zeros((2, 3))}
    assert cache.lookup(density, 1024) is None
    assert cache.store(density, parts, 1024)
    original = cache.lookup(density, 1024)
    assert original is not None
    assert cache.retained_bytes == (9 + 12) * 8
    assert cache.peak_bytes == cache.retained_bytes + 9
    parts["xc_ao"][0, 0] = 99
    assert original["xc_ao"][0, 0] == 0
    with pytest.raises(ValueError):
        original["xc_ao"][0, 0] = 4
    different = density.copy()
    different[0, 0] = np.nextafter(1.0, 2.0)
    assert cache.lookup(different, 1024) is None
    different = density.copy()
    different[0, 1] = -0.0
    assert cache.lookup(different, 1024) is None
    assert cache.lookup(density.reshape(1, 3, 3), 1024) is None
    assert cache.lookup(density.astype(np.float32), 1024) is None
    assert cache.lookup(density, cache.peak_bytes - 1) is None
    assert cache.retained_bytes == 0
    assert not cache.store(density, parts, 1)
    assert cache.store(density, parts, 1024)
    cache.clear()  # The owning geometry rebuild retires every old component.
    assert cache.lookup(density, 1024) is None


@pytest.mark.parametrize("bad_key", [True, False])
def test_nonfinite_result_cannot_replace_successful_key(bad_key: bool) -> None:
    cache = ExactGeometryResponseCache()
    density = np.eye(2)
    parts = {"xc": np.ones((1, 3))}
    assert cache.store(density, parts, 1024)
    (density if bad_key else parts["xc"])[0, 0] = np.nan
    assert not cache.store(density, parts, 1024)
    assert cache.retained_bytes == 0


def test_optional_allocation_failure_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ExactGeometryResponseCache()
    density = np.eye(2)
    parts = {"xc": np.ones((1, 3))}
    assert cache.store(density, parts, 1024)

    def refuse(*args: object, **kwargs: object) -> None:
        raise MemoryError("injected optional scratch refusal")

    with monkeypatch.context() as patch:
        patch.setattr(np, "array_equal", refuse)
        assert cache.lookup(density, 1024) is None
        assert cache.retained_bytes == 0
    assert cache.store(density, parts, 1024)
    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", refuse)
        assert not cache.store(density, parts, 1024)
        assert cache.retained_bytes == 0
    assert cache.store(density, parts, 1024)
    assert cache.lookup(density, 1024) is not None
