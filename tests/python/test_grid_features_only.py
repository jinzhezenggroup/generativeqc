"""Read-only grid contracts reject unreserved matrix staging before allocation."""

from types import SimpleNamespace

import pytest
from generativeqc_compiler.dft.cuda import DeviceGridTask
from generativeqc_compiler.dft.plan import plan_tiles


def test_feature_only_grid_retains_native_capacity() -> None:
    basis = SimpleNamespace(
        nao=1088, natom=96, numeric_bytes=10000, packed=SimpleNamespace(size=2000)
    )
    arguments = {
        "backend": "cuda",
        "order": 2,
        "tile_points": 64,
        "active_ao_capacity": 1088,
        "budget_bytes": 1 << 30,
    }
    ordinary = plan_tiles(basis, **arguments)
    features = plan_tiles(basis, features_only=True, **arguments)
    assert features.host_bytes < ordinary.host_bytes
    assert features.allocation_bytes == ordinary.allocation_bytes
    assert features.device_bytes == ordinary.device_bytes
    assert features.active_ao_capacity == ordinary.active_ao_capacity


@pytest.mark.parametrize("consumer", ("scatter", "xc"))
def test_feature_only_rejects_potential_inputs_before_array_conversion(
    consumer: str,
) -> None:
    class Unpackable:
        def __array__(self, *args: object, **kwargs: object) -> object:
            pytest.fail("unreserved host matrix packing")

    owner = SimpleNamespace(plan=SimpleNamespace(features_only=True))
    task = DeviceGridTask(owner, SimpleNamespace())
    with pytest.raises(RuntimeError, match="features-only"):
        if consumer == "scatter":
            task.scatter(Unpackable(), download=True)
        else:
            task.xc(Unpackable(), 0, download=True)
