"""Worker-safe numerical reporting, independent of native library availability."""

from collections.abc import Callable

import numpy as np
import pytest
from test_global_hybrid_molecular_hessian import record_scalar_property


@pytest.mark.parametrize("scalar", (float, np.float32, np.float64))
def test_scalar_properties_survive_worker_transport(
    scalar: Callable[[float], float],
    record_property: Callable[[str, object], None],
    request: pytest.FixtureRequest,
) -> None:
    """NumPy error scalars must reach the controller as ordinary Python numbers."""
    error = abs(scalar(0.25) - scalar(0.125))
    record_scalar_property(record_property, "independent_energy_error", error)
    name, recorded = request.node.user_properties[-1]
    assert name == "independent_energy_error"
    assert type(recorded) is float
    assert recorded == 0.125
