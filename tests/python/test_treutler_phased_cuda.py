"""Actual stationary CUDA owners with adjusted partitions and ragged grids."""

from types import SimpleNamespace

import pytest
from test_stationary_becke_phased_cuda import native as native  # noqa: PLC0414
from test_stationary_becke_phased_cuda import (
    test_shared_owner_phases_preserve_sources_and_work as check_owner,
)


@pytest.mark.parametrize("atoms", [48, 96, 128])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("primitive", [False, True, 2, "2-off", "2-serial"])
def test_treutler_owner_phase_sources_work_and_lifetime(
    native: SimpleNamespace, atoms: int, implicit: bool, primitive: bool | int | str
) -> None:
    """Include external VV10 seeds, reused panels and transactional failures."""
    check_owner(
        native,
        atoms=atoms,
        implicit=implicit,
        selection="subset",
        external=True,
        primitive=primitive,
        profiled=True,
        adjusted=True,
    )
