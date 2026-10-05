"""Homogeneous prepared regions refuse unimplemented precision obligations."""

from dataclasses import replace

import pytest
from generativeqc_compiler.tensor.native_lowering import (
    emit_contraction_region_portfolio,
)
from test_joint_lowering import _request


def test_homogeneous_region_does_not_invent_precision_obligations() -> None:
    """The simple region executor must refuse casts/audits it cannot implement."""
    request = _request()
    with pytest.raises(ValueError, match="homogeneous arithmetic"):
        emit_contraction_region_portfolio(request, "e" * 64, name="region")
    strict = replace(request, precisions=(request.precisions[0],))
    source = emit_contraction_region_portfolio(strict, "e" * 64, name="region")
    assert source == emit_contraction_region_portfolio(strict, "e" * 64, name="region")
    assert strict.identity in source
    assert strict.precisions[0].identity in source
    assert all(
        f'"{name}"' in source for name in ("cublas", "generated.cuda", "cutensor")
    )
