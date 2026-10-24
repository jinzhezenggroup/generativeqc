"""Validate the explicit custom-moment compiler interface."""

import pytest
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_geometry_cuda


@pytest.mark.parametrize("moments", ("custom_moments(total, g.f);", ""))
def test_range_separated_geometry_rejects_custom_moments(moments: str) -> None:
    """A requested quadrature must never be silently replaced by Boys moments."""
    with pytest.raises(ValueError, match="does not support custom moments"):
        emit_df_geometry_cuda(moments=moments, range_separated=True)


@pytest.mark.parametrize("compensated", (False, True))
def test_custom_geometry_moments_remain_supported(compensated: bool) -> None:
    """Ordinary geometry must preserve the caller-provided quadrature call."""
    moments = "custom_moments(total, g.f);"
    source = emit_df_geometry_cuda(moments=moments, compensated=compensated)
    assert moments in source
