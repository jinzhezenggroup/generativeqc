"""Explicit exceptions to default-allow automatic Libxc admission.

Automatic semilocal admission is structural: an imported Graph whose required
ingredients are supported is available unless it appears here. This module must
not become a positive allow-list. Add entries only for reproducible
functional-specific defects that cannot yet be expressed as a generic
ingredient/domain capability rule.
"""

from types import MappingProxyType

_ZERO_SPIN_SINGULAR = frozenset(
    {
        "GGA_C_AM05",
        "GGA_C_BMK",
        "GGA_C_GAM",
        "GGA_C_HCTH_A",
        "GGA_C_HYB_TAU_HCTH",
        "GGA_C_LM",
        "GGA_C_N12",
        "GGA_C_N12_SX",
        "GGA_C_OPTC",
        "GGA_C_TAU_HCTH",
        "GGA_C_TM_LYP",
        "GGA_C_WL",
        "GGA_XC_TH1",
        "GGA_XC_TH2",
        "GGA_XC_TH_FC",
        "GGA_XC_TH_FCFO",
        "GGA_XC_TH_FCO",
        "MGGA_C_CC",
        "MGGA_C_CCALDA",
        "MGGA_C_R2SCAN01",
        "MGGA_C_REVTM",
        "MGGA_C_REVTPSS",
        "MGGA_C_RMGGAC",
        "MGGA_C_RPPSCAN",
        "MGGA_C_RREGTM",
        "MGGA_C_RSCAN",
        "MGGA_C_TM",
        "MGGA_C_TPSS",
        "MGGA_C_TPSSLOC",
        "MGGA_C_TPSS_GAUSSIAN",
    }
)

_ZERO_SIGMA_SINGULAR = frozenset(
    {
        "GGA_C_CCDF",
        "GGA_C_WI",
        "GGA_C_WI0",
        "GGA_X_AIRY",
        "GGA_X_AK13",
        "GGA_X_B88M",
        "GGA_X_B88_6311G",
        "GGA_X_BAYESIAN",
    }
)

LIBXC_SEMILOCAL_BLACKLIST = MappingProxyType(
    {
        **{
            name: "known zero-spin-channel production singularity"
            for name in _ZERO_SPIN_SINGULAR
        },
        **{
            name: "known zero-gradient production singularity"
            for name in _ZERO_SIGMA_SINGULAR
        },
    }
)


def blacklist_reason(name: str) -> str | None:
    """Return the explicit functional-specific blocker, if any."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Libxc functional name must be a nonempty string")
    return LIBXC_SEMILOCAL_BLACKLIST.get(name.upper())


__all__ = ["LIBXC_SEMILOCAL_BLACKLIST", "blacklist_reason"]
