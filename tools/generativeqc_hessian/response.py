"""Compatibility exports for the production stationary nuclear-response algebra.

The canonical closed-shell RHS and metric-connection implementation lives in
:mod:`generativeqc.stationary_nuclear`.  Keep this repository-tools import
surface for existing Hessian consumers without retaining a second scientific
implementation.
"""

from generativeqc.stationary_nuclear import (
    build_rhf_nuclear_rhs,
    build_stationary_nuclear_rhs,
    metric_density_response_mo,
)

__all__ = [
    "build_rhf_nuclear_rhs",
    "build_stationary_nuclear_rhs",
    "metric_density_response_mo",
]
