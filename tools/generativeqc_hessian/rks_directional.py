"""Compatibility alias for the installed RKS Hessian owner."""

import sys

from generativeqc import rks_hessian_directional as _implementation

sys.modules[__name__] = _implementation
