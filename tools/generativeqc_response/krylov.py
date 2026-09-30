"""Compatibility alias for the installed response solver owner."""

import sys

from generativeqc import response_solver as _implementation

sys.modules[__name__] = _implementation
