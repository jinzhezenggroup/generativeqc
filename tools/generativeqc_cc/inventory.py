"""Compatibility alias for the compiler-owned RCCSD inventory."""

import sys

from generativeqc_compiler.cc import inventory as _canonical

sys.modules[__name__] = _canonical
