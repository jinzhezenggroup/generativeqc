"""Compatibility alias for the compiler-owned RCCSD doubles equations."""

import sys

from generativeqc_compiler.cc import doubles as _canonical

sys.modules[__name__] = _canonical
