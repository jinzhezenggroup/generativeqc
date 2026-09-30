"""Compatibility alias for the compiler-owned CC triples module."""

import sys

from generativeqc_compiler.cc import triples as _canonical

sys.modules[__name__] = _canonical
