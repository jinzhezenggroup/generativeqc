"""Compatibility alias for the compiler-owned CC triples tiles module."""

import sys

from generativeqc_compiler.cc import triples_tiles as _canonical

sys.modules[__name__] = _canonical
