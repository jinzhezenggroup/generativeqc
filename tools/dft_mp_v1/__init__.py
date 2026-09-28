"""Source-only import boundary for the stationary capacity qualifier."""

from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import CodeType, ModuleType

_REPOSITORY = Path(__file__).resolve().parents[2]
_SOURCE_PYTHON = _REPOSITORY / "python"


class _DftMpSourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compile current source directly, bypassing every bytecode cache."""

    def get_code(self, fullname: str) -> CodeType:
        source = self.get_data(self.path)
        return self.source_to_code(source, self.path)


class _DftMpSourceOnlyFinder:
    """Use source-only loading for the qualifier and its local dependencies."""

    @staticmethod
    def find_spec(
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if (
            fullname != "tools.dft_mp_v1.qualify_capacity"
            and fullname not in ("vibeqc", "vibeqc_compiler")
            and not fullname.startswith(("vibeqc.", "vibeqc_compiler."))
        ):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            return None
        origin = Path(spec.origin).resolve()
        if fullname == "tools.dft_mp_v1.qualify_capacity":
            allowed = origin.is_relative_to(_REPOSITORY / "tools/dft_mp_v1")
        else:
            allowed = origin.is_relative_to(_SOURCE_PYTHON)
        if not allowed:
            return None
        spec.loader = _DftMpSourceOnlyLoader(fullname, str(origin))
        spec.cached = None
        return spec


_SOURCE_ONLY_FINDER = _DftMpSourceOnlyFinder()
sys.meta_path.insert(0, _SOURCE_ONLY_FINDER)
