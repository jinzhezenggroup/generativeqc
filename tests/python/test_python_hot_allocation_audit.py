"""Pure-Python hot-loop allocation inventory; no native build or CUDA required."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tools.audit_native_work import audit_tree, fingerprint
from tools.work_audit_python import audit_python


def hot(source: str) -> list[dict]:
    return [
        row
        for row in audit_python(source, "python/synthetic.py")
        if row["rule_id"] == "python.loop-host-allocation"
    ]


def test_numpy_allocations_inside_for_but_not_setup_or_views() -> None:
    source = """
import numpy as np
def replay(n):
    planned = np.zeros(n)
    for tile in range(n):
        tmp = np.empty((n, 3))
        out = np.zeros_like(tmp)
        np.asarray(tmp)
        tmp.reshape(-1)
"""
    rows = hot(source)
    assert [row["details"]["numpy_operation"] for row in rows] == [
        "empty",
        "zeros_like",
    ]
    assert all(row["details"]["phase_hint"] == "per-tile-candidate" for row in rows)
    assert all(
        row["details"]["count_kind"] == "static-site-not-runtime-count"
        and row["details"]["requested_bytes"] is None
        for row in rows
    )


def test_import_aliases_and_while_conditions() -> None:
    source = """
from numpy import full as build
import numpy as np
def replay(n):
    while build((1,), 0).size < n:
        for iteration in range(n):
            np.stack(([1], [2]))
"""
    rows = hot(source)
    assert [row["details"]["numpy_operation"] for row in rows] == [
        "full",
        "stack",
    ]
    assert rows[0]["details"]["phase_hint"] == "unknown-loop"
    assert rows[1]["details"]["phase_hint"] == "per-iteration-candidate"


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\ndef f(np, n):\n    for i in range(n): np.zeros(n)",
        "import numpy as np\ndef f(n):\n    np = other\n    for i in range(n): np.zeros(n)",
        "def f(n):\n    for i in range(n): np.zeros(n)\n    import numpy as np",
        "def f(n):\n    if n: import numpy as np\n    for i in range(n): np.zeros(n)",
        "import numpy as np\ndef f(n):\n    for i in range(n):\n        def delayed(): return np.zeros(n)",
    ],
)
def test_unknown_import_ownership_and_nested_def_not_claimed(source: str) -> None:
    assert not hot(source)


def test_direct_local_import_before_loop_is_admitted() -> None:
    source = """
def f(n):
    import numpy as xp
    for step in range(n):
        xp.ones(n)
"""
    rows = hot(source)
    assert len(rows) == 1
    assert rows[0]["details"]["phase_hint"] == "per-iteration-candidate"


def test_numpy_stack_and_concat_explicit_out_are_not_guaranteed_allocations() -> None:
    source = """
import numpy as np
def f(n, workspace, values):
    for tile in range(n):
        np.stack(values, out=workspace)
        np.concatenate(values, out=workspace)
        np.stack(values, **{'out': workspace})
        np.stack(values, out=None)
        np.empty(n)
"""
    rows = hot(source)
    assert [row["details"]["numpy_operation"] for row in rows] == [
        "stack",
        "empty",
    ]


def test_for_else_and_iterable_setup_are_not_per_iteration() -> None:
    source = """
import numpy as np
def f(n):
    for i in np.zeros(n):
        pass
    else:
        np.ones(n)
    for outer in range(n):
        for inner in range(n):
            pass
        else:
            np.full(n, 1)
"""
    rows = hot(source)
    assert len(rows) == 1
    assert rows[0]["details"]["numpy_operation"] == "full"
    assert rows[0]["details"]["loop_context"] == ["for outer in range(n)"]


def test_fingerprint_survives_line_motion_and_changed_operation() -> None:
    source = "import numpy as np\ndef f(n):\n    for tile in range(n): np.zeros(n)\n"
    first = hot(source)[0]
    moved = hot("\n\n" + source)[0]
    changed = hot(source.replace("np.zeros", "np.ones"))[0]
    assert fingerprint(first) == fingerprint(moved)
    assert fingerprint(first) != fingerprint(changed)


def test_advisory_tree_exports_python_hot_sites_without_runtime_claim(
    tmp_path: Path,
) -> None:
    src = tmp_path / "python"
    src.mkdir()
    (src / "sample.py").write_text(
        "import numpy as np\ndef f(n):\n    for tile in range(n): np.empty(n)\n"
    )
    report = audit_tree(tmp_path, paths=("python",))
    assert report["advisory_only"] is True
    assert report["counts"] == {"python.loop-host-allocation": 1}
    assert report["findings"][0]["details"]["requested_bytes"] is None
    assert report["findings"][0]["fingerprint"]
