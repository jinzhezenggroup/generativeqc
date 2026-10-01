"""Structured materialization requires an inspectable, nonescaping producer."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from tools.work_audit_python import audit_python


def audit(
    body: str, parameters: str = "n, occupied, virtual, values"
) -> list[dict[str, Any]]:
    source = "import numpy as np\n\ndef producer(" + parameters + "):\n"
    return audit_python(
        source + textwrap.indent(textwrap.dedent(body).strip(), "    "), "fixture.py"
    )


def test_rank_four_occupied_virtual_support_is_symbolic() -> None:
    rows = audit("""
        out = np.zeros((n, n, n, n))
        out[np.ix_(occupied, occupied, virtual, virtual)] = values
        return out
    """)
    assert len(rows) == 1
    row = rows[0]
    assert row["path"] == "fixture.py"
    assert row["line"] == 4
    assert row["function"] == "producer"
    assert row["details"]["allocated_elements"] == "(n) * (n) * (n) * (n)"
    assert row["details"]["write_support"][0]["kind"] == "cartesian-block"
    assert row["details"]["write_support"][0]["upper_bound"] == (
        "(len(occupied)) * (len(occupied)) * (len(virtual)) * (len(virtual))"
    )
    assert row["details"]["consumer"] == "unresolved"
    assert row["details"]["strict_reduction_proven"] is False
    assert "overlap" in " ".join(row["evidence"])


def test_response_constructors_use_general_rules() -> None:
    source = Path("python/generativeqc/response_problem.py").read_text()
    rows = audit_python(source, "renamed_module.py")
    assert {row["function"] for row in rows} == {
        "RotationLayout.density_matrix",
        "RotationLayout.generator_matrix",
    }
    assert {len(row["details"]["write_support"]) for row in rows} == {2}
    assert all("self.nmo" in row["details"]["allocated_elements"] for row in rows)
    assert all(row["path"] == "renamed_module.py" for row in rows)


@pytest.mark.parametrize(
    "write",
    [
        "out[:occupied, occupied:] = values",
        "out[0, :] = values",
        "out[np.diag_indices(n)] = values",
        "for i in range(n):\n    out[i, i] = values[i]",
    ],
)
def test_block_and_diagonal_support(write: str) -> None:
    rows = audit("out = np.zeros((n, n))\n" + write + "\nreturn out")
    assert len(rows) == 1
    assert rows[0]["confidence"] == "high"
    assert not rows[0]["details"]["write_support"][0]["full"]


@pytest.mark.parametrize("triangle", ["triu_indices", "tril_indices"])
def test_triangular_storage_is_still_quadratic(triangle: str) -> None:
    rows = audit(f"out = np.zeros((n, n))\nout[np.{triangle}(n)] = values\nreturn out")
    assert len(rows) == 1
    assert rows[0]["details"]["growth"] == "quadratic-triangle"
    assert (
        rows[0]["details"]["write_support"][0]["upper_bound"] == "(n) * ((n) + 1) // 2"
    )
    assert "not a lower-order sparse domain" in " ".join(rows[0]["evidence"])


@pytest.mark.parametrize(
    "body",
    [
        "out = np.zeros((n, n))\nout[:, :] = values\nreturn out",
        "out = np.zeros((n, n))\nout[0:n, :n:1] = values\nreturn out",
        "out = np.zeros((n, n))\nfor i in range(n):\n    for j in range(n):\n        out[i, j] = values[i, j]\nreturn out",
        "out = np.zeros((n, n))\nout[...] = values\nreturn out",
        "out = np.zeros((n, n))\nreturn out",
        "out = np.ones((n, n))\nout[0, :] = values\nreturn out",
    ],
)
def test_genuine_dense_producers_and_empty_writes_do_not_report(body: str) -> None:
    assert audit(body) == []


@pytest.mark.parametrize(
    "statement",
    [
        "alias = out",
        "alias = out[:]",
        "alias = out.T",
        "alias = (out,)",
        "self.result = out",
        "publish(out)",
        "out.fill(1)",
        "out += values",
        "out[0, :] += values",
        "out[0, :] = mutate(out)",
        "out[0, :] = out[1, :]",
        "unknown()",
        "values = unknown()",
        "out = values",
        "del out",
        "del out[0]",
        "np.copyto(out, values)",
        "out.shape = (n * n,)",
        "if n:\n    out[:, :] = values",
        "try:\n    unknown()\nfinally:\n    out[:] = values",
        "occupied = virtual",
        "n = n + 1",
    ],
)
def test_alias_mutation_escape_and_unknown_control_flow_fail_closed(
    statement: str,
) -> None:
    body = "out = np.zeros((n, n))\nout[np.ix_(occupied, virtual)] = values\n"
    assert audit(body + statement + "\nreturn out") == []


@pytest.mark.parametrize(
    "returned",
    [
        "out + 1",
        "out + values",
        "consume(out)",
        "out.copy()",
        "out * values",
        "(out, values)",
    ],
)
def test_return_must_preserve_support(returned: str) -> None:
    assert audit("out = np.zeros((n, n))\nout[0, :] = values\nreturn " + returned) == []


@pytest.mark.parametrize(
    "prefix, parameters",
    [
        ("", "n, values, np"),
        ("np = replacement\n", "n, values"),
        ("np.zeros = replacement\n", "n, values"),
    ],
)
def test_numpy_alias_shadowing_is_rejected(prefix: str, parameters: str) -> None:
    assert (
        audit(
            prefix + "out = np.zeros((n, n))\nout[0, :] = values\nreturn out",
            parameters,
        )
        == []
    )


def test_import_aliases_are_resolved_without_spelling_assumptions() -> None:
    source = """
        from numpy import zeros as allocate, ix_ as cross
        def build(n, occupied, virtual, values):
            out = allocate((n, n))
            out[cross(occupied, virtual)] = values
            return -out
    """
    assert len(audit_python(textwrap.dedent(source), "anything.py")) == 1


def test_fake_numpy_spelling_does_not_report() -> None:
    source = "def build(n, values):\n    out = np.zeros((n, n))\n    out[0, :] = values\n    return out\n"
    assert audit_python(source, "anything.py") == []


@pytest.mark.parametrize("declaration", ["global out", "nonlocal out"])
def test_nonlocal_allocator_is_an_escape(declaration: str) -> None:
    assert (
        audit(declaration + "\nout = np.zeros((n, n))\nout[0, :] = values\nreturn out")
        == []
    )


def test_closure_capture_is_rejected() -> None:
    assert (
        audit(
            "def capture():\n    return out\nout = np.zeros((n, n))\nout[0, :] = values\nreturn out"
        )
        == []
    )


def test_syntax_error_does_not_crash_scanner() -> None:
    assert audit_python("def incomplete(", "broken.py") == []


@pytest.mark.parametrize(
    "body, parameters",
    [
        (
            "out = np.zeros((n, n))\nfor i in range(n):\n    out[i, i] = values\nreturn out",
            "n, values, range",
        ),
        (
            "out = np.zeros((n, n))\nfor n in range(n):\n    out[n, n] = values\nreturn out",
            "n, values",
        ),
        (
            "out = np.zeros((n, n))\nfor i in range(n):\n    out[:i, :] = values\nreturn out",
            "n, values",
        ),
        (
            "out = np.zeros((n, n))\nfor i in range(n):\n    out[np.ix_(occupied[i], virtual)] = values\nreturn out",
            "n, occupied, virtual, values",
        ),
        (
            "out = np.zeros((n, n))\nfor i in range(n):\n    for j in range(i):\n        out[i, j] = values\nreturn out",
            "n, values",
        ),
    ],
)
def test_unsupported_loop_bindings_fail_closed(body: str, parameters: str) -> None:
    assert audit(body, parameters) == []


@pytest.mark.parametrize(
    "first, second",
    [
        (":occupied, :", "occupied:, :"),
        (":, :occupied", ":, occupied:"),
        ("np.triu_indices(n)", "np.tril_indices(n)"),
    ],
)
def test_known_dense_unions_do_not_report(first: str, second: str) -> None:
    assert (
        audit(
            f"out = np.zeros((n, n))\nout[{first}] = values\nout[{second}] = values\nreturn out"
        )
        == []
    )


def test_class_import_does_not_bind_method_globals() -> None:
    source = """
        class Container:
            import numpy as np
            def build(self, n, values):
                out = np.zeros((n, n))
                out[0, :] = values
                return out
    """
    assert audit_python(textwrap.dedent(source), "fixture.py") == []


def test_late_local_import_does_not_bind_allocator() -> None:
    assert (
        audit(
            "out = np.zeros((n, n))\nout[0, :] = values\nreturn out\nimport numpy as np"
        )
        == []
    )


def test_rebinding_loop_index_to_advanced_array_fails_closed() -> None:
    assert (
        audit(
            "out = np.zeros((n, n))\nfor i in range(1):\n    i = occupied\n    out[i, i] = values\nreturn out"
        )
        == []
    )


def test_loop_cannot_rebind_an_earlier_support_index() -> None:
    assert (
        audit(
            "out = np.zeros((n, n))\nout[np.ix_(occupied, virtual)] = values\nfor occupied in range(n):\n    pass\nreturn out"
        )
        == []
    )


def test_boolean_or_negative_cartesian_indices_keep_numpy_semantics() -> None:
    rows = audit(
        "out = np.zeros((n, n))\nout[np.ix_(occupied, virtual)] = values\nreturn out"
    )
    assert (
        rows[0]["details"]["write_support"][0]["support"] == "np.ix_(occupied, virtual)"
    )
    assert (
        rows[0]["details"]["support_interpretation"]
        == "NumPy indexing coordinates at each write line"
    )


def test_rebound_induction_target_cannot_hide_dense_fill() -> None:
    assert (
        audit(
            "rows = list(range(n))\nout = np.zeros((n, n))\nfor i in range(1):\n    i = rows\n    out[i, :] = 1\nreturn out"
        )
        == []
    )


@pytest.mark.parametrize("index", ["occupied", "self.index"])
def test_repeated_unknown_index_does_not_prove_diagonal_support(index: str) -> None:
    # Ordinary NumPy accepts slice(None), None or True here; each fills the
    # whole matrix. Repeated names only prove a diagonal with a known index type.
    assert (
        audit(f"out = np.zeros((n, n))\nout[{index}, {index}] = values\nreturn out")
        == []
    )


@pytest.mark.parametrize("value", ["slice(None)", "None", "True"])
def test_repeated_index_that_selects_dense_matrix_fails_closed(value: str) -> None:
    assert (
        audit(
            f"index = {value}\nout = np.zeros((n, n))\nout[index, index] = values\nreturn out"
        )
        == []
    )
