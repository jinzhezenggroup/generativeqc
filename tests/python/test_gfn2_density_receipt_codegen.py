"""Check the opt-in density receipt fragment against the retained schedule."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.method.gfn2_density_lowering import (
    _instrument_checked_pairs,
    emit_gfn2_density_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHA256 = "19701164b11a0afb1490daa147237f8022ea9d684d58299ac119fa470b3c2538"
HOOKS = (
    "    ++density_pairs_visited;\n",
    "      ++density_plain_visits;\n",
    "      ++density_plain_completed;\n",
    "      ++density_weighted_visits;\n",
    "      ++density_weighted_completed;\n",
    "    if (!finite) ++density_failed_pairs;\n",
    "      ++density_published_pairs;\n",
)


def test_default_fragment_identity_and_instrumented_visits() -> None:
    default = emit_gfn2_density_contract()
    instrumented = emit_gfn2_density_contract(instrumented=True)
    assert hashlib.sha256(default.encode()).hexdigest() == DEFAULT_SHA256
    assert instrumented != default
    for hook in HOOKS:
        assert instrumented.count(hook) == 1
        instrumented = instrumented.replace(hook, "", 1)
    assert instrumented == default


def test_diagnostic_schedule_change_fails_closed() -> None:
    default = emit_gfn2_density_contract()
    marker = "    const MatrixPair indices = matrix_pair(pair);"
    with pytest.raises(ValueError, match="diagnostic anchor changed"):
        _instrument_checked_pairs(default.replace(marker, ""))
    with pytest.raises(ValueError, match="diagnostic anchor changed"):
        _instrument_checked_pairs(default.replace(marker, marker + "\n" + marker))


def test_generator_writes_distinct_default_and_receipt_outputs(tmp_path: Path) -> None:
    regular = tmp_path / "generated_gfn2_density_contract.inc"
    diagnostic = tmp_path / "generated_gfn2_density_contract_receipt.inc"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_density_cuda.py"),
            "--output",
            str(regular),
            "--instrumented-output",
            str(diagnostic),
        ],
        check=True,
        timeout=60,
    )
    assert regular.read_text() == emit_gfn2_density_contract()
    assert diagnostic.read_text() == emit_gfn2_density_contract(instrumented=True)
