"""Behavioral gates for bounded shell-factor reuse, not loop-spelling checks."""

import subprocess
from pathlib import Path

import pytest
from test_cpu_ao_jet_numerics import ao_probe as ao_probe  # noqa: PLC0414


@pytest.mark.parametrize("order", range(4))
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("mode", [6, 7, 8, 9, 13])
def test_partial_shells_and_distinct_contractions_match_oracle(
    ao_probe: Path, order: int, expanded: bool, mode: int
) -> None:
    """Slices, sparse holes and contraction boundaries retain the selected jets."""
    subprocess.run(
        [str(ao_probe), str(order), str(int(expanded)), str(mode)],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("mode", [10, 11, 12, 14, 15, 16, 17])
def test_invalid_domains_and_nonfinite_results_preserve_canaries(
    ao_probe: Path, mode: int
) -> None:
    """The optimized traversal must retain buffer, selection and finite gates."""
    subprocess.run([str(ao_probe), "3", "1", str(mode)], check=True, timeout=10)
