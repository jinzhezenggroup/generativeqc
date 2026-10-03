"""The real-device qualification option must actually select the new owner path."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_full_force_fixture_selects_opt_in_and_preserves_matrix_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import _stationary_cuda as runtime
    from test_dft_complete_cuda import cooperative_becke_qualification

    observed = []

    class Owner:
        def __init__(
            self,
            atoms: int,
            *,
            integral_derivatives: bool = True,
            cooperative_becke: bool = False,
        ) -> None:
            self.natom = atoms
            self.selected = cooperative_becke
            observed.append((integral_derivatives, cooperative_becke))

        def metrics(self) -> dict[str, int]:
            return {"becke_threads_per_point": 32 if self.selected else 1}

    monkeypatch.setattr(runtime, "_CudaSources", Owner)
    monkeypatch.setenv("GENERATIVEQC_TEST_COOPERATIVE_BECKE", "1")
    cooperative_becke_qualification.__wrapped__(monkeypatch)
    Owner(12)
    Owner(12, cooperative_becke=False)
    Owner(33)
    Owner(12, integral_derivatives=False)
    assert observed == [(True, True), (True, False), (True, True), (False, True)]
    monkeypatch.setattr(Owner, "metrics", lambda _self: {"becke_threads_per_point": 1})
    with pytest.raises(AssertionError, match="generic device fallback"):
        Owner(12)
    with pytest.raises(AssertionError, match="generic device fallback"):
        Owner(33)
    Owner(1)  # The single-atom generic schedule is intentional.


def test_qualification_runner_passes_explicit_cooperative_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import run_stationary_cuda_validation as runner

    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin/nvcc").touch()
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    # An inherited test flag cannot secretly opt in when the CLI option is absent.
    monkeypatch.setenv("GENERATIVEQC_TEST_COOPERATIVE_BECKE", "1")
    monkeypatch.setattr(
        runner,
        "resolve_cuda_execution_profile",
        lambda **_kwargs: SimpleNamespace(wrap=lambda argv: argv, to_dict=dict),
    )
    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append((argv, kwargs)),
            SimpleNamespace(returncode=0),
        )[1],
    )
    for options, expected in (([], "0"), (["--cooperative-becke", "--full-fd"], "1")):
        monkeypatch.setattr(sys, "argv", ["runner", *options, "-k", "complete_r2scan"])
        assert runner.main() == 0
        command, kwargs = calls[-1]
        assert kwargs["env"]["GENERATIVEQC_TEST_COOPERATIVE_BECKE"] == expected
        assert command[-2:] == ["-k", "complete_r2scan"]
        if expected == "1":
            assert kwargs["env"]["GENERATIVEQC_STATIONARY_FULL_FD"] == "1"
    monkeypatch.setattr(
        sys, "argv", ["runner", "--cooperative-becke", "--cpu-regression"]
    )
    with pytest.raises(SystemExit):
        runner.main()


def test_owner_options_keep_integral_and_cooperative_scopes_distinct() -> None:
    import inspect

    from generativeqc._stationary_cuda import _CudaSources

    parameters = inspect.signature(_CudaSources).parameters
    assert parameters["integral_derivatives"].default is True
    assert parameters["cooperative_becke"].default is None
    names = tuple(parameters)
    assert names.index("integral_derivatives") < names.index("cooperative_becke")
