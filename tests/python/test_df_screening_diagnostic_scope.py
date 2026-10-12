"""Exercise the real endpoint wrapper without requiring a CUDA allocation."""

import typing
from types import SimpleNamespace

import pytest

from benchmarks import df_policy_endpoint as endpoint


@pytest.mark.parametrize("inherited", [None, "1", "off"])
def test_clean_and_cold_calls_disable_inherited_screening(
    monkeypatch: pytest.MonkeyPatch, inherited: str | None
) -> None:
    if inherited is None:
        monkeypatch.delenv("GENERATIVEQC_DF_SCREENING_FEATURES", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_SCREENING_FEATURES", inherited)
    seen = []
    owner = SimpleNamespace(
        execute=lambda **kw: seen.append(
            endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES")
        )
    )
    execute = endpoint.execute_endpoint
    execute(owner)
    execute(owner, cold=True)
    assert seen == [None, None]
    assert endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES") == inherited


@pytest.mark.parametrize("inherited", [None, "off"])
def test_diagnostic_failure_restores_original_environment(
    monkeypatch: pytest.MonkeyPatch, inherited: str | None
) -> None:
    if inherited is None:
        monkeypatch.delenv("GENERATIVEQC_DF_SCREENING_FEATURES", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_SCREENING_FEATURES", inherited)

    def fail(**kw: typing.Any) -> typing.NoReturn:
        assert endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES") == "1"
        raise RuntimeError("injected endpoint failure")

    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        endpoint.execute_endpoint(
            SimpleNamespace(execute=fail), screening_features=True
        )
    assert endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES") == inherited


@pytest.mark.parametrize("energy_only", [False, True])
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("screening_features", [False, True])
def test_complete_endpoint_properties_and_timing(
    monkeypatch: pytest.MonkeyPatch,
    energy_only: bool,
    cold: bool,
    screening_features: bool,
) -> None:
    """Cold evidence includes forces even when later samples request only energy."""
    monkeypatch.setenv("GENERATIVEQC_DF_SCREENING_FEATURES", "off")
    events = []
    ticks = iter([10.0, 14.5])
    result = object()
    expected_features = "1" if screening_features else None

    def clock() -> float:
        assert (
            endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES")
            == expected_features
        )
        events.append("clock")
        return next(ticks)

    def execute(**kwargs: typing.Any) -> object:
        assert (
            endpoint.os.environ.get("GENERATIVEQC_DF_SCREENING_FEATURES")
            == expected_features
        )
        assert kwargs == {
            "strict": True,
            "properties": ("energy",)
            if energy_only and not cold
            else ("energy", "forces"),
        }
        events.append("endpoint")
        return result

    monkeypatch.setattr(endpoint, "time", SimpleNamespace(perf_counter=clock))
    observed, seconds = endpoint.execute_endpoint(
        SimpleNamespace(execute=execute),
        energy_only=energy_only,
        cold=cold,
        screening_features=screening_features,
    )
    assert observed is result and seconds == 4.5
    assert events == ["clock", "endpoint", "clock"]
    assert endpoint.os.environ["GENERATIVEQC_DF_SCREENING_FEATURES"] == "off"
