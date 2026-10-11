"""CPU-only arity guards for audited strict-zip compiler domains."""

from types import SimpleNamespace

import pytest

from generativeqc_compiler.integral.blocks import ShellTile, TensorLayout
from generativeqc_compiler.integral.df_values import build_df_value_ir
from generativeqc_compiler.integral.ir import OperatorFamily
from generativeqc_compiler.integral.one_electron_values import (
    build_one_electron_value_ir,
    evaluate_one_electron_primitive,
)


@pytest.mark.parametrize("angular", [(0,), (0, 0, 0)])
def test_df_metric_rejects_shell_role_arity_mismatch(
    angular: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="basis roles"):
        build_df_value_ir(OperatorFamily.COULOMB_METRIC, angular)


def test_df_three_center_rejects_missing_auxiliary_shell() -> None:
    with pytest.raises(ValueError, match="basis roles"):
        build_df_value_ir(OperatorFamily.THREE_CENTER_ERI, (0, 0))


@pytest.mark.parametrize("angular", [(0,), (0, 0, 0)])
def test_one_electron_rejects_wrong_shell_pair_arity(
    angular: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="two public"):
        build_one_electron_value_ir(OperatorFamily.OVERLAP, angular)


def test_tensor_layout_rejects_stride_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="stride and shape ranks differ"):
        TensorLayout(("i", "j"), (2, 3), (3,))


def test_shell_tile_rejects_mismatched_offset_extent_arity() -> None:
    with pytest.raises(ValueError, match="matching two-, three-, or four-slot"):
        ShellTile((0, 0), (2, 2, 2))


def test_two_center_primitive_binds_only_a_and_b_positions() -> None:
    class _RecordingGraph:
        def evaluate(
            self, _value: object, variables: dict[str, float]
        ) -> dict[str, float]:
            return variables

    kernel = SimpleNamespace(boys_argument=None, value=None, graph=_RecordingGraph())
    variables = evaluate_one_electron_primitive(
        kernel, (1.0, 2.0), ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    )
    assert variables["alpha"] == 1.0
    assert variables["beta"] == 2.0
    assert (variables["b_x"], variables["b_y"], variables["b_z"]) == (1.0, 2.0, 3.0)
    assert all(f"c_{axis}" not in variables for axis in "xyz")
