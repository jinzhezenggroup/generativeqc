from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src/scf/cuda/df_scf_kernels.cu"


def test_df_density_arithmetic_is_compiler_owned() -> None:
    # Scalar order, symmetric write counts and clipped domains are executable
    # gates in test_df_density_launch_domain and test_scf_retained_lowering.
    source = SOURCE.read_text()
    assert "__global__ void build_device_density_kernel" not in source
    assert '#include "generated_scf_density_cuda.cuh"' in source


@pytest.mark.parametrize(
    ("nbf", "full_pairs", "unique_pairs"),
    [
        (384, 147_456, 73_920),
        (768, 589_824, 295_296),
    ],
)
def test_df_cuda_density_pair_work_census(
    nbf: int, full_pairs: int, unique_pairs: int
) -> None:
    assert nbf * nbf == full_pairs
    assert nbf * (nbf + 1) // 2 == unique_pairs
    assert unique_pairs < full_pairs
