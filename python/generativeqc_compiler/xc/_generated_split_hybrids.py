"""Generated qualified split-global-hybrid metadata; do not edit."""

# fmt: off
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypedDict, cast


class SplitHybridRecord(TypedDict):
    family: str
    functional_code: int
    components: tuple[tuple[str, str], ...]
    exact_exchange: str
    cuda_df_energy_force: bool
    cuda_fp64_local_ao: bool


SPLIT_HYBRIDS: Mapping[str, SplitHybridRecord] = MappingProxyType({
    "M06-2X": cast("SplitHybridRecord", MappingProxyType({
        "family": "mgga",
        "functional_code": 131522,
        "components": (
            ("HYB_MGGA_X_M06_2X", "1"),
            ("MGGA_C_M06_2X", "1"),
        ),
        "exact_exchange": "27/50",
        "cuda_df_energy_force": True,
        "cuda_fp64_local_ao": True,
    })),
    "MN15": cast("SplitHybridRecord", MappingProxyType({
        "family": "mgga",
        "functional_code": 131340,
        "components": (
            ("HYB_MGGA_X_MN15", "1"),
            ("MGGA_C_MN15", "1"),
        ),
        "exact_exchange": "11/25",
        "cuda_df_energy_force": False,
        "cuda_fp64_local_ao": False,
    })),
})

SPLIT_HYBRID_COMPONENTS = frozenset(
    component
    for record in SPLIT_HYBRIDS.values()
    for component, _ in record["components"]
)

# fmt: on
