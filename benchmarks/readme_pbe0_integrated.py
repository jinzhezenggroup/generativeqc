"""Same-source PBE0 policy comparison; no default production promotion.

The requested policy is separate from observed work. The complete endpoint
normalizer is imported from its canonical module, including inside the shared
runner, rather than patched through a shadow module attribute. Validate each
successful force call before spending time on another replay.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from generativeqc import _stationary_cuda
from generativeqc_compiler.method.stationary_resources import (
    BECKE_RETAINED_MAX_ATOMS,
    StationaryCudaResources,
    plan_stationary_cuda_resources,
)

from benchmarks import dft_force_components, readme_omol25
from benchmarks.readme_pbe0 import PBE0

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class QualificationPolicy:
    """Independent qualification choices, not an auto-tuning cost model."""

    name: str
    indexed_force_requested: bool = False
    resident_ao_cutoff: float | None = None
    resident_ao_cache_bytes: int = 16 << 20
    phased_becke_requested: bool = False


POLICIES = {
    policy.name: policy
    for policy in (
        QualificationPolicy("default"),
        QualificationPolicy("local-indexed", True, 1e-16),
        QualificationPolicy("local-indexed-phased", True, 1e-16, 16 << 20, True),
        QualificationPolicy("local-indexed-zero-cache", True, 1e-16, 0),
    )
}


def verify_force_work(
    work: dict[str, Any], policy: QualificationPolicy, *, atoms: int
) -> None:
    """Reject missing observers and silent route misses, never fill counters.

    There is no endpoint-native indexed-page counter in the current ABI. Its
    request remains only a request in the receipt; neither task counts nor a
    timing reduction prove a page count. Native integral route, AO selection
    and Becke execution do have independent endpoint observers.
    """
    record = dft_force_components.normalize_force_work(work)
    if work["native_integrals_required"]:
        if record["stationary_integral_derivative_route"] != "prepared-native-complete":
            raise RuntimeError(
                "integrated qualification lost the native integral route"
            )
        seconds = record["wall_seconds"]["stationary_integral_derivatives"]
        if seconds is None or seconds <= 0:
            raise RuntimeError("native derivative wall observer is missing")
    grid = record["grid_work_plan"]
    batches = (grid["grid_points"] + grid["tile_points"] - 1) // grid["tile_points"]
    observed = record["work_counts"]["observed"]
    if observed.get("geometry_batches") != batches:
        raise RuntimeError("complete grid geometry batch count changed")
    phased = policy.phased_becke_requested and atoms > BECKE_RETAINED_MAX_ATOMS
    if observed.get("phased_becke_batches") != (batches if phased else 0):
        raise RuntimeError("requested Becke policy did not execute")
    selection = record["resident_ao_selection"]
    if selection is None or selection["cutoff"] != policy.resident_ao_cutoff:
        raise RuntimeError("resident AO policy observer is missing or inconsistent")
    if policy.resident_ao_cutoff is None:
        if selection["mode"] != "disabled" or selection["work"] is not None:
            raise RuntimeError("default qualification unexpectedly selected local AO")
        return
    if selection["mode"] != "explicit-sampled-jet-cutoff":
        raise RuntimeError("local AO policy lost its resident grid")
    selected = selection["work"]
    if selected is None or selected["tile_count"] != batches:
        raise RuntimeError("local AO observer does not cover the complete grid")
    if policy.resident_ao_cache_bytes == 0:
        if selected["dense_budget_tiles"] != batches or selected["discoveries"] != 0:
            raise RuntimeError("zero-cache qualification did not use dense fallback")
    elif selected["point_ao_square_sum"] >= selected["dense_point_ao_square_sum"]:
        raise RuntimeError("local AO qualification did not reduce AO contraction work")


@contextmanager
def qualification_policy(
    policy: QualificationPolicy, *, scf_active_ao: bool = False
) -> Iterator[None]:
    """Select SCF and force policies independently, restoring them on failure.

    Explicitly disable SCF selection by default: inherited environment must not
    turn the dense control into a second local-AO candidate. This is a request,
    not proof that native admission or discovery succeeded.
    """
    original_force = _stationary_cuda._complete_rks_cuda_gradient_diagnostic

    def force(state: Any, basis: Any, **options: Any) -> Any:
        options["resident_ao_cutoff"] = policy.resident_ao_cutoff
        options["resident_ao_cache_bytes"] = policy.resident_ao_cache_bytes
        result = original_force(state, basis, **options)
        verify_force_work(result.work, policy, atoms=basis.natom)
        return result

    def resources(**options: Any) -> StationaryCudaResources:
        options["phased_becke"] = policy.phased_becke_requested
        return plan_stationary_cuda_resources(**options)

    environment = {
        "GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE": (
            "indexed" if policy.indexed_force_requested else ""
        ),
        "GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING": "0",
        "GENERATIVEQC_CUDA_KS_ACTIVE_AO": "1" if scf_active_ao else "0",
    }
    with (
        patch.dict(os.environ, environment),
        patch.object(_stationary_cuda, "_complete_rks_cuda_gradient_diagnostic", force),
        patch.object(_stationary_cuda, "plan_stationary_cuda_resources", resources),
    ):
        yield


def main() -> None:
    """Run the unchanged scientific protocol with a separate policy receipt."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--scf-active-ao", action="store_true")
    parser.add_argument("engine", choices=("reference", "native"))
    args, remaining = parser.parse_known_args()
    if args.scf_active_ao and args.engine == "reference":
        parser.error("--scf-active-ao requires the native engine")
    policy = POLICIES[args.policy]
    receipt = {**asdict(policy), "scf_active_ao_requested": args.scf_active_ao}
    with (
        patch.object(sys, "argv", [sys.argv[0], args.engine, *remaining]),
        qualification_policy(policy, scf_active_ao=args.scf_active_ao),
    ):
        readme_omol25.main(PBE0, qualification_policy=receipt)


if __name__ == "__main__":
    main()
