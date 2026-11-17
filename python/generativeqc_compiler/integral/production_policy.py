"""Report independent CUDA build intent, compiled coverage and preferences.

This is a build contract, not an autotuner or a numerical qualification ledger.
The runtime still admits each consumer against its actual device and workload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from generativeqc_compiler.common.cuda_target import (
    normalize_cuda_architecture,
    normalize_cuda_compile_architecture,
)

from .capabilities import CAPABILITY_MIXED_FOCK, CAPABILITY_STREAMING_FOCK
from .ir import KernelConsumer
from .production_cost import shell_class_index
from .production_k_block import direct_k_block_candidates
from .production_rys_tasks import (
    direct_rys_task_candidates,
    preferred_rys_task_candidates,
)
from .production_rys_values import direct_rys_value_candidates
from .production_shared_rys_xy import (
    shared_rys_xy_companions,
    shared_rys_xy_streaming_companions,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .production_profile import ResolvedProductionProfile
    from .production_selection import KernelSelection


def profile_overrides(
    architectures: Iterable[str], entries: Iterable[str]
) -> dict[str, str]:
    """Reject unknown and duplicate targets, including normalized aliases.

    Per-target intent supersedes a global default, but two overrides never
    silently compete. Primary and derivative generators share this contract.
    """
    configured = {normalize_cuda_architecture(item) for item in architectures}
    result = {}
    for entry in entries:
        target, separator, profile = entry.partition("=")
        if not separator or not target or not profile or profile.strip() != profile:
            raise ValueError("--profile-map requires sm_XX=profile")
        architecture = normalize_cuda_architecture(target)
        if architecture not in configured or architecture in result:
            raise ValueError("--profile-map requires a unique configured target")
        result[architecture] = profile
    return result


def cuda_image_inventory(
    architectures: Iterable[str], images: Iterable[str]
) -> dict[str, list[str]]:
    """Record supplied NVCC images without inferring them from profile names."""
    result: dict[str, list[str]] = {
        normalize_cuda_architecture(item): [] for item in architectures
    }
    for image in images:
        architecture = normalize_cuda_architecture(image)
        if architecture not in result:
            raise ValueError("CUDA image has no configured target")
        normalized = normalize_cuda_compile_architecture(image)
        if normalized not in result[architecture]:
            result[architecture].append(normalized)
    return {target: sorted(values) for target, values in result.items()}


def _mask(selections: Iterable[KernelSelection]) -> int:
    result = 0
    for selection in selections:
        result |= 1 << shell_class_index(selection.spec)
    return result


def resolved_profile_inventory(
    profile: ResolvedProductionProfile,
    *,
    images: Iterable[str] = (),
    request_source: str = "global-default",
) -> dict[str, object]:
    """Describe compiled possibilities separately from measured preferences.

    Masks here precede environment and runtime resource/precision admission.
    Exact derivative packages are a separate compiled consumer, not proof of
    independent scientific acceptance. Empty image lists mean not recorded.
    """
    force = tuple(
        item for item in profile.selections if KernelConsumer.FORCE in item.consumers
    )
    fock = tuple(
        item for item in profile.selections if KernelConsumer.FOCK in item.consumers
    )
    return {
        "architecture": profile.target.architecture,
        "cuda_images": sorted(images),
        "selection": {
            "requested": profile.requested_profile,
            "request_source": request_source,
            "resolved": profile.profile,
            "match": profile.match.value,
            "tuned": profile.tuned,
            "portable": profile.portable,
        },
        "compiled_masks": {
            "force": _mask(force),
            "fock": _mask(fock),
            "mixed_fock": _mask(
                item for item in fock if item.has_capability(CAPABILITY_MIXED_FOCK)
            ),
            "streaming_fock": _mask(
                item for item in fock if item.has_capability(CAPABILITY_STREAMING_FOCK)
            ),
            "rys_value_fock": _mask(direct_rys_value_candidates(profile)),
            "k_block_fock": _mask(direct_k_block_candidates(profile)),
            "rys_task_fock": _mask(direct_rys_task_candidates(profile)),
            "shared_rys_xy_fock": _mask(
                item.incumbent for item in shared_rys_xy_companions(profile)
            ),
            "shared_rys_weighted_z_fock": _mask(
                item.incumbent
                for item in shared_rys_xy_companions(profile, weighted_z=True)
            ),
            "shared_rys_xy_streaming_fock": _mask(
                item.incumbent for item in shared_rys_xy_streaming_companions(profile)
            ),
            "exact_derivative_shells": _mask(force)
            if profile.packaging.derivative_shells
            else 0,
        },
        "default_preferences": {
            "streaming_fock": _mask(
                item for item in fock if item.fock_route == "streaming"
            ),
            "rys_task_fock": _mask(preferred_rys_task_candidates(profile)),
        },
        "packaging": profile.packaging.to_payload(),
        "force_execution": profile.force_policy.to_payload(),
        "runtime_admission": "environment, resource, precision and workload checks still required",
        "qualification": "compiled coverage and profile identity are not scientific or endpoint evidence",
    }


def production_policy_inventory(
    profiles: Iterable[ResolvedProductionProfile],
    *,
    images: Iterable[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Emit a deterministic, versioned per-target build-resolution record."""
    items = sorted(profiles, key=lambda item: item.target.architecture)
    image_map = cuda_image_inventory(
        (item.target.architecture for item in items), images
    )
    return {
        "schema": "generativeqc.cuda-aot-build-policy",
        "schema_version": 2,
        "targets": [
            resolved_profile_inventory(
                item,
                images=image_map[item.target.architecture],
                request_source="target-override"
                if item.target.architecture in (overrides or {})
                else "global-default",
            )
            for item in items
        ],
    }
