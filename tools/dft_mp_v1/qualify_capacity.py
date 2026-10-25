"""Report frozen DFT-MP-v1 stationary CUDA capacity without running science.

The report resolves the committed inputs, bundled basis expansion, frozen grid,
current stationary plan and current fail-closed limits.  It performs no native
library load, CUDA initialization, SCF calculation or scientific compilation.
Passing this static report is therefore only a preflight result, never a
scientific qualification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import inspect
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

SOURCE_REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = SOURCE_REPOSITORY / "python"
_DIRECT_SOURCE_EXECUTION = __name__ == "__main__" and __spec__ is None

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import CodeType


class _DftMpSourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compile current source directly, bypassing every bytecode cache."""

    def get_code(self, fullname: str) -> CodeType:
        source = self.get_data(self.path)
        return self.source_to_code(source, self.path)


class _DftMpSourceOnlyFinder:
    """Use source-only loading for repository-local qualifier dependencies."""

    @staticmethod
    def find_spec(
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname not in (
            "generativeqc",
            "generativeqc_compiler",
        ) and not fullname.startswith(("generativeqc.", "generativeqc_compiler.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            return None
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(SOURCE_PYTHON):
            return None
        spec.loader = _DftMpSourceOnlyLoader(fullname, str(origin))
        spec.cached = None
        return spec


_SOURCE_ONLY_FINDER = _DftMpSourceOnlyFinder()
sys.meta_path.insert(0, _SOURCE_ONLY_FINDER)

_PRELOADED_LOCAL_MODULES = frozenset(
    name
    for name, module in tuple(sys.modules.items())
    if module is not None
    and (source := getattr(module, "__file__", None)) is not None
    and Path(source).resolve().is_relative_to(SOURCE_PYTHON)
)


def _git_head(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("Git did not return a full source SHA")
    return revision


_IMPORTED_TOOL_HEAD = _git_head(SOURCE_REPOSITORY)
_IMPORTED_TOOL_SOURCE = Path(__file__).resolve()
_IMPORTED_TOOL_SOURCE_SHA256 = hashlib.sha256(
    _IMPORTED_TOOL_SOURCE.read_bytes().replace(b"\r\n", b"\n")
).hexdigest()

source_python = str(SOURCE_PYTHON)
if source_python in sys.path:
    sys.path.remove(source_python)
sys.path.insert(0, source_python)

import numpy as np
from generativeqc import Atom
from generativeqc import _generated_methods as generated_methods
from generativeqc._model_resolution import snapshot_basis
from generativeqc._stationary_cuda import (
    COMPONENT_LABELS,
    _resolve_becke_primitive_policy,
    _resolve_phased_becke_policy,
    complete_rks_cuda_gradient_diagnostic,
)
from generativeqc.basis import BasisSet
from generativeqc.basis_capabilities import resolved_basis_metadata
from generativeqc.calculator import _basis_pack, _named_basis_record
from generativeqc.ks import (
    KsOptions,
    ks_coefficients,
    native_dft_carrier,
    native_xc_functional_code,
    resolve_ks_method,
    resolve_ks_options,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_COMPONENTS,
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
    STATIONARY_RUNTIME_SOURCE_NAMES,
    _qualified_aot_plan,
    _stationary_aot_name,
    load_stationary_aot_artifact,
    stationary_aot_profile_contract_identity,
    stationary_aot_profile_for_plan,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_resources import (
    STATIONARY_MAX_AOS,
    STATIONARY_MAX_ATOMS,
    STATIONARY_MAX_PRIMITIVES,
    plan_stationary_cuda_grid_work,
    plan_stationary_cuda_resources,
    stationary_cuda_allocation_bytes,
    stationary_cuda_requires_native_integrals,
    stationary_native_pair_reserve,
)

if _SOURCE_ONLY_FINDER in sys.meta_path:
    sys.meta_path.remove(_SOURCE_ONLY_FINDER)

_LOCAL_HELPERS = {
    "Atom": Atom,
    "generated_methods": generated_methods,
    "snapshot_basis": snapshot_basis,
    "BasisSet": BasisSet,
    "complete_rks_cuda_gradient_diagnostic": complete_rks_cuda_gradient_diagnostic,
    "_resolve_becke_primitive_policy": _resolve_becke_primitive_policy,
    "resolved_basis_metadata": resolved_basis_metadata,
    "KsOptions": KsOptions,
    "ks_coefficients": ks_coefficients,
    "native_dft_carrier": native_dft_carrier,
    "native_xc_functional_code": native_xc_functional_code,
    "resolve_ks_method": resolve_ks_method,
    "resolve_ks_options": resolve_ks_options,
    "_basis_pack": _basis_pack,
    "_named_basis_record": _named_basis_record,
    "GridSpec": GridSpec,
    "MolecularGrid": MolecularGrid,
    "jet_indices": jet_indices,
    "plan_tiles": plan_tiles,
    "_qualified_aot_plan": _qualified_aot_plan,
    "load_stationary_aot_artifact": load_stationary_aot_artifact,
    "stationary_aot_profile_contract_identity": stationary_aot_profile_contract_identity,
    "stationary_runtime_sources": stationary_runtime_sources,
    "StationaryGradientPlan": StationaryGradientPlan,
    "StationaryMeanField": StationaryMeanField,
    "plan_stationary_cuda_grid_work": plan_stationary_cuda_grid_work,
    "plan_stationary_cuda_resources": plan_stationary_cuda_resources,
    "stationary_cuda_allocation_bytes": stationary_cuda_allocation_bytes,
    "stationary_cuda_requires_native_integrals": stationary_cuda_requires_native_integrals,
    "stationary_native_pair_reserve": stationary_native_pair_reserve,
}


def _helper_source_path(helper: Any) -> Path | None:
    module = helper if inspect.ismodule(helper) else inspect.getmodule(helper)
    source = None if module is None else getattr(module, "__file__", None)
    return None if source is None else Path(source).resolve()


_IMPORTED_HELPER_SOURCES = {
    name: (
        path,
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    for name, helper in _LOCAL_HELPERS.items()
    if (path := _helper_source_path(helper)) is not None
}
_PUBLIC_METHOD_MANIFEST = SOURCE_REPOSITORY / "manifests/public_methods.json"
_IMPORTED_DATA_DEPENDENCIES = {
    "public_methods_manifest": (
        _PUBLIC_METHOD_MANIFEST,
        hashlib.sha256(
            _PUBLIC_METHOD_MANIFEST.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
}
_IMPORTED_LOCAL_MODULE_SOURCES = {
    name: (
        path,
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    for name, module in tuple(sys.modules.items())
    if module is not None
    and (source := getattr(module, "__file__", None)) is not None
    and (path := Path(source).resolve()).is_relative_to(SOURCE_PYTHON)
}

SCHEMA = "generativeqc.dft-mp-v1.stationary-capacity.v2"
SEMILOCAL_FUNCTIONALS = {"lda": 0, "pbe": 1, "r2scan": 2}
FP64_FORCE_METHODS = frozenset((*SEMILOCAL_FUNCTIONALS, "pbe0", "b3lyp"))
SEMILOCAL_ABI_IDS = {
    "lda-rks": 6,
    "pbe-rks": 7,
    "lda-uks": 8,
    "pbe-uks": 9,
    "r2scan-rks": 10,
    "r2scan-uks": 11,
}
SPARSE_SPHERICAL_COMPONENT_TERMS = {0: 1, 1: 3, 2: 8}
SPD_EXPANSION_CONTRACT_SHA256 = (
    "a482b3ace40fd01758f504c1e47bfa795dcabb5483149873fcd36e71c6b47ae0"
)
AO_PACKER_CONTRACT_SHA256 = (
    "07858ba7f9a78fe6348bbcb9430eb4f8321db8774ea3ce1ecef495629abe2a1c"
)
AO_PACK_BRIDGE_CONTRACT_SHA256 = (
    "aba8691ce81438c66517604c5b2e1e0d39df474888763b4d44686ca8816fabbc"
)
NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256 = (
    "9e210b215cd43f21af6899b328ee738b87f53eaa9b20654b16de760274a8c372"
)
BASIS_SNAPSHOT_CONTRACT_SHA256 = (
    "4dea9a2041897bf843012c01f64c580b6b8696f444611a54c0d161141f1894fd"
)
BASIS_SHELL_EXPANSION_CONTRACT_SHA256 = (
    "300a64c1815273cf31ed5b463eac2e32f24bf4a1db5938975ffbad5cfb27f60c"
)
# Reviewed strict Cartesian x/y/z pairing preserves the native basis layout.
STATIONARY_LAYOUT_CONTRACT_SHA256 = (
    "b3a4132dfc6b5863813894177dc83db175a448b9f207874b0847ab4caefe6460"
)
# The normalized-system caller delegates representation arithmetic to these
# shared shell owners. Pin both helpers as well as the caller so moving the
# equation does not leave arithmetic, width or overflow checks unauthenticated.
NATIVE_SHELL_AO_COUNT_CONTRACT_SHA256 = (
    "e69f53d3da18e44119894075fc898559006f88edcddab5770204f88e306154e5"
)
NATIVE_CARTESIAN_SHELL_COUNT_CONTRACT_SHA256 = (
    "3084647a9f477c2941cc64fa7c66092c5730d813c91194f5082ce3de2ceeb23a"
)
NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256 = (
    "73c8ab3e116dde28f94c4f988f3b9f418ea11a09414ac342ac310d76cf020410"
)
# The complete predicate also admits exact automatic Libxc components through
# the bounded all-electron CPU owner. Its shared context rejects CUDA. CUDA
# coefficients, automatic-Libxc/nonlocal exclusions and runtime ownership retain
# their contracts; structural D4/ECP rejection preserves the admitted D3 owner.
# Keep hashing the full predicate so future guard drift fails closed.
PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256 = (
    "038dcd55175e305001d264203391147a79e0e98238cdfe64df5f581e1a4e38fb"
)
PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256 = (
    "07aac35e787923d81b5e6aad929c55d417a00dfce599f80c361797fb8b4dba9c"
)
# Public force owner bytes are reported as provenance, not an acceptance pin.
# Executable public-route controls cover model eligibility, AOT selection,
# complete-grid overrides, selected AO policy and native-owner cleanup.
# The hybrid predicate also remains source provenance. Executable admission
# controls cover device, ECP, options, schedule and MethodIR eligibility.
NATIVE_GRID_ROUTE_CONTRACT_SHA256 = (
    "a7a81679f2f854149cbd498f481149c529b8b1fdc5963432f3dc06c2ccb79c30"
)
NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256 = (
    "92cd50078b7a96f371ed8d4fcdb77930b8c472134bd1e97bba803ac445d85867"
)
# The additional native static wrappers reuse the existing primitive objects.
# The compiler now supplies the inventory and grouped native generation; retain
# the full wiring/discovery-policy digest so either integration fails closed.
STATIONARY_AOT_CMAKE_CONTRACT_SHA256 = (
    "a3b47ec9184ebf5f987c748d997d2599ec091d25e8236dda8922ab00a020631b"
)
STATIONARY_PAGE_FLUSH_CONTRACT_SHA256 = (
    "1c2e0bb83a12eed7113825855cbe2164f53366b6bb270dd6c1247b498737c77b"
)
# Both primitive modes share the admitted phase storage. Bind constructor-only
# mode selection, legacy-artifact fallback, and boolean resource reservation.
# The zero-seed override configures an existing owner without adding capacity.
# Producer-bound restricted points add optional ABI binding and execution counters;
# they borrow admitted phased scratch without changing any allocation equation.
# Keep whole-owner fingerprints so these controls and metric deltas fail closed.
# Component packing also binds its ragged scratch/descriptor helpers. The
# generic restricted-point request still borrows the same admitted geometry
# scratch; refreshing these owners does not relax any promotion evidence gate.
# Explicit FP32 adds boolean/profile admission and requires both optional
# transport and work-counter ABIs. Descriptor capacity, primitive page budgets,
# resource planning and legacy Becke fallback/control remain unchanged.
STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256 = (
    "d79b0f5c255b78f1f56f017ca4363aeadb99d978de87279395edb4acd53cf183"
)
# Explicit strict pairing retains the fixed native counter-array arities. The
# optional three-field FP32 ABI reports capability and actual enqueued work;
# deltas subtract both work counters but never subtract the capability flag.
STATIONARY_PAGE_METRICS_CONTRACT_SHA256 = (
    "8bf8189b68d9da2c421efe0ec13abfb3296d5106ae2f3df73b7106edf3a0f01e"
)
STATIONARY_METRIC_DELTA_CONTRACT_SHA256 = (
    "c9df420cacb6a16733a023c4de396db5d5c37cbfbecffdc8a232817c197017ec"
)
STATIONARY_PAGE_BULK_CONTRACT_SHA256 = (
    "f49aa6e298b8149d4d359f0cf7cf5c7c734e0c2ad93fecd4e30454df74003a8e"
)
STATIONARY_KIND_TABLE_CONTRACT_SHA256 = (
    "eeec6bd71f566741e4d0978df5b354095c875be183b0cf3138df3a1e8fd47a6a"
)
STATIONARY_COMPONENT_ARRAYS_CONTRACT_SHA256 = (
    "08ea283e18c790a26edcdcac01479def6a7188d9c1a96b8831b4f2b71c8d6045"
)
STATIONARY_COMPONENT_PAGE_CONTRACT_SHA256 = (
    "09be8b78cfee217fef2e37ed5bb26b7a5231f63bd5303dcccacdb6fd1ccb4d9b"
)
STATIONARY_DESCRIPTOR_PACKER_CONTRACT_SHA256 = (
    "92af808cc0d0551072b0a00c29631b86f9c4b65e95bacd68701c4cfbdab926e0"
)
STATIONARY_RESTRICTED_POINT_POLICY_CONTRACT_SHA256 = (
    "14eb08de9be6003c466bff8f9449c95749a38e3bcbb9cf13bcad3d4420ccca51"
)
STATIONARY_DENSITY_JET_POLICY_CONTRACT_SHA256 = (
    "2a117b6ec5c566b996f75795650288182faa93a83b263689431f62d6c8e3a529"
)
STATIONARY_PAGE_SCALAR_CONTRACT_SHA256 = (
    "c5b8ef983462f6c56ebfe6bd6eb8d5cf98f92f36f3e5425504b596730846205f"
)
STATIONARY_PAGE_COMPONENT_INTEGRAL_CONTRACT_SHA256 = (
    "d3f61e820c8bcf0df4bf4fce639f342936b79aceb956cb6e9f43caa3d13cdaa3"
)
STATIONARY_PAGE_NUCLEAR_CONTRACT_SHA256 = (
    "1e86737d8732ef8637378ab925f829dfe229bcf049219c2705a0a4fbf7afdb85"
)
STATIONARY_PAGE_GEOMETRY_CONTRACT_SHA256 = (
    "d469560a2b776a9b86ff5082ba35d3f8ae956c0d63df76aec1ab39550ab92a30"
)
STATIONARY_PAGE_FINISH_SPAN_CONTRACT_SHA256 = (
    "419e21953688eb24d214e6fb43d33ca974cb94632c3797e3f4f0113704d9a1f9"
)
STATIONARY_COMPONENT_MODE_CONTRACT_SHA256 = (
    "8d9819961d3014d161aff8c5c798f926fe6f1d9de54b84a725fdf2f6694b76bb"
)
STATIONARY_TASK_EXECUTOR_CONTRACT_SHA256 = (
    "5f4bf38658ca1da15d845e1a65ebd72195191aebf6009873f131c2cbe8a34195"
)
STATIONARY_SUBMIT_PAGE_CONTRACT_SHA256 = (
    "2fcd280569106fe3cbcf1256a02693aa3532e7db0fa62f7c97d7319454a62a48"
)
STATIONARY_NUCLEAR_PAIR_LOOP_CONTRACT_SHA256 = (
    "5a69bf4fd85d28b137e1ae35bce4a1d32134375bbaca9f66f60c9377a0c8f935"
)
# The endpoint owner is recorded as provenance. Its whole-function hash is not
# a capacity gate: executable orchestration controls cover source arity, complete
# work, concurrent reserves, provider selection, reductions and publication.
# The independent resource/planner/native ABI contracts below remain fail-closed.
STATIONARY_AO_MAP_RESERVE_CONTRACT_SHA256 = (
    "0b9f834f9405340009f7af3a5712840728e5dd46328dad4b52fa07122bc2ecb1"
)
STATIONARY_DEVICE_AO_MAP_RESERVE_CONTRACT_SHA256 = (
    "2ae396067d6e7610a2f0591c3a9eb61bd001d13a60377194b85823074ace5e65"
)
STATIONARY_AO_MAP_CACHE_CONTRACT_SHA256 = (
    "59bdbe506d3868c2299acda5142e9f6a61eaf0657d8d033aa15a08167a495fdc"
)
STATIONARY_TILE_RESOURCE_CONTRACT_SHA256 = (
    "d9683ad506e608bd7043bea3480079379f7a180cdedcd6425d0dab5622063b8f"
)
PHASED_BECKE_POLICY_CONTRACT_SHA256 = (
    "b1ff9a17cefee83a133a8217574f92c902ed601c46c0534e38ee3d5b121876b9"
)
BECKE_PRIMITIVE_POLICY_CONTRACT_SHA256 = (
    "099ccef7e0204d3627bb8ad4f5f8b9181bf6eab28203241dd137df812d157387"
)
BECKE_ZERO_SEED_POLICY_CONTRACT_SHA256 = (
    "3e881038ead5082a0297c98d37d5c8d636f80f6647611f9cdc4720970582bb44"
)
STATIONARY_TILE_LAYOUT_CONTRACT_SHA256 = (
    "2887f95c615859955f768bee0be2a8b47a4d424f02e686748a92321bc9f5c3a7"
)
# Snapshot initialization bytes are recorded as provenance. Executable ABI
# controls verify legacy/explicit allowances, bounds, failure cleanup, immutable
# storage and domain/backend admission without pinning the initializer spelling.
# Audited capability-driven electronic projection and molecular-nonlocal proof.
# Owner/token, coefficient/spin checks and exact grid/provider binding are retained.
# Strict pairing of the three export counters changes source identity only.
# The fitted reserve is valid only for a current, known native DF snapshot.
NATIVE_KS_SNAPSHOT_INTEGRAL_RESERVE_CONTRACT_SHA256 = (
    "1e2eb25ca455dd5505a535a3917a148fbf8cfd59dc6839eeb770aa7c38219f64"
)
SNAPSHOT_GRID_CACHE_CONTRACT_SHA256 = (
    "503c86800926f501f06e3f9b53ed7853cac4a5282f096e56fa6792e46e87872d"
)
# Strict defaults and all resource forwarding remain intact. Explicit candidates
# may retry numerical failure only after positive mixed work, or replay strict
# for missing optional transport. Both paths drain/invalidate before one retry;
# driver/allocation/identity failures never grant an arithmetic retry.
STATIONARY_PUBLIC_WRAPPER_CONTRACT_SHA256 = (
    "45097f6787b42bd6fb13ddd38fd77dead329af1ec357fc8e620610ac967081af"
)
# Native owner/create/allocation identities are recorded below as provenance.
# Their resource, failure, policy and counter contracts execute in host probes.
# Elision preserves dense reservation/launch bounds. Bind its first-derivative
# admission, both tolerance-refresh paths, immutable controls and charged D2H
# observation independently of the unchanged allocation formula.
NATIVE_PHASED_BECKE_INPUT_CONTRACT_SHA256 = (
    "650cf22e25da20ed16fcb6a996e08b9786d540fb51cc7dc19f5f1af6a86f73a8"
)
NATIVE_BECKE_ZERO_SEED_CONFIGURATION_CONTRACT_SHA256 = (
    "cbdd375e3c81bb1a6473186ca5fc9b190bae96c7b96bf8ccad651955a6f295aa"
)
NATIVE_BECKE_ZERO_SEED_METRICS_CONTRACT_SHA256 = (
    "5972d3b6096fc8c4c42ec152d084b7da8fd98d3902b34c1f8544e411680c743e"
)
NATIVE_STATIONARY_TASKS_CONTRACT_SHA256 = (
    "5b0148f4f48019115a82e638d1d6671dd2548f3df6141da6e5254c8967bad2bc"
)
NATIVE_STATIONARY_NUCLEAR_CONTRACT_SHA256 = (
    "be4a553ba6117c7f772882a551d50817935954c5c4d66190e86d9bf2be043902"
)
# Local AO admission changes only collocation/contraction domain. The census
# still charges global n AO capacity and every molecular-grid/Becke point pair.
# Bind the out-of-line predicate itself so later weakening cannot hide behind
# unchanged callers and stale favorable memory/work bounds.
NATIVE_STATIONARY_GEOMETRY_AO_MAP_CONTRACT_SHA256 = (
    "d4830d6d9695219f4bf4c59611717b943c7aa1da016fdba67ceb6036241f1dc0"
)
NATIVE_STATIONARY_GEOMETRY_EXTERNAL_CONTRACT_SHA256 = (
    "7270f2f21f44baf101f9e503f9238dd972f729e431a95b3dd0212311014fe601"
)
NATIVE_STATIONARY_GEOMETRY_ENQUEUE_CONTRACT_SHA256 = (
    "ff5b6e5a6790cc2a75d29906011cf863e04dac04930205d09fc36dcacdaac9e1"
)
NATIVE_STATIONARY_GEOMETRY_ROUTE_CONTRACT_SHA256 = (
    "3fc0a5f613dfaa01ab02104e15929680f3f61fa17c07d59d54241201f903d476"
)
# The phased point/AO split borrows the existing per-lane inline scratch and
# preserves the existing point/AO work and the seven Becke phases.
# Its additional producer launch changes neither retained capacity nor pair
# visits; bind the scratch admission and launch/publication gates explicitly.
# Ordered cooperative normalization reuses the existing phased reservation and
# exact work counts. Audit allocation, actual-device/kernel admission, immutable
# configuration and counters as well as the launch route; a digest refresh must
# not leave the new schedule's capacity or fallback predicates unauthenticated.
NATIVE_PHASED_BECKE_ALLOCATION_CONTRACT_SHA256 = (
    "b61a4ea89c0e68e81cf044c73b075dfcba414cb956be4888bf45e7e222fc8894"
)
NATIVE_PHASED_BECKE_ADMISSION_CONTRACT_SHA256 = (
    "7e542218111e02482d3b4afea38bd02d270003fed47502fb4bc20fb83b2df920"
)
# Binary-scaled admission and the out-of-line normalization dispatch are part
# of the reviewed owner contract, including the incumbent fallback schedule.
BECKE_SCALED_POLICY_CONTRACT_SHA256 = (
    "201fa31bb36d8a3331200e48abf63977727e6825ac301f1e6a6a9fb2c07499c0"
)
NATIVE_BECKE_NORMALIZE_DISPATCH_CONTRACT_SHA256 = (
    "94a74e1093603c69cb24ee1c2a6732e413f94bc82c25f734b011fcbd2a612e1e"
)
NATIVE_BECKE_BINARY_SCALED_METRICS_CONTRACT_SHA256 = (
    "89e7d6e1d62e133b1bd48500024013c426aaede9688fdb248ee32d2c453cccae"
)
NATIVE_BECKE_NORMALIZE_CONFIGURATION_CONTRACT_SHA256 = (
    "972e73f41fa143a8a470fc4ba8bb5178cb82eb92081bcd323328dacaa1ea9801"
)
NATIVE_BECKE_NORMALIZE_METRICS_CONTRACT_SHA256 = (
    "0b9c9d546fff87884bd0279f6a39231a5821afeb5b1664cbfea1ef54abb3550b"
)
NATIVE_BECKE_PHASE_METRICS_CONTRACT_SHA256 = (
    "6357559cf08d355460465b3f374c0898a30a8a5d4829b55a587921db28269934"
)
NATIVE_BECKE_PHASE_PROFILE_CONTRACT_SHA256 = (
    "d8d61c1a2240790216ea931bef7c41c7ac1a5325de9b76b96449b8f1108a3e5d"
)
NATIVE_STATIONARY_PROFILE_CONTRACT_SHA256 = (
    "39de20bb679f7000ed62211ddb8bafcd292052bbb8561bc25eb18f47bb055d86"
)
NATIVE_BECKE_PRIMITIVE_ADMISSION_CONTRACT_SHA256 = (
    "bf0f5dc9db02a8b1e13c965eeb928734b775c8dd96af8b8f7215191e980ed94e"
)
NATIVE_BECKE_NORMALIZED_ADJOINT_CONTRACT_SHA256 = (
    "14fc6c3d9944a82c610b79333618f37ff2b592ab2b1a7b4435ecd1331f40e9dd"
)
NATIVE_BECKE_PRIMITIVE_METRICS_CONTRACT_SHA256 = (
    "a966bc33dc595f2467d359aa7772a3adf7daf37889fec41a9c29937fcba3f32a"
)
NATIVE_STATIONARY_CONFIGURE_BECKE_CONTRACT_SHA256 = (
    "dc844781c888d1bdd281238d4dd23c76048d17f816cb81b5a0616756a22ffe91"
)
NATIVE_STATIONARY_METRICS_CONTRACT_SHA256 = (
    "21e067818117b8ebaf8eeb218aeface0680681fdd6f39285ed6f981c3cef969a"
)
NATIVE_STATIONARY_FINISH_SPAN_CONTRACT_SHA256 = (
    "3f12a2c23709399c56776e34f5d7cd2394a95e153f754694bb7d523772efa431"
)
PRIMITIVE_SUM_DEFINITION = (
    "sum((int(row[2]) * len(expansion) for row, expansion in "
    "zip(aos, expansions, strict=True)))"
)
PRIMITIVE_RECORDS_DEFINITION = (
    "(1 + int(has_exchange)) * primitive_sum ** 4 + "
    "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
)
GRID_PAIR_VISITS_DEFINITION = "grid_work.grid_pair_visits"
GRID_WORK_DEFINITION = (
    "plan_stationary_cuda_grid_work(atoms=na, grid_points=len(state.grid.points), "
    "tile_points=points, max_grid_points=max_grid_points, "
    "max_grid_pair_visits=max_grid_pair_visits, "
    "max_pending_tiles=max_pending_grid_tiles, "
    "max_pending_pair_visits=max_pending_grid_pair_visits)"
)
NATIVE_REQUIREMENT_DEFINITION = "stationary_cuda_requires_native_integrals(atoms=na, aos=n, primitives=basis.nprimitive)"
NATIVE_HOST_RESERVE_DEFINITION = (
    "stationary_native_pair_reserve(atoms=na, aos=n, primitives=basis.nprimitive) "
    "if not ecp and (not bool(getattr(state._source, 'density_fitted', False))) "
    "and callable(getattr(state._source, 'cuda_integral_derivatives', None)) else 0"
)
METHOD_IR_DEFINITION = "state._source.method_ir"
FUNCTIONAL_LOWERING_DEFINITION = "int(state._source.metadata[6])"
INGREDIENTS_DEFINITION = "state._source.functional.ingredients"
NEEDS_FIRST_DEFINITION = "'sigma' in ingredients"
GRID_PLAN_DEFINITION = (
    "plan_tiles(basis, backend='cuda', order=2 if needs_first else 1, "
    "tile_points=tile_points, active_ao_capacity=n if reserve_active_ao else None, "
    "features_only=features_only, budget_bytes=max_device_bytes)"
)
EXACT_AO_MAP_RESOURCES_CONTRACT_SHA256 = (
    "79bf92d98faa27523d70d16578e31e38af699727c78ec4c1f194869ad2c0dcb9"
)
MINIMUM_SOURCE_BYTES_DEFINITION = (
    "stationary_cuda_allocation_bytes(atoms=na, aos=n, primitives=basis.nprimitive, "
    "points=tile_points, tasks=primitive_tile, spins=plan.spin_blocks, "
    "sources=len(source_names), geometry_lanes=min(32, tile_points))"
)
SOURCE_RESOURCES_DEFINITION = (
    "plan_stationary_cuda_resources(atoms=na, aos=n, primitives=basis.nprimitive, "
    "points=tile_points, tasks=primitive_tile, spins=plan.spin_blocks, "
    "sources=len(source_names), target=target, budget_bytes=max_device_bytes - "
    "grid_plan.peak_bytes - sum((value.peak_bytes for value in tensor_plans.values())) - "
    "native_geometry_reserve, phased_becke=_resolve_phased_becke_policy(na, None), "
    "becke_primitive=bool(_resolve_becke_primitive_policy()))"
)
SOURCE_BYTES_DEFINITION = "source_resources.allocation_bytes"
HOST_BOUND_DEFINITION = (
    "grid_plan.host_bytes + 8 * (82 * primitive_tile + "
    "4 * plan.spin_blocks * n * n + 120 * na + "
    "12 * (len(source_names) - len(_SOURCE_NAMES)) * na + "
    "26 * integral_terms + len(COMPONENT_LABELS) ** 4 + "
    "3 * len(COMPONENT_LABELS) ** 2 + 3 * tile_points + 2 * basis.nprimitive + "
    "11 * n + 80) + max((tp.host_bytes for tp in tensor_plans.values()), default=0)"
)
AVAILABLE_DEVICE_BYTES_DEFINITION = (
    "max_device_bytes - grid_plan.peak_bytes - minimum_source_bytes"
)
GATE_PREDICATES = {
    "primitive_metric_range": "records > np.iinfo(np.uint64).max",
    "additional_device": "available <= 0",
    "additional_host": "host_bound > max_host_bytes",
}
BASIS_PACKED_CAPACITY_DEFINITION = (
    "np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)"
)
BASIS_NUMERIC_CAPACITY_DEFINITION = (
    "2 * self.packed.nbytes + 32 * self.natom + "
    "32 * len(self.shells) + 16 * self.nprimitive"
)
STATIONARY_OWNER = {
    "file": "python/generativeqc/_stationary_cuda.py",
    "function": "_complete_rks_cuda_gradient_diagnostic",
}


def _assert_local_imports(*, require_fresh: bool = False) -> None:
    """Reject helpers already imported from an installed or foreign checkout."""

    if _git_head(SOURCE_REPOSITORY) != _IMPORTED_TOOL_HEAD:
        raise RuntimeError(
            "capacity tool checkout changed since import; start a fresh interpreter"
        )
    if (
        hashlib.sha256(
            _IMPORTED_TOOL_SOURCE.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        != _IMPORTED_TOOL_SOURCE_SHA256
    ):
        raise RuntimeError(
            "capacity qualifier source changed since import; start a fresh interpreter"
        )
    if require_fresh and not _DIRECT_SOURCE_EXECUTION:
        raise RuntimeError(
            "authoritative capacity reports require direct source CLI execution"
        )
    if require_fresh and _PRELOADED_LOCAL_MODULES:
        raise RuntimeError(
            "capacity qualifier requires a fresh interpreter; preloaded local modules: "
            + ", ".join(sorted(_PRELOADED_LOCAL_MODULES))
        )
    foreign = []
    stale = []
    for name, original in _LOCAL_HELPERS.items():
        helper = globals()[name]
        source = _helper_source_path(helper)
        imported = _IMPORTED_HELPER_SOURCES.get(name)
        if (
            source is None
            or not source.is_relative_to(SOURCE_PYTHON)
            or imported is None
            or source != imported[0]
        ):
            foreign.append(name)
            continue
        current_digest = hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if helper is not original or current_digest != imported[1]:
            stale.append(name)
    if foreign:
        raise RuntimeError(
            "capacity helper imported outside the tool checkout: "
            + ", ".join(sorted(foreign))
        )
    if stale:
        raise RuntimeError(
            "capacity helper source changed since import: " + ", ".join(sorted(stale))
        )
    stale_data = [
        name
        for name, (path, imported_digest) in _IMPORTED_DATA_DEPENDENCIES.items()
        if not path.is_relative_to(SOURCE_REPOSITORY)
        or not path.is_file()
        or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != imported_digest
    ]
    if stale_data:
        raise RuntimeError(
            "capacity import-time data dependency changed: "
            + ", ".join(sorted(stale_data))
        )
    stale_modules = [
        name
        for name, (path, imported_digest) in _IMPORTED_LOCAL_MODULE_SOURCES.items()
        if not path.is_file()
        or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != imported_digest
    ]
    if stale_modules:
        raise RuntimeError(
            "capacity imported module source changed: "
            + ", ".join(sorted(stale_modules))
        )
    if plan_tiles.__globals__.get("jet_indices") is not jet_indices:
        raise RuntimeError("capacity planner captured AO helper changed since import")


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_node_sha256(source: str, node: ast.AST) -> str:
    """Hash the selected source span without Python-version AST formatting."""

    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("source contract segment is unavailable")
    return _lf_sha256(segment.encode())


def _source_span_sha256(
    source: str,
    *,
    begin: str,
    end: str,
    label: str,
) -> str:
    try:
        start = source.index(begin)
        stop = source.index(end, start)
    except ValueError as error:
        raise RuntimeError(f"{label} source contract is missing") from error
    return _lf_sha256(source[start:stop].encode())


def _cpp_block_sha256(source: str, marker: str) -> str:
    try:
        start = source.index(marker)
        opening = source.index("{", start)
    except ValueError as error:
        raise RuntimeError(f"native source block is missing: {marker}") from error
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return _lf_sha256(source[start : index + 1].encode())
    raise RuntimeError(f"native source block is unterminated: {marker}")


def _snapshot_functional_contract(repository: Path) -> dict[str, str]:
    """Bind native snapshot selector provenance consumed by stationary CUDA."""

    source = (repository / "python/generativeqc/_ks_snapshot.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    owners = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeKsSnapshot"
    ]
    if len(owners) != 1:
        raise RuntimeError("native KS snapshot owner is missing or ambiguous")
    methods = {}
    expected = {
        "__init__": None,
        "decode": None,
        "stationary_integral_device_reserve": (
            NATIVE_KS_SNAPSHOT_INTEGRAL_RESERVE_CONTRACT_SHA256
        ),
    }
    for name, expected_digest in expected.items():
        candidates = [
            node
            for node in owners[0].body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"native KS snapshot {name} contract is missing or ambiguous"
            )
        digest = _source_node_sha256(source, candidates[0])
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError("native KS snapshot functional contract changed")
        key = "init_sha256" if name == "__init__" else f"{name}_sha256"
        methods[key] = digest
    # Decode now delegates exact grid reuse. Bind the delegated content proof
    # and retention cap as well as the caller; a caller hash alone is insufficient.
    cache_source = repository / "python/generativeqc/_snapshot_grid_cache.py"
    cache_digest = _lf_sha256(cache_source.read_bytes())
    if cache_digest != SNAPSHOT_GRID_CACHE_CONTRACT_SHA256:
        raise RuntimeError("native KS snapshot grid-cache contract changed")
    methods["grid_cache_sha256"] = cache_digest
    return methods


def _require_ast_fragments(
    owner: ast.AST, fragments: tuple[str, ...], *, label: str
) -> None:
    """Explain newly admitted source semantics without relaxing owner hashes.

    Full owner fingerprints below still reject additional statements, reordered
    gates, and any other drift. These independent AST checks prevent refreshing
    those fingerprints from accidentally admitting broken CSR policy wiring.
    """

    actual = [ast.dump(node) for node in ast.walk(owner)]
    for fragment in fragments:
        expected = ast.parse(fragment).body[0]
        if isinstance(expected, ast.Expr):
            expected = expected.value
        if actual.count(ast.dump(expected)) != 1:
            raise RuntimeError(f"{label} contract changed: {fragment}")


def _fitted_geometry_policy_contract(
    resource_owner: ast.FunctionDef, endpoint: ast.FunctionDef
) -> None:
    """Bind the fitted-only reserve and preference without changing Direct rows.

    The frozen census estimates Direct providers. Its existing resource equations
    and explicit tile size remain valid only while the new bounded reserve and
    cap-gated tile preference stay guarded by the fitted-provider contract.
    Keep these semantic checks in addition to the complete owner fingerprints.
    """
    _require_ast_fragments(
        resource_owner,
        (
            (
                "native_geometry_reserve = ("
                "min(max(0, available - sum(value.peak_bytes for value in tensor_plans.values())), "
                "state._source.stationary_integral_device_reserve(atoms=na, aos=n, primitives=basis.nprimitive)) "
                "if not ecp and bool(getattr(state._source, 'density_fitted', False)) "
                "and callable(getattr(state._source, 'stationary_integral_device_reserve', None)) "
                "else max(0, available - sum(value.peak_bytes for value in tensor_plans.values())) "
                "if not ecp and bool(getattr(state._source, 'density_fitted', False)) "
                "else min(max(0, available - sum(value.peak_bytes for value in tensor_plans.values())), "
                "stationary_native_pair_reserve(atoms=na, aos=n, primitives=basis.nprimitive)) "
                "if not ecp and callable(getattr(state._source, 'cuda_integral_derivatives', None)) "
                "else 0)"
            ),
        ),
        label="stationary CUDA fitted geometry reserve",
    )
    _require_ast_fragments(
        endpoint,
        (
            (
                "large_fitted = (na >= _AUTO_PHASED_BECKE_MIN_ATOMS "
                "and bool(getattr(state._source, 'density_fitted', False)) "
                "and callable(getattr(state._source, 'stationary_integral_device_reserve', None)))"
            ),
            "from ._cuda_force_resources import DF_FORCE_DEVICE_CAP, DF_FORCE_HOST_CAP",
            "preferred_tile_points = 512",
            (
                "if large_fitted:\n"
                "    preferred_tile_points = (1024 if max_device_bytes >= DF_FORCE_DEVICE_CAP "
                "and max_host_bytes >= DF_FORCE_HOST_CAP else 256)"
            ),
            (
                "if (use_fitted_integrals and "
                "int(native_integral_resources.get('one_electron_device_peak_bytes', 0)) "
                "> layout.native_geometry_reserve):\n"
                "    raise RuntimeError('fitted stationary device staging exceeds admitted reserve')"
            ),
        ),
        label="stationary CUDA fitted geometry policy",
    )


def _resident_ao_policy_contract(tree: ast.Module) -> None:
    """Bind CSR selection, replay identity, and reserves to their source owners."""

    names = (
        "complete_rks_cuda_gradient_diagnostic",
        "_complete_rks_cuda_gradient_diagnostic",
        "_stationary_resident_ao_cache",
        "ensure",
        "_request",
        "_stationary_device_ao_map_reserve",
    )
    owners = {}
    for name in names:
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(candidates) != 1:
            raise RuntimeError("stationary CUDA resident AO policy owner is ambiguous")
        owners[name] = candidates[0]
    wrapper, endpoint, cache, ensure, request, device_reserve = (
        owners[name] for name in names
    )
    label = "stationary CUDA resident AO policy"
    for owner in (wrapper, endpoint, ensure, request, cache):
        defaults = {
            arg.arg: ast.unparse(value)
            for arg, value in zip(
                owner.args.kwonlyargs, owner.args.kw_defaults, strict=True
            )
            if value is not None
        }
        prefix = "" if owner is cache else "resident_ao_"
        producer = "producer" if owner is cache else "resident_ao_producer"
        if (
            defaults.get(producer) != "'sampled-jets'"
            or defaults.get(prefix + "max_active_fraction") != "1.0"
        ):
            raise RuntimeError(f"{label} defaults contract changed")
    # Both the public wrapper and prepared replay identity must preserve the
    # selected producer and threshold, even when current defaults are dense.
    for owner in (wrapper, request):
        for field in ("resident_ao_producer", "resident_ao_max_active_fraction"):
            bindings = [
                ast.unparse(value)
                for node in ast.walk(owner)
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and key.value == field
            ]
            if bindings != [field]:
                raise RuntimeError(f"{label} identity/forwarding contract changed")
    for owner, callee in ((endpoint, "prepared.ensure"), (ensure, "self._request")):
        calls = [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        if len(calls) != 1:
            raise RuntimeError(f"{label} forwarding owner is ambiguous")
        for field in ("resident_ao_producer", "resident_ao_max_active_fraction"):
            bindings = [
                ast.unparse(keyword.value)
                for keyword in calls[0].keywords
                if keyword.arg == field
            ]
            if bindings != [field]:
                raise RuntimeError(f"{label} forwarding contract changed")
    _require_ast_fragments(
        endpoint,
        (
            (
                'if resident_ao_producer in {"pre-ao-envelope-native-csr", "exact-jets-native-bitmask"}:\n'
                "    ao_map_reserve = _stationary_device_ao_map_reserve("
                "layout, ao_map_reserve, max_device_bytes)"
            ),
            (
                'if resident_ao_producer == "exact-jets-native-bitmask":\n'
                "    ao_map_reserve = ExactAoMapResources("
                "n, len(state.grid.points), tile_points).admitted_bytes(ao_map_reserve)"
            ),
            "host_bound += ao_map_reserve",
            (
                "_stationary_resident_ao_cache(prepared, ao, state, resident_grid, "
                "cutoff=resident_ao_cutoff, budget_bytes=ao_map_reserve, "
                "producer=resident_ao_producer, "
                "max_active_fraction=resident_ao_max_active_fraction)"
            ),
            (
                "feature_lease = (ao.feature_task_device_points(point_pointer, "
                "end - begin, None, ingredients) if ao_maps is None else "
                "ao_maps.feature_task(ao, ao_maps.domain, begin, end - begin, ingredients))"
            ),
            "task.layout.require_derivative_order(2 if needs_first else 1)",
        ),
        label=label,
    )
    _require_ast_fragments(
        device_reserve,
        (
            (
                "dense_device_bound = (layout.grid_plan.peak_bytes + "
                "layout.source_resources.allocation_bytes + "
                "sum(value.peak_bytes for value in layout.tensor_plans.values()))"
            ),
            (
                "available = max_device_bytes - dense_device_bound - "
                "layout.native_geometry_reserve"
            ),
            "return min(requested_bytes, max(0, available))",
        ),
        label=label,
    )
    _require_ast_fragments(
        ensure,
        (
            (
                'if resident_ao_producer in {"pre-ao-envelope-native-csr", "exact-jets-native-bitmask"}:\n'
                "    device_peak_bound += resident_ao_cache_bytes"
            ),
            (
                "if device_peak_bound > max_device_bytes:\n"
                '    raise ValueError("prepared stationary CUDA device budget exceeded")'
            ),
        ),
        label=label,
    )
    _require_ast_fragments(
        cache,
        (
            (
                "key = (domain, id(grid), grid.geometry_generation, "
                "grid.basis_generation, float(cutoff), budget_bytes, "
                "producer, max_active_fraction)"
            ),
            (
                "owner = (ResidentDeviceAoMapOwner(grid, domain, cutoff=cutoff, "
                "budget_bytes=budget_bytes, max_active_fraction=max_active_fraction, "
                "**({'producer': producer} if producer == 'exact-jets-native-bitmask' else {})) "
                "if producer in {'pre-ao-envelope-native-csr', 'exact-jets-native-bitmask'} else "
                "ResidentAoMapCache(grid, domain, cutoff=cutoff, "
                "budget_bytes=budget_bytes, producer=producer))"
            ),
        ),
        label=label,
    )


def _public_resident_ao_policy_contract(owner: ast.FunctionDef) -> None:
    """Keep the public policy decision attached to ordinary and composite calls."""

    label = "public CUDA resident AO policy"
    expected = {
        "resident_ao_producer": "decision.producer",
        "resident_ao_max_active_fraction": "decision.max_active_fraction",
    }
    for field, value in expected.items():
        bindings = [
            ast.unparse(item)
            for node in ast.walk(owner)
            if isinstance(node, ast.Dict)
            for key, item in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and key.value == field
        ]
        if bindings != [value]:
            raise RuntimeError(f"{label} forwarding contract changed")
    for callee, fields in (
        (
            "ForceActiveAoWorkload",
            {"device_name": "getattr(self, '_stationary_cuda_device_name', None)"},
        ),
        (
            "prepared.execute",
            {
                "active_ao_producer": "decision.producer",
                "active_ao_max_active_fraction": "decision.max_active_fraction",
            },
        ),
    ):
        calls = [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        if len(calls) != 1:
            raise RuntimeError(f"{label} forwarding owner is ambiguous")
        for field, value in fields.items():
            if [
                ast.unparse(keyword.value)
                for keyword in calls[0].keywords
                if keyword.arg == field
            ] != [value]:
                raise RuntimeError(f"{label} forwarding contract changed")


def _optional_ao_schedule_contract(tree: ast.Module) -> None:
    """Execute current dry fallback policy without loading a native owner.

    The census represents the default no-map request. Optional indexed storage
    must not alter that route or consume a native failure as a capacity retry.
    These behavioral witnesses allow equivalent implementation refactors.
    """
    from generativeqc import _stationary_cuda as runtime

    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plan_stationary_optional_ao_schedule"
    ]
    if len(helpers) != 1:
        raise RuntimeError("stationary CUDA optional AO schedule owner is ambiguous")
    namespace = dict(vars(runtime))
    exec(  # noqa: S102 - current dry helper, resource admission replaced below
        compile(ast.Module(body=helpers, type_ignores=[]), "<AO schedule>", "exec"),
        namespace,
    )
    choose = namespace["_plan_stationary_optional_ao_schedule"]
    for requested, refusal in (
        (False, None),
        (True, None),
        (True, "indexed"),
        (False, "all"),
        (True, "all"),
        (True, "runtime"),
    ):
        layout, work = object(), object()
        admitted: list[bool] = []

        def admit(
            points: int,
            indexed: bool,
            *,
            refusal: str | None = refusal,
            admitted: list[bool] = admitted,
            layout: Any = layout,
            work: Any = work,
        ) -> tuple[Any, Any]:
            if points != 7:
                raise AssertionError("explicit tile was changed")
            if refusal == "runtime" and indexed:
                raise LookupError("native failure must propagate")
            if refusal == "all" or (refusal == "indexed" and indexed):
                raise ValueError("numeric capacity refused")
            admitted.append(indexed)
            return layout, work

        try:
            selected, selected_work, reason = choose(
                grid_points=19,
                tile_points=7,
                preferred_tile_points=16,
                requested_maps=requested,
                admit=admit,
            )
        except (ValueError, LookupError) as error:
            if (refusal == "all" and isinstance(error, ValueError)) or (
                refusal == "runtime" and isinstance(error, LookupError)
            ):
                continue
            raise RuntimeError(
                "stationary CUDA optional AO schedule contract changed"
            ) from error
        except AssertionError as error:
            raise RuntimeError(
                "stationary CUDA optional AO schedule contract changed"
            ) from error
        if (
            refusal in {"all", "runtime"}
            or selected is not layout
            or selected_work is not work
            or not admitted
            or set(admitted) != {requested and refusal != "indexed"}
            or (reason is not None) != (refusal == "indexed")
        ):
            raise RuntimeError("stationary CUDA optional AO schedule contract changed")


def _geometry_resource_contract(path: Path) -> None:
    """Execute dry planning equations and boundaries, retaining source provenance.

    A whole-file fingerprint used to reject the additional immutable grid panels
    before any negative control ran. Validate the actual source under review,
    including mutated copies, against independent byte/work equations instead.
    No native library, CUDA context, allocation or scientific compiler is used.
    """
    module = ModuleType("_dft_mp_resource_contract")
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        _DftMpSourceOnlyLoader(module.__name__, str(path)).exec_module(module)
        _check_geometry_resources(module)
    except Exception as error:
        raise RuntimeError(
            f"stationary CUDA geometry-resource contract changed: {error}"
        ) from error
    finally:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous


def _check_geometry_resources(module: ModuleType) -> None:
    def require(condition: bool, description: str) -> None:
        if not condition:
            raise ValueError(description)

    def rejected(function: Any, **arguments: Any) -> None:
        try:
            function(**arguments)
        except ValueError:
            return
        raise ValueError(f"invalid resource request admitted: {arguments}")

    def arena(shape: dict[str, int], lanes: int, cached: bool = False) -> int:
        atoms, aos = shape["atoms"], shape["aos"]
        # Sum separately owned FP64/int64 panels, including immutable topology.
        words = (
            2 * shape["primitives"]
            + 4 * aos
            + 22 * shape["tasks"]
            + 3 * atoms
            + atoms
            + (atoms + 1)
            + 3 * shape["sources"] * atoms
            + 3 * shape["points"]
            + 2 * shape["spins"] * aos**2
            + 18 * lanes * atoms
        )
        return 8 * words + (48 * (atoms * (atoms - 1) // 2) if cached else 0) + 256

    allocation = module.stationary_cuda_allocation_bytes
    resources = module.plan_stationary_cuda_resources
    native = module.stationary_cuda_requires_native_integrals
    shape = {
        "atoms": 12,
        "aos": 96,
        "primitives": 240,
        "points": 4096,
        "tasks": 256,
        "spins": 2,
        "sources": 8,
    }
    target = cuda_target_info("sm_120")
    caps = {
        "atoms": 128,
        "aos": 2048,
        "primitives": 16384,
        "points": 4096,
        "tasks": 4096,
        "spins": 2,
        "sources": 128,
        "geometry_lanes": 2048,
    }
    for field, cap in caps.items():
        for invalid in (0, -1, True, 1.5, cap + 1, 1 << 64):
            rejected(allocation, **{**shape, "geometry_lanes": 1, field: invalid})
        valid = {**shape, "geometry_lanes": 1, field: cap}
        require(
            allocation(**valid)
            == arena(
                {key: value for key, value in valid.items() if key != "geometry_lanes"},
                valid["geometry_lanes"],
            ),
            f"inclusive {field} capacity",
        )
    rejected(allocation, **shape, geometry_lanes=1, cache_center_geometry=1)
    for atoms, aos, primitives in (
        (1, 1, 1),
        (32, 128, 4096),
        (33, 129, 4097),
        (96, 768, 704),
        (128, 2048, 16384),
    ):
        basis = {"atoms": atoms, "aos": aos, "primitives": primitives}
        require(
            native(**basis) == (atoms > 32 or aos > 128 or primitives > 4096),
            "native-integral-only boundary",
        )
        reserve = (
            2 * (58 * aos + 52 * atoms + 16 * primitives + 4 * aos * (aos + 1) + 32)
            + 48 * atoms
        )
        require(
            module.stationary_native_pair_reserve(**basis) == reserve,
            "paired native reserve",
        )
        require(
            module.stationary_fitted_integral_reserve(**basis)
            == reserve + 16 * aos**2 + 192 * atoms,
            "fitted native reserve",
        )
        for points in (1, 17, 256, 4096):
            candidate = {**shape, **basis, "points": points}
            lanes = min(points, 2048, (16 << 20) // (144 * atoms))
            for count in {1, lanes}:
                for cached in (False, True):
                    require(
                        allocation(
                            **candidate,
                            geometry_lanes=count,
                            cache_center_geometry=cached,
                        )
                        == arena(candidate, count, cached),
                        "arena panel accounting",
                    )
                budget = arena(candidate, count)
                planned = resources(**candidate, target=target, budget_bytes=budget)
                require(
                    (
                        planned.geometry_lanes,
                        planned.allocation_bytes,
                        planned.geometry_scratch_bytes,
                        planned.center_geometry_bytes,
                        planned.phased_becke_bytes,
                    )
                    == (count, budget, 144 * atoms * count, 0, 0),
                    "tight budget preserves bounded lanes",
                )
            rejected(
                resources,
                **candidate,
                target=target,
                budget_bytes=arena(candidate, 1) - 1,
            )
            if lanes < points:
                rejected(allocation, **candidate, geometry_lanes=lanes + 1)
    for field, cap in (("atoms", 128), ("aos", 2048), ("primitives", 16384)):
        for function in (
            native,
            module.stationary_native_pair_reserve,
            module.stationary_fitted_integral_reserve,
        ):
            rejected(
                function, **{"atoms": 1, "aos": 1, "primitives": 1, field: cap + 1}
            )
    for field, boundary in (("atoms", 32), ("aos", 128), ("primitives", 4096)):
        basis = {"atoms": 1, "aos": 1, "primitives": 1}
        require(not native(**{**basis, field: boundary}), "small-domain fallback")
        require(native(**{**basis, field: boundary + 1}), "native-only admission")
    for invalid in (-1, True, 1 << 64):
        rejected(resources, **shape, target=target, budget_bytes=invalid)
    for flag in ("cooperative_becke", "phased_becke", "becke_primitive"):
        rejected(resources, **shape, target=target, budget_bytes=1 << 30, **{flag: 1})
    # Pair/phase storage may consume only spare budget after full tile lanes.
    for atoms in (1, 11, 12, 32, 33, 48, 128):
        candidate = {**shape, "atoms": atoms, "points": 17}
        pairs = atoms * (atoms - 1) // 2
        base = arena(candidate, 17, True)
        phase = 8 * (4 * pairs * 17 + (12 * atoms + 2) * 17 + pairs)
        for architecture in ("sm_80", "sm_120"):
            device = cuda_target_info(architecture)
            for explicit in (None, False, True):
                for requested in (False, True):
                    for delta in (-1, 0):
                        planned = resources(
                            **candidate,
                            target=device,
                            budget_bytes=base + phase + delta,
                            cooperative_becke=explicit,
                            phased_becke=requested,
                        )
                        cooperative = (
                            explicit
                            if explicit is not None
                            else architecture == "sm_120" and atoms >= 12
                        )
                        active = cooperative and atoms > 1
                        admitted = (
                            requested
                            and active
                            and atoms > 32
                            and architecture == "sm_120"
                            and delta == 0
                        )
                        require(
                            planned.phased_becke_bytes == (phase if admitted else 0),
                            "optional phase budget/target admission",
                        )
                        require(
                            planned.becke_threads_per_point == (128 if active else 1),
                            "cooperative target and shape admission",
                        )
            default = resources(**candidate, target=device, budget_bytes=1 << 30)
            require(default.phased_becke_bytes == 0, "phased storage is opt-in")

    schedule = module.plan_stationary_cuda_grid_schedule
    for explicit in (None, 37, 4096):
        for preferred in (256, 1024):

            def admit(points: int, requested: int | None = explicit) -> int:
                if requested is None and points > 128:
                    raise ValueError("capacity refused")
                return points

            require(
                schedule(
                    grid_points=513,
                    tile_points=explicit,
                    admit=admit,
                    preferred_tile_points=preferred,
                )
                == (128 if explicit is None else explicit),
                "explicit tile and bounded capacity fallback",
            )

    class RuntimeFailure(RuntimeError):
        pass

    def runtime_failure(points: int) -> int:
        raise RuntimeFailure("execution failures must propagate")

    try:
        schedule(grid_points=513, tile_points=None, admit=runtime_failure)
    except RuntimeFailure:
        pass
    else:
        raise ValueError("runtime failure swallowed by grid schedule")
    for field, value in (
        ("grid_points", 0),
        ("grid_points", 1 << 64),
        ("tile_points", True),
        ("tile_points", 4097),
        ("preferred_tile_points", 0),
    ):
        rejected(
            schedule,
            **{
                "grid_points": 1,
                "tile_points": None,
                "admit": lambda n: n,
                field: value,
            },
        )

    work = module.plan_stationary_cuda_grid_work
    for atoms in (1, 3, 96, 128):
        pairs = atoms * (atoms - 1) // 2
        for points in (1, 16, 17, 35):
            total = (1 + 2 * points) * pairs
            budget = max(1, 2 * min(points, 16) * pairs)
            arguments = {
                "atoms": atoms,
                "grid_points": points,
                "tile_points": 16,
                "max_grid_points": points,
                "max_grid_pair_visits": max(1, total),
                "max_pending_tiles": 2,
                "max_pending_pair_visits": budget,
            }
            planned = work(**arguments)
            chunk = min(points, 16 if pairs else 32)
            require(
                (
                    planned.grid_pair_visits,
                    planned.tile_count,
                    planned.chunk_points,
                    planned.chunk_count,
                    planned.chunk_pair_visits,
                )
                == (
                    total,
                    (points + 15) // 16,
                    chunk,
                    (points + chunk - 1) // chunk,
                    2 * chunk * pairs,
                ),
                "complete grid work and bounded windows",
            )
            require(
                list(planned.chunks())
                == [
                    (begin, min(begin + chunk, points))
                    for begin in range(0, points, chunk)
                ],
                "complete contiguous grid windows",
            )
            if points > 1:
                rejected(work, **{**arguments, "max_grid_points": points - 1})
            if total > 1:
                rejected(work, **{**arguments, "max_grid_pair_visits": total - 1})
                if budget > 1:
                    rejected(
                        work, **{**arguments, "max_pending_pair_visits": budget - 1}
                    )
            require(
                work(
                    **{
                        **arguments,
                        "max_grid_points": None,
                        "max_grid_pair_visits": None,
                    }
                )
                == planned,
                "optional diagnostic guards preserve complete public work",
            )
    for field, cap in (
        ("atoms", 128),
        ("grid_points", 1 << 40),
        ("tile_points", 4096),
        ("max_pending_tiles", 4096),
        ("max_pending_pair_visits", 1 << 40),
    ):
        rejected(
            work, **{"atoms": 1, "grid_points": 1, "tile_points": 1, field: cap + 1}
        )


def _source_limits(repository: Path) -> dict[str, Any]:
    """Bind current compiler/native capacity and private diagnostic defaults.

    Public whole-grid overrides are audited independently in _source_public_route.
    A static-cap pass remains conditional on the required native provider result.
    """

    snapshot_functional_contract = _snapshot_functional_contract(repository)
    source_path = repository / STATIONARY_OWNER["file"]
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    _optional_ao_schedule_contract(tree)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == STATIONARY_OWNER["function"]
    ]
    if len(functions) != 1:
        raise RuntimeError("stationary CUDA admission owner is missing or ambiguous")
    owner = functions[0]
    wrappers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "complete_rks_cuda_gradient_diagnostic"
    ]
    if len(wrappers) != 1:
        raise RuntimeError("stationary CUDA public wrapper is missing or ambiguous")
    wrapper_digest = _source_node_sha256(source, wrappers[0])
    if wrapper_digest != STATIONARY_PUBLIC_WRAPPER_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA public wrapper contract changed")
    phase_policies = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_phased_becke_policy"
    ]
    phase_thresholds = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_AUTO_PHASED_BECKE_MIN_ATOMS"
    ]
    if (
        len(phase_policies) != 1
        or _source_node_sha256(source, phase_policies[0])
        != PHASED_BECKE_POLICY_CONTRACT_SHA256
        or len(phase_thresholds) != 1
        or ast.unparse(phase_thresholds[0].value) != "48"
    ):
        raise RuntimeError("stationary CUDA phased Becke policy contract changed")
    primitive_policies = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_becke_primitive_policy"
    ]
    if (
        len(primitive_policies) != 1
        or _source_node_sha256(source, primitive_policies[0])
        != BECKE_PRIMITIVE_POLICY_CONTRACT_SHA256
    ):
        raise RuntimeError("stationary CUDA Becke primitive policy contract changed")
    zero_seed_helpers = {
        "density_jet_policy_sha256": (
            "_stationary_density_jet_count",
            STATIONARY_DENSITY_JET_POLICY_CONTRACT_SHA256,
        ),
        "restricted_point_policy_sha256": (
            "_resolve_restricted_point_policy",
            STATIONARY_RESTRICTED_POINT_POLICY_CONTRACT_SHA256,
        ),
        "becke_zero_seed_policy_sha256": (
            "_resolve_becke_zero_seed_policy",
            BECKE_ZERO_SEED_POLICY_CONTRACT_SHA256,
        ),
        "metric_delta_sha256": (
            "_metric_delta",
            STATIONARY_METRIC_DELTA_CONTRACT_SHA256,
        ),
    }
    for label, (name, expected_digest) in zero_seed_helpers.items():
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if (
            len(helpers) != 1
            or _source_node_sha256(source, helpers[0]) != expected_digest
        ):
            raise RuntimeError(f"stationary CUDA {label} contract changed")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    resource_owners = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plan_stationary_cuda_tile"
    ]
    tile_callbacks = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "admit_tile"
    ]
    if len(resource_owners) != 1 or len(tile_callbacks) != 1:
        raise RuntimeError("stationary CUDA tile admission owners are ambiguous")
    resource_owner, tile_callback = resource_owners[0], tile_callbacks[0]
    _fitted_geometry_policy_contract(resource_owner, owner)
    layout_node = classes.get("_StationaryCudaTileLayout")
    if (
        layout_node is None
        or _source_node_sha256(source, layout_node)
        != STATIONARY_TILE_LAYOUT_CONTRACT_SHA256
    ):
        raise RuntimeError("stationary CUDA tile layout contract changed")
    page_methods = {
        "kind_table": (
            "_CudaSources",
            "_kind_table",
            STATIONARY_KIND_TABLE_CONTRACT_SHA256,
        ),
        "component_arrays": (
            "_CudaSources",
            "_component_arrays",
            STATIONARY_COMPONENT_ARRAYS_CONTRACT_SHA256,
        ),
        "component_page": (
            "_CudaSources",
            "_component_integral_page",
            STATIONARY_COMPONENT_PAGE_CONTRACT_SHA256,
        ),
        "descriptor_packer": (
            "_CudaSources",
            "_pack_integral_descriptors",
            STATIONARY_DESCRIPTOR_PACKER_CONTRACT_SHA256,
        ),
        "initializer": (
            "_CudaSources",
            "__init__",
            STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256,
        ),
        "metrics": ("_CudaSources", "metrics", STATIONARY_PAGE_METRICS_CONTRACT_SHA256),
        "flush": ("_CudaSources", "flush", STATIONARY_PAGE_FLUSH_CONTRACT_SHA256),
        "bulk": (
            "_CudaSources",
            "integral_page",
            STATIONARY_PAGE_BULK_CONTRACT_SHA256,
        ),
        "scalar": (
            "_CudaSources",
            "_append_task",
            STATIONARY_PAGE_SCALAR_CONTRACT_SHA256,
        ),
        "component_integral": (
            "_CudaSources",
            "integral",
            STATIONARY_PAGE_COMPONENT_INTEGRAL_CONTRACT_SHA256,
        ),
        "nuclear": (
            "_CudaSources",
            "nuclear",
            STATIONARY_PAGE_NUCLEAR_CONTRACT_SHA256,
        ),
        "geometry": (
            "_CudaSources",
            "geometry",
            STATIONARY_PAGE_GEOMETRY_CONTRACT_SHA256,
        ),
        "finish_span": (
            "_CudaSources",
            "finish_span",
            STATIONARY_PAGE_FINISH_SPAN_CONTRACT_SHA256,
        ),
        "executor": (
            "_BoundedStationaryTaskExecutor",
            "execute_pages",
            STATIONARY_TASK_EXECUTOR_CONTRACT_SHA256,
        ),
    }
    resource_source = (
        repository / "python/generativeqc_compiler/method/stationary_resources.py"
    )
    resource_digest = _lf_sha256(resource_source.read_bytes())
    _geometry_resource_contract(resource_source)
    exact_ao_map_digest = _lf_sha256(
        (repository / "python/generativeqc_compiler/dft/ao_map_plan.py").read_bytes()
    )
    if exact_ao_map_digest != EXACT_AO_MAP_RESOURCES_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA exact AO map resource contract changed")
    scaled_policy_digest = _lf_sha256(
        (repository / "src/dft/becke_scaled_policy.hpp").read_bytes()
    )
    if scaled_policy_digest != BECKE_SCALED_POLICY_CONTRACT_SHA256:
        raise RuntimeError(
            "stationary CUDA binary-scaled Becke policy contract changed"
        )
    page_contract = {
        "becke_scaled_policy_sha256": scaled_policy_digest,
        "exact_ao_map_resources_sha256": exact_ao_map_digest,
        "public_wrapper_sha256": wrapper_digest,
        "geometry_resources_sha256": resource_digest,
        "ordinary_tile_layout_sha256": STATIONARY_TILE_LAYOUT_CONTRACT_SHA256,
        "phased_becke_policy_sha256": PHASED_BECKE_POLICY_CONTRACT_SHA256,
        "becke_primitive_policy_sha256": BECKE_PRIMITIVE_POLICY_CONTRACT_SHA256,
        **{label: digest for label, (_, digest) in zero_seed_helpers.items()},
    }
    for label, (class_name, method_name, expected_digest) in page_methods.items():
        class_node = classes.get(class_name)
        methods = (
            []
            if class_node is None
            else [
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ]
        )
        if len(methods) != 1:
            raise RuntimeError(f"stationary CUDA {label} page owner is ambiguous")
        digest = _source_node_sha256(source, methods[0])
        if digest != expected_digest:
            raise RuntimeError(f"stationary CUDA {label} page contract changed")
        page_contract[f"{label}_sha256"] = digest
    component_modes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_component_mode"
    ]
    if len(component_modes) != 1:
        raise RuntimeError("stationary CUDA component-mode owner is ambiguous")
    component_mode_digest = _source_node_sha256(source, component_modes[0])
    if component_mode_digest != STATIONARY_COMPONENT_MODE_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA component-mode contract changed")
    page_contract["component_mode_sha256"] = component_mode_digest
    native_source = (repository / "src/dft/stationary_gradient_cuda.cuh").read_text(
        encoding="utf-8"
    )
    native_blocks = {
        "native_owner_sha256": (
            "struct Owner {",
            None,
        ),
        "native_allocation_sha256": (
            "size_t allocation(",
            None,
        ),
        "native_create_sha256": (
            "int stationary_create(",
            None,
        ),
        "native_reset_sha256": (
            "int stationary_reset(",
            None,
        ),
        "native_geometry_reset_sha256": (
            "int stationary_geometry_reset(",
            None,
        ),
        "native_phased_becke_input_sha256": (
            "PhasedBeckeInput phased_input(",
            NATIVE_PHASED_BECKE_INPUT_CONTRACT_SHA256,
        ),
        "native_becke_zero_seed_configuration_sha256": (
            "int stationary_configure_becke_zero_seed_v1(",
            NATIVE_BECKE_ZERO_SEED_CONFIGURATION_CONTRACT_SHA256,
        ),
        "native_becke_zero_seed_metrics_sha256": (
            "int stationary_becke_zero_seed_metrics_v1(",
            NATIVE_BECKE_ZERO_SEED_METRICS_CONTRACT_SHA256,
        ),
        "native_tasks_sha256": (
            "int stationary_tasks(",
            NATIVE_STATIONARY_TASKS_CONTRACT_SHA256,
        ),
        "native_nuclear_sha256": (
            "int stationary_nuclear(",
            NATIVE_STATIONARY_NUCLEAR_CONTRACT_SHA256,
        ),
        "native_geometry_ao_map_sha256": (
            "bool valid_geometry_ao_map(",
            NATIVE_STATIONARY_GEOMETRY_AO_MAP_CONTRACT_SHA256,
        ),
        "native_geometry_external_sha256": (
            "int stationary_geometry_external(",
            NATIVE_STATIONARY_GEOMETRY_EXTERNAL_CONTRACT_SHA256,
        ),
        "native_geometry_enqueue_sha256": (
            "int stationary_geometry_enqueue(",
            NATIVE_STATIONARY_GEOMETRY_ENQUEUE_CONTRACT_SHA256,
        ),
        "native_geometry_route_sha256": (
            "int stationary_geometry(",
            NATIVE_STATIONARY_GEOMETRY_ROUTE_CONTRACT_SHA256,
        ),
        "native_launch_geometry_sha256": (
            "void launch_geometry(",
            None,
        ),
        "native_configure_becke_sha256": (
            "int stationary_configure_becke(",
            NATIVE_STATIONARY_CONFIGURE_BECKE_CONTRACT_SHA256,
        ),
        "native_metrics_sha256": (
            "int stationary_metrics(",
            NATIVE_STATIONARY_METRICS_CONTRACT_SHA256,
        ),
        "native_becke_primitive_admission_sha256": (
            "int stationary_configure_becke_primitive_v1(",
            NATIVE_BECKE_PRIMITIVE_ADMISSION_CONTRACT_SHA256,
        ),
        "native_becke_normalized_adjoint_sha256": (
            "int stationary_configure_becke_normalized_adjoint_v1(",
            NATIVE_BECKE_NORMALIZED_ADJOINT_CONTRACT_SHA256,
        ),
        "native_becke_primitive_metrics_sha256": (
            "int stationary_becke_primitive_metrics_v1(",
            NATIVE_BECKE_PRIMITIVE_METRICS_CONTRACT_SHA256,
        ),
        "native_phased_becke_allocation_sha256": (
            "size_t phased_allocation(",
            NATIVE_PHASED_BECKE_ALLOCATION_CONTRACT_SHA256,
        ),
        "native_phased_becke_admission_sha256": (
            "int stationary_configure_phased_becke_v1(",
            NATIVE_PHASED_BECKE_ADMISSION_CONTRACT_SHA256,
        ),
        "native_becke_normalize_dispatch_sha256": (
            "void launch_phased_becke_normalize(",
            NATIVE_BECKE_NORMALIZE_DISPATCH_CONTRACT_SHA256,
        ),
        "native_becke_binary_scaled_metrics_sha256": (
            "int stationary_becke_binary_scaled_metrics_v1(",
            NATIVE_BECKE_BINARY_SCALED_METRICS_CONTRACT_SHA256,
        ),
        "native_becke_normalize_configuration_sha256": (
            "int stationary_configure_becke_normalize_v1(",
            NATIVE_BECKE_NORMALIZE_CONFIGURATION_CONTRACT_SHA256,
        ),
        "native_becke_normalize_metrics_sha256": (
            "int stationary_becke_normalize_metrics_v1(",
            NATIVE_BECKE_NORMALIZE_METRICS_CONTRACT_SHA256,
        ),
        "native_becke_phase_metrics_sha256": (
            "int stationary_becke_phase_metrics_v1(",
            NATIVE_BECKE_PHASE_METRICS_CONTRACT_SHA256,
        ),
        "native_becke_phase_profile_sha256": (
            "int stationary_becke_phase_profile_v1(",
            NATIVE_BECKE_PHASE_PROFILE_CONTRACT_SHA256,
        ),
        "native_profile_sha256": (
            "int stationary_profile(",
            NATIVE_STATIONARY_PROFILE_CONTRACT_SHA256,
        ),
        "native_finish_span_sha256": (
            "int stationary_finish_span(",
            NATIVE_STATIONARY_FINISH_SPAN_CONTRACT_SHA256,
        ),
    }
    for label, (marker, expected_digest) in native_blocks.items():
        digest = _cpp_block_sha256(native_source, marker)
        # Current native owner/allocation/create and reset/dispatch identities
        # are provenance. Executable host contracts cover arena accounting,
        # admission, cleanup, fencing, tolerance and typed routes.
        # A static preflight is never their execution/numerical qualification.
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError(f"stationary CUDA {label} contract changed")
        page_contract[label] = digest
    if tuple(COMPONENT_LABELS) != tuple(QUALIFIED_SPD_COMPONENTS):
        raise RuntimeError("stationary CUDA component-label capacity changed")
    # Arithmetic remains owned by the dry resource helper. The endpoint binds
    # the selected layout, and the callback supplies the actual candidate size.
    # Audit all three owners rather than treating layout attributes as equations.
    resource_names = {
        "native_integral_host_reserve",
        "grid_plan",
        "minimum_source_bytes",
        "source_resources",
        "available",
        "host_bound",
    }
    definition_nodes = {
        name: [
            node
            for node in (
                resource_owner.body
                if name in resource_names
                else tile_callback.body
                if name == "grid_work"
                else owner.body
            )
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == ("work" if name == "grid_work" else name)
        ]
        for name in (
            "requires_native_integrals",
            "grid_work",
            "native_integral_host_reserve",
            "primitive_sum",
            "records",
            "pair_visits",
            "method",
            "functional",
            "ingredients",
            "needs_first",
            "grid_plan",
            "minimum_source_bytes",
            "source_resources",
            "source_bytes",
            "available",
            "host_bound",
        )
    }
    if any(len(nodes) != 1 for nodes in definition_nodes.values()):
        raise RuntimeError(
            "stationary CUDA capacity definitions are missing or ambiguous"
        )
    definitions = {
        name: ast.unparse(nodes[0].value) for name, nodes in definition_nodes.items()
    }
    expected_definitions = {
        "requires_native_integrals": NATIVE_REQUIREMENT_DEFINITION,
        "grid_work": GRID_WORK_DEFINITION,
        "native_integral_host_reserve": NATIVE_HOST_RESERVE_DEFINITION,
        "primitive_sum": PRIMITIVE_SUM_DEFINITION,
        "records": PRIMITIVE_RECORDS_DEFINITION,
        "pair_visits": GRID_PAIR_VISITS_DEFINITION,
        "method": METHOD_IR_DEFINITION,
        "functional": FUNCTIONAL_LOWERING_DEFINITION,
        "ingredients": INGREDIENTS_DEFINITION,
        "needs_first": NEEDS_FIRST_DEFINITION,
        "grid_plan": GRID_PLAN_DEFINITION,
        "minimum_source_bytes": MINIMUM_SOURCE_BYTES_DEFINITION,
        "source_resources": SOURCE_RESOURCES_DEFINITION,
        "source_bytes": SOURCE_BYTES_DEFINITION,
        "available": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "host_bound": HOST_BOUND_DEFINITION,
    }
    definition_labels = {
        "requires_native_integrals": "native-integral requirement",
        "grid_work": "bounded grid-work plan",
        "native_integral_host_reserve": "native-integral host reserve",
        "primitive_sum": "primitive-sum",
        "records": "primitive-record",
        "pair_visits": "grid-pair-visits",
        "method": "stationary MethodIR",
        "functional": "native snapshot functional-code lowering",
        "ingredients": "native semilocal ingredient provenance",
        "needs_first": "grid derivative-order",
        "grid_plan": "grid-plan input",
        "minimum_source_bytes": "minimum-source-bytes",
        "source_resources": "source-resources",
        "source_bytes": "source-bytes",
        "available": "available-device-bytes",
        "host_bound": "host-bound",
    }
    for name, expected in expected_definitions.items():
        if definitions[name] != expected:
            raise RuntimeError(
                f"stationary CUDA {definition_labels[name]} definition changed"
            )
    direct_if_tests = [
        ast.unparse(node.test)
        for node in (*owner.body, *resource_owner.body)
        if isinstance(node, ast.If)
    ]
    gate_labels = {
        "primitive_metric_range": "logical primitive metric range",
        "additional_device": "positive additional-device remainder",
        "additional_host": "additional-host",
    }
    for name, predicate in GATE_PREDICATES.items():
        if direct_if_tests.count(predicate) != 1:
            raise RuntimeError(f"stationary CUDA {gate_labels[name]} predicate changed")
    if direct_if_tests.count("records > max_primitive_records") != 0:
        raise RuntimeError("stationary CUDA restored a whole-force primitive cap")
    page_budget_bindings = [
        ast.unparse(keyword.value)
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "page_work_budget"
    ]
    if page_budget_bindings.count("max_primitive_records") != 2:
        raise RuntimeError("stationary CUDA page-work budget wiring changed")
    executor_definitions = {
        name: [
            ast.unparse(node.value)
            for node in ast.walk(owner)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        for name in ("fixed_task_capacity", "task_executor")
    }
    expected_executor_definitions = {
        "fixed_task_capacity": "min(integral_terms, primitive_tile)",
        "task_executor": (
            "_BoundedStationaryTaskExecutor(fixed_capacity=fixed_task_capacity, "
            "resident_capacity=primitive_tile, page_capacity=primitive_tile)"
        ),
    }
    for name, expected in expected_executor_definitions.items():
        if executor_definitions[name] != [expected]:
            raise RuntimeError(f"stationary CUDA {name} definition changed")
    page_execution_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "task_executor.execute_pages"
        and [ast.unparse(argument) for argument in node.args]
        == [
            "domain",
            "submit_page",
        ]
        and not node.keywords
    ]
    host_gates = [
        node
        for node in resource_owner.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == GATE_PREDICATES["additional_host"]
    ]
    selections = [
        node
        for node in owner.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "(layout, grid_work, ao_capacity_fallback)"
    ]
    if len(selections) != 1 or ast.unparse(selections[0].value) != (
        "_plan_stationary_optional_ao_schedule(grid_points=len(state.grid.points), "
        "tile_points=tile_points, admit=admit_tile, preferred_tile_points=preferred_tile_points, "
        "requested_maps=resident_ao_cutoff is not None)"
    ):
        raise RuntimeError("stationary CUDA tile schedule binding changed")
    if (
        len(page_execution_calls) != 1
        or len(host_gates) != 1
        or page_execution_calls[0].lineno <= selections[0].end_lineno
    ):
        raise RuntimeError("stationary CUDA primitive descriptor page order changed")
    submit_pages = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.FunctionDef) and node.name == "submit_page"
    ]
    if len(submit_pages) != 1:
        raise RuntimeError("stationary CUDA submit-page owner is ambiguous")
    submit_page_digest = _source_node_sha256(source, submit_pages[0])
    if submit_page_digest != STATIONARY_SUBMIT_PAGE_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA submit-page contract changed")
    page_contract["submit_page_sha256"] = submit_page_digest
    nuclear_pair_loops = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.For)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.For)
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "sources.nuclear"
            for call in ast.walk(node.body[0])
        )
    ]
    if len(nuclear_pair_loops) != 1:
        raise RuntimeError("stationary CUDA nuclear-pair loop is ambiguous")
    nuclear_pair_loop_digest = _source_node_sha256(source, nuclear_pair_loops[0])
    if nuclear_pair_loop_digest != STATIONARY_NUCLEAR_PAIR_LOOP_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA nuclear-pair loop contract changed")
    page_contract["nuclear_pair_loop_sha256"] = nuclear_pair_loop_digest
    # Hashes retain complete fail-closed coverage; these structural checks explain
    # the critical admission ordering and concurrent host ownership explicitly.
    native_requirement = definition_nodes["requires_native_integrals"][0]
    grid_work = definition_nodes["grid_work"][0]
    if (
        not native_requirement.lineno
        < definition_nodes["records"][0].lineno
        < grid_work.lineno
        < selections[0].lineno
    ):
        raise RuntimeError("stationary CUDA admission gate order changed")
    reserve_additions = [
        node
        for node in resource_owner.body
        if isinstance(node, ast.AugAssign)
        and ast.unparse(node.target) == "host_bound"
        and isinstance(node.op, ast.Add)
        and ast.unparse(node.value) == "native_integral_host_reserve"
    ]
    if (
        len(reserve_additions) != 1
        or not definition_nodes["native_integral_host_reserve"][0].lineno
        < reserve_additions[0].lineno
        < host_gates[0].lineno
    ):
        raise RuntimeError(
            "stationary CUDA native host-reserve admission order changed"
        )
    resource_owner_digest = _source_node_sha256(source, resource_owner)
    if resource_owner_digest != STATIONARY_TILE_RESOURCE_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA ordinary tile-resource contract changed")
    page_contract["ordinary_tile_resources_sha256"] = resource_owner_digest
    signature = inspect.signature(complete_rks_cuda_gradient_diagnostic)
    if signature.parameters["resident_ao_cutoff"].default is not None:
        raise RuntimeError("stationary CUDA default AO membership changed")
    for name, expected in (
        ("_stationary_ao_map_reserve", STATIONARY_AO_MAP_RESERVE_CONTRACT_SHA256),
        (
            "_stationary_device_ao_map_reserve",
            STATIONARY_DEVICE_AO_MAP_RESERVE_CONTRACT_SHA256,
        ),
        ("_stationary_resident_ao_cache", STATIONARY_AO_MAP_CACHE_CONTRACT_SHA256),
    ):
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(helpers) != 1 or _source_node_sha256(source, helpers[0]) != expected:
            raise RuntimeError(f"stationary CUDA {name} contract changed")
        page_contract[f"{name.removeprefix('_stationary_')}_sha256"] = expected

    def default(name: str) -> int:
        value = signature.parameters[name].default
        if type(value) is not int:
            raise RuntimeError(f"stationary CUDA {name} default is not an integer")
        return value

    messages = (
        "enlarged stationary CUDA domains require prepared native integral derivatives",
        "primitive work count exceeds uint64 metric range",
        "stationary additional-device budget exceeded",
        "stationary additional-host byte budget exceeded",
        "native stationary host staging exceeds admitted reserve",
        "prepared native integral derivatives are unavailable within the admitted ",
    )
    positions = [source.find(message) for message in messages]
    if any(position < 0 for position in positions):
        raise RuntimeError("stationary CUDA admission messages are incomplete")
    # The helper is declared before its caller; source-file order no longer
    # represents execution. Preserve order within each owner and bind the
    # actual selection-before-execution ordering separately above.
    for indices in ((0, 1, 4, 5), (2, 3)):
        ordered = [positions[index] for index in indices]
        if ordered != sorted(ordered):
            raise RuntimeError("stationary CUDA admission gate order changed")
    endpoint_owner_digest = _source_node_sha256(source, owner)
    page_contract["endpoint_owner_sha256"] = endpoint_owner_digest
    _resident_ao_policy_contract(tree)

    return {
        "owner": STATIONARY_OWNER,
        "resource_owner": "python/generativeqc_compiler/method/stationary_resources.py",
        "native_owner_capacity": {
            "atom_count": STATIONARY_MAX_ATOMS,
            "ao_count": STATIONARY_MAX_AOS,
            "basis_primitive_count": STATIONARY_MAX_PRIMITIVES,
        },
        "ao_task_fallback_capacity": {
            "atom_count": 32,
            "ao_count": 128,
            "basis_primitive_count": 4096,
        },
        "phased_becke_auto_min_atoms": 48,
        "becke_primitive_requested": _resolve_becke_primitive_policy(),
        "native_integral_requirement_definition": NATIVE_REQUIREMENT_DEFINITION,
        "native_integral_host_reserve_definition": NATIVE_HOST_RESERVE_DEFINITION,
        "host_bound_total_definition": "host_bound + native_integral_host_reserve",
        "grid_work_definition": GRID_WORK_DEFINITION,
        "diagnostic_work_limits": {
            "grid_points": default("max_grid_points"),
            "grid_pair_visits": default("max_grid_pair_visits"),
        },
        "grid_work_capacity": {
            "grid_points": 1 << 40,
            "grid_pair_visits": (1 << 64) - 1,
        },
        "pending_grid_tiles": default("max_pending_grid_tiles"),
        "pending_grid_pair_visits": default("max_pending_grid_pair_visits"),
        "primitive_records": default("max_primitive_records"),
        "primitive_records_scope": "per_native_page_on_ao_task_fallback_only",
        "primitive_logical_metric_limit": (1 << 64) - 1,
        "primitive_sum_definition": PRIMITIVE_SUM_DEFINITION,
        "primitive_records_definition": PRIMITIVE_RECORDS_DEFINITION,
        "snapshot_functional_contract_sha256": snapshot_functional_contract,
        "primitive_page_contract_sha256": page_contract,
        "primitive_page_budget_bindings": list(page_budget_bindings),
        "fixed_task_capacity_definition": expected_executor_definitions[
            "fixed_task_capacity"
        ],
        "task_executor_definition": expected_executor_definitions["task_executor"],
        "primitive_page_execution_definition": (
            "task_executor.execute_pages(domain, submit_page)"
        ),
        "grid_pair_visits_definition": GRID_PAIR_VISITS_DEFINITION,
        "method_ir_definition": METHOD_IR_DEFINITION,
        "functional_lowering_definition": FUNCTIONAL_LOWERING_DEFINITION,
        "functional_ingredients_definition": INGREDIENTS_DEFINITION,
        "grid_derivative_order_definition": NEEDS_FIRST_DEFINITION,
        "grid_plan_definition": GRID_PLAN_DEFINITION,
        "minimum_source_bytes_definition": MINIMUM_SOURCE_BYTES_DEFINITION,
        "source_resources_definition": SOURCE_RESOURCES_DEFINITION,
        "source_bytes_definition": SOURCE_BYTES_DEFINITION,
        "host_bound_definition": HOST_BOUND_DEFINITION,
        "available_device_bytes_definition": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "additional_device_admission": (
            "minimum_additional_device_bytes < additional_device_budget and additional_device_peak_bound <= additional_device_budget"
        ),
        "gate_predicates": dict(GATE_PREDICATES),
        "tile_points": default("tile_points"),
        "primitive_tile": default("primitive_tile"),
        "integral_terms": default("integral_terms"),
        "additional_device_bytes": default("max_device_bytes"),
        "additional_host_bytes": default("max_host_bytes"),
        "gate_order": [
            "native_owner_capacity",
            "native_integral_provider_required",
            "primitive_logical_metric_range",
            "grid_work_capacity",
            "grid_point_work_budget",
            "grid_pair_work_budget",
            "pending_grid_pair_budget",
            "additional_device_budget",
            "additional_host_budget",
            "native_integral_result_required",
            "primitive_descriptor_page_budget",
        ],
    }


def _grid_spec(payload: dict[str, Any]) -> GridSpec:
    values = dict(payload)
    values["element_radii"] = tuple(tuple(item) for item in values["element_radii"])
    return GridSpec(**values)


def _calculator_basis_contract(repository: Path) -> tuple[str, str]:
    """Exercise the census's Gaussian lowering without a native library.

    Source digests remain report provenance, not admission gates. Run the current
    methods against the real immutable basis records and ctypes ABI, replacing
    only the native creation boundary and observing capability admission. This
    permits unrelated auxiliary consumers and behavior-preserving refactors while
    rejecting changes to the orbital/ordinary-DF shell and primitive census.
    """
    import ctypes
    from dataclasses import replace

    from generativeqc import calculator as calculator_module

    source = (repository / "python/generativeqc/calculator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    owners = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    ]
    if len(owners) != 1:
        raise RuntimeError("Calculator basis owner is missing or ambiguous")
    # Compile the current class, not an imported/stale method. Its constructor is
    # never run; all native calls terminate at the descriptor capture below.
    namespace = dict(vars(calculator_module))
    exec(  # noqa: S102 - current repository class, native boundary replaced
        compile(ast.Module(body=owners, type_ignores=[]), str(repository), "exec"),
        namespace,
    )
    calculator_type = namespace["Calculator"]
    native = calculator_module._native
    atoms = (Atom(6, (0.25, -0.5, 0.75)), Atom(1, (1.5, 0.25, -0.125)))
    context = ctypes.c_void_p(101)
    returned_system = 202
    preflight_seen = False
    reject_preflight = False
    native_seen = False
    expected_route = "ordinary"
    native_status = native.STATUS_SUCCESS
    expected_basis: Any = None
    expected_options: dict[str, Any] = {}
    expected_shells: tuple[Any, ...] = ()
    expected_representation = native.BASIS_SPHERICAL

    class AdmissionRejected(Exception):
        pass

    def observe_admission(basis: Any, observed_atoms: Any, **options: Any) -> None:
        nonlocal preflight_seen
        preflight_seen = True
        options.setdefault("consumer", None)
        if (
            basis != expected_basis
            or tuple(observed_atoms) != atoms
            or options != expected_options
        ):
            raise RuntimeError("basis capability inputs changed")
        if reject_preflight:
            raise AdmissionRejected
        calculator_module.require_basis(basis, observed_atoms, **options)

    def capture_system(
        observed_context: Any,
        descriptor_pointer: Any,
        system_pointer: Any,
        *,
        route: str = "ordinary",
    ) -> int:
        nonlocal native_seen
        native_seen = True
        descriptor = ctypes.cast(
            descriptor_pointer, ctypes.POINTER(native.SystemDescriptor)
        ).contents
        primitives = tuple(p for shell in expected_shells for p in shell.primitives)
        header = (
            descriptor.struct_size,
            descriptor.abi_version,
            descriptor.atom_count,
            descriptor.shell_count,
            descriptor.primitive_count,
            descriptor.charge,
            descriptor.multiplicity,
            descriptor.basis_representation,
        )
        if (
            route != expected_route
            or getattr(observed_context, "value", None) != context.value
            or header
            != (
                ctypes.sizeof(native.SystemDescriptor),
                native.ABI_VERSION,
                len(atoms),
                len(expected_shells),
                len(primitives),
                -1,
                2,
                expected_representation,
            )
        ):
            raise RuntimeError("native basis descriptor dimensions changed")
        actual_atoms = tuple(
            (atom.atomic_number, (atom.x, atom.y, atom.z))
            for atom in descriptor.atoms[: descriptor.atom_count]
        )
        actual_shells = tuple(
            (
                shell.atom_index,
                shell.angular_momentum,
                shell.primitive_offset,
                shell.primitive_count,
            )
            for shell in descriptor.shells[: descriptor.shell_count]
        )
        actual_primitives = tuple(
            (p.exponent, p.coefficient)
            for p in descriptor.primitives[: descriptor.primitive_count]
        )
        expected_shell_records = tuple(
            (
                shell.atom_index,
                shell.angular_momentum,
                sum(len(previous.primitives) for previous in expected_shells[:i]),
                len(shell.primitives),
            )
            for i, shell in enumerate(expected_shells)
        )
        if (
            actual_atoms != tuple((atom.atomic_number, atom.position) for atom in atoms)
            or actual_shells != expected_shell_records
            or actual_primitives
            != tuple((p.exponent, p.coefficient) for p in primitives)
        ):
            raise RuntimeError("native basis descriptor values changed")
        ctypes.cast(system_pointer, ctypes.POINTER(ctypes.c_void_p))[0] = (
            returned_system
        )
        return native_status

    def capture_ecp(
        observed_context: Any,
        descriptor: Any,
        cores: Any,
        terms: Any,
        count: int,
        system: Any,
    ) -> int:
        if (
            tuple(cores) != (2, 0)
            or count != 1
            or tuple(
                getattr(terms[0], field)
                for field in (
                    "atom_index",
                    "channel",
                    "power",
                    "exponent",
                    "coefficient",
                )
            )
            != (0, -1, 2, 0.8, -2.0)
        ):
            raise RuntimeError("native ECP descriptor changed")
        return capture_system(observed_context, descriptor, system, route="ecp")

    namespace["require_basis"] = observe_admission
    label = "_shells_for_atoms"
    try:
        for representation in ("spherical", "cartesian"):
            orbital = snapshot_basis("def2-svp", representation)
            auxiliary = snapshot_basis("sto-3g", representation)
            expected_representation = (
                native.BASIS_SPHERICAL
                if representation == "spherical"
                else native.BASIS_CARTESIAN
            )
            for device in ("cpu", "cuda"):
                for form in (orbital, "def2-svp", orbital.shells_for(atoms)):
                    calculator = object.__new__(calculator_type)
                    calculator._basis = form
                    calculator._representation_name = representation
                    calculator._basis_representation = expected_representation
                    calculator._device_name = device
                    calculator._density_fitting_mode = (
                        native.DENSITY_FITTING_CPU_REFERENCE
                    )
                    calculator._method = native.METHOD_PBE_RKS
                    calculator._dispersion_method_ir = None
                    calculator._composite_rks_second_order = False
                    calculator._library = SimpleNamespace(
                        generativeqc_system_create=capture_system,
                        generativeqc_system_create_ecp=capture_ecp,
                        generativeqc_correlation_auxiliary_system_create_v1=lambda *args: (
                            capture_system(*args, route="correlation")
                        ),
                        generativeqc_status_message=lambda status: (
                            b"capacity probe rejected"
                        ),
                        generativeqc_context_get_last_detail=lambda context: None,
                    )
                    for explicit_basis in (None, auxiliary):
                        expected_basis = snapshot_basis(
                            form if explicit_basis is None else explicit_basis,
                            representation,
                        )
                        expected_shells = (
                            expected_basis.shells_for(atoms)
                            if isinstance(expected_basis, BasisSet)
                            else tuple(expected_basis)
                        )
                        for operator, derivative_order in ((None, 0), ("kinetic", 1)):
                            expected_options = {
                                "backend": device if explicit_basis is None else "cpu",
                                "operator": operator
                                or ("eri" if explicit_basis is None else "df_metric"),
                                "derivative_order": derivative_order,
                                "role": "orbital"
                                if explicit_basis is None
                                else "auxiliary",
                                "representation": representation,
                                "consumer": None,
                            }
                            label = "_shells_for_atoms"
                            preflight_seen = False
                            shells = calculator._shells_for_atoms(
                                atoms,
                                explicit_basis,
                                operator=operator,
                                derivative_order=derivative_order,
                            )
                            if not preflight_seen or tuple(shells) != expected_shells:
                                raise RuntimeError("basis shell expansion changed")
                        expected_options.update(
                            operator="eri" if explicit_basis is None else "df_metric",
                            derivative_order=0,
                        )
                        label = "_create_native_system"
                        preflight_seen = native_seen = False
                        system = calculator._create_native_system(
                            context, atoms, -1, 2, explicit_basis
                        )
                        if (
                            not preflight_seen
                            or not native_seen
                            or system.value != returned_system
                        ):
                            raise RuntimeError("native basis forwarding changed")
                        # A failed capability preflight must reach neither the
                        # native allocator nor a successful shell/result return.
                        reject_preflight = True
                        for label, arguments in (
                            ("_shells_for_atoms", (atoms, explicit_basis)),
                            (
                                "_create_native_system",
                                (context, atoms, -1, 2, explicit_basis),
                            ),
                        ):
                            native_seen = False
                            try:
                                getattr(calculator, label)(*arguments)
                            except AdmissionRejected:
                                if native_seen:
                                    raise RuntimeError(
                                        "native allocation preceded basis admission"
                                    ) from None
                            else:
                                raise RuntimeError(
                                    "basis admission failure was swallowed"
                                )
                        reject_preflight = False
        # Keep scoped correlation auxiliary creation separate from the ordinary
        # orbital ABI, including a library that lacks the optional entry point.
        calculator._basis = orbital
        calculator._device_name = "cuda"
        calculator._density_fitting_mode = native.DENSITY_FITTING_CORRELATION_CUDA
        calculator._method = native.METHOD_MP2
        for explicit_basis in (None, auxiliary):
            expected_basis = orbital if explicit_basis is None else auxiliary
            expected_shells = expected_basis.shells_for(atoms)
            expected_options.update(
                backend="cuda",
                role="orbital" if explicit_basis is None else "auxiliary",
                operator="eri" if explicit_basis is None else "df_metric",
                consumer=None if explicit_basis is None else "correlation_df",
            )
            expected_route = "ordinary" if explicit_basis is None else "correlation"
            calculator._create_native_system(context, atoms, -1, 2, explicit_basis)
        del calculator._library.generativeqc_correlation_auxiliary_system_create_v1
        native_seen = False
        try:
            calculator._create_native_system(context, atoms, -1, 2, auxiliary)
        except NotImplementedError:
            if native_seen:
                raise RuntimeError(
                    "missing auxiliary ABI reached native creation"
                ) from None
        else:
            raise RuntimeError("missing auxiliary ABI did not fail closed")

        ecp_element = replace(
            orbital.by_element[6],
            ecp_core_electrons=2,
            ecp_data=json.dumps(
                [
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [0],
                        "r_exponents": [2],
                        "gaussian_exponents": ["0.8"],
                        "coefficients": [["-2"]],
                    }
                ]
            ),
        )
        calculator._basis = expected_basis = replace(
            orbital, elements=(ecp_element, orbital.by_element[1])
        )
        calculator._density_fitting_mode = native.DENSITY_FITTING_NONE
        expected_shells = expected_basis.shells_for(atoms)
        expected_options.update(role="orbital", operator="eri", consumer=None)
        expected_route = "ecp"
        calculator._create_native_system(context, atoms, -1, 2)
        calculator._density_fitting_mode = native.DENSITY_FITTING_CUDA
        native_seen = False
        try:
            calculator._create_native_system(context, atoms, -1, 2)
        except NotImplementedError:
            if native_seen:
                raise RuntimeError(
                    "ECP density fitting reached native creation"
                ) from None
        else:
            raise RuntimeError("ECP density fitting did not fail closed")

        calculator._basis = expected_basis = orbital
        expected_shells = orbital.shells_for(atoms)
        expected_route = "ordinary"
        for charge, multiplicity in ((2**31, 2), (-1, 0)):
            native_seen = False
            try:
                calculator._create_native_system(context, atoms, charge, multiplicity)
            except ValueError:
                if native_seen:
                    raise RuntimeError(
                        "invalid electron state reached native creation"
                    ) from None
            else:
                raise RuntimeError("invalid electron state did not fail closed")
        native_status = native.STATUS_INVALID_ARGUMENT
        try:
            calculator._create_native_system(context, atoms, -1, 2)
        except RuntimeError as error:
            if "capacity probe rejected" not in str(error):
                raise
        else:
            raise RuntimeError("native basis creation failure was swallowed")
    except Exception as error:
        raise RuntimeError(
            f"{label} basis lowering contract changed: {error}"
        ) from error
    functions = {
        node.name: node for node in owners[0].body if isinstance(node, ast.FunctionDef)
    }
    return tuple(
        _source_node_sha256(source, functions[name])
        for name in ("_shells_for_atoms", "_create_native_system")
    )


def _basis_layout_contract(repository: Path) -> dict[str, str]:
    source = (repository / "python/generativeqc_compiler/dft/ao.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeAO"
    ]
    if len(classes) != 1:
        raise RuntimeError("NativeAO capacity owner is missing or ambiguous")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("NativeAO capacity constructor is missing or ambiguous")
    constructor_digest = _source_node_sha256(source, constructors[0])
    packed = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "packed"
    ]
    numeric = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and ast.unparse(node.targets[0]) == "self.numeric_bytes"
    ]
    if packed != [BASIS_PACKED_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO packed capacity definition changed")
    if numeric != [BASIS_NUMERIC_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO numeric capacity definition changed")
    if constructor_digest != NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256:
        raise RuntimeError("NativeAO constructor contract changed")

    def function_digest(
        relative: str,
        function_name: str,
        expected: str,
        *,
        class_name: str | None = None,
    ) -> str:
        owner_source = (repository / relative).read_text(encoding="utf-8")
        owner_tree = ast.parse(owner_source)
        body: list[ast.stmt] = owner_tree.body
        if class_name is not None:
            owners = [
                node
                for node in owner_tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            if len(owners) != 1:
                raise RuntimeError(f"{class_name} basis owner is missing or ambiguous")
            body = owners[0].body
        functions = [
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ]
        if len(functions) != 1:
            raise RuntimeError(
                f"{function_name} basis lowering is missing or ambiguous"
            )
        digest = _source_node_sha256(owner_source, functions[0])
        if digest != expected:
            raise RuntimeError(f"{function_name} basis lowering contract changed")
        return digest

    snapshot_digest = function_digest(
        "python/generativeqc/_model_resolution.py",
        "snapshot_basis",
        BASIS_SNAPSHOT_CONTRACT_SHA256,
    )
    expansion_digest = function_digest(
        "python/generativeqc/basis.py",
        "shells_for",
        BASIS_SHELL_EXPANSION_CONTRACT_SHA256,
        class_name="BasisSet",
    )
    shell_forwarding_digest, native_system_digest = _calculator_basis_contract(
        repository
    )
    return {
        "packed_capacity_definition": packed[0],
        "numeric_capacity_definition": numeric[0],
        "native_ao_constructor_contract_sha256": constructor_digest,
        "basis_snapshot_contract_sha256": snapshot_digest,
        "basis_shell_expansion_contract_sha256": expansion_digest,
        "calculator_shell_forwarding_contract_sha256": shell_forwarding_digest,
        "native_system_basis_forwarding_contract_sha256": native_system_digest,
        "production_shell_expansion": (
            "snapshot_basis('def2-svp', 'spherical').shells_for(atoms)"
        ),
    }


def _spd_expansion_contract(repository: Path) -> dict[str, Any]:
    """Bind frozen s/p/d term counts to the native public-AO expansion table."""

    source = (repository / "src/molecule/basis.cpp").read_text(encoding="utf-8")
    try:
        begin = source.index("std::vector<CartesianComponent> cartesian_components")
        end = source.index("  if (l == 3)", begin)
    except ValueError as error:
        raise RuntimeError("native s/p/d expansion owner is missing") from error
    digest = _lf_sha256(source[begin:end].encode())
    if digest != SPD_EXPANSION_CONTRACT_SHA256:
        raise RuntimeError("native s/p/d expansion contract changed")

    packer_source = (repository / "src/dft/ao_grid.cpp").read_text(encoding="utf-8")
    try:
        packer_begin = packer_source.index("AoBasis::AoBasis(")
        packer_end = packer_source.index("void AoBasis::evaluate(", packer_begin)
    except ValueError as error:
        raise RuntimeError("native packed-AO owner is missing") from error
    packer_digest = _lf_sha256(packer_source[packer_begin:packer_end].encode())
    if packer_digest != AO_PACKER_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    bridge_source = (repository / "src/dft/bridge.cpp").read_text(encoding="utf-8")
    try:
        bridge_begin = bridge_source.index(
            "GENERATIVEQC_API int generativeqc_grid_basis_create_v1"
        )
        bridge_end = bridge_source.index(
            "GENERATIVEQC_API int generativeqc_grid_ao_v1", bridge_begin
        )
    except ValueError as error:
        raise RuntimeError("native AO pack bridge is missing") from error
    bridge_digest = _lf_sha256(bridge_source[bridge_begin:bridge_end].encode())
    if bridge_digest != AO_PACK_BRIDGE_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    ao_count_digest = _source_span_sha256(
        source,
        begin="std::size_t ao_count(",
        end="std::size_t cartesian_ao_count(",
        label="native spherical AO count",
    )
    if ao_count_digest != NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256:
        raise RuntimeError("native spherical AO count contract changed")

    shell_source = (repository / "src/molecule/basis.hpp").read_text(encoding="utf-8")
    shell_count_digest = _cpp_block_sha256(
        shell_source, "[[nodiscard]] constexpr std::size_t shell_ao_count("
    )
    if shell_count_digest != NATIVE_SHELL_AO_COUNT_CONTRACT_SHA256:
        raise RuntimeError("native shell AO count contract changed")
    cartesian_count_digest = _cpp_block_sha256(
        shell_source, "[[nodiscard]] constexpr std::size_t cartesian_count("
    )
    if cartesian_count_digest != NATIVE_CARTESIAN_SHELL_COUNT_CONTRACT_SHA256:
        raise RuntimeError("native Cartesian shell count contract changed")

    stationary_source = (
        repository / "python/generativeqc/_stationary_cuda.py"
    ).read_text(encoding="utf-8")
    stationary_tree = ast.parse(stationary_source)
    layouts = [
        node
        for node in stationary_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_layout"
    ]
    if len(layouts) != 1:
        raise RuntimeError("stationary layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(stationary_source, layouts[0])
    if layout_digest != STATIONARY_LAYOUT_CONTRACT_SHA256:
        raise RuntimeError("stationary layout contract changed")

    return {
        "spd_expansion_contract_sha256": digest,
        "ao_packer_contract_sha256": packer_digest,
        "ao_pack_bridge_contract_sha256": bridge_digest,
        "native_spherical_ao_count_contract_sha256": ao_count_digest,
        "native_shell_ao_count_contract_sha256": shell_count_digest,
        "native_cartesian_shell_count_contract_sha256": cartesian_count_digest,
        "stationary_layout_contract_sha256": layout_digest,
        "sparse_spherical_component_terms": {
            "s": SPARSE_SPHERICAL_COMPONENT_TERMS[0],
            "p": SPARSE_SPHERICAL_COMPONENT_TERMS[1],
            "d": SPARSE_SPHERICAL_COMPONENT_TERMS[2],
        },
        "spd_expansion_owner": (
            "basis.cpp::cartesian_components+ao_expansions -> "
            "ao_grid.cpp::AoBasis -> bridge.cpp::generativeqc_grid_basis_pack_v1 -> "
            "_stationary_cuda.py::_layout"
        ),
    }


def _basis_shape(
    atoms: tuple[Atom, ...], charge: int, multiplicity: int
) -> tuple[dict[str, Any], Any]:
    basis = snapshot_basis("def2-svp", "spherical")
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        raise RuntimeError(
            "production basis snapshot did not retain spherical def2-SVP"
        )
    shells = basis.shells_for(atoms)
    if any(
        shell.angular_momentum not in SPARSE_SPHERICAL_COMPONENT_TERMS
        for shell in shells
    ):
        raise NotImplementedError(
            "DFT-MP-v1 capacity audit only covers the frozen s/p/d def2-SVP domain"
        )
    ao_count = sum(2 * shell.angular_momentum + 1 for shell in shells)
    primitive_count = sum(len(shell.primitives) for shell in shells)
    ao_primitive_peak = max(len(shell.primitives) for shell in shells)
    component_primitive_sum = sum(
        len(shell.primitives) * SPARSE_SPHERICAL_COMPONENT_TERMS[shell.angular_momentum]
        for shell in shells
    )
    packed = np.empty(3 * len(atoms) + 2 * primitive_count + 16 * ao_count)
    numeric_bytes = (
        2 * packed.nbytes + 32 * len(atoms) + 32 * len(shells) + 16 * primitive_count
    )
    synthetic = SimpleNamespace(
        nao=ao_count,
        natom=len(atoms),
        nprimitive=primitive_count,
        numeric_bytes=numeric_bytes,
        packed=packed,
    )
    record = _named_basis_record("def2-svp", "spherical")
    metadata = resolved_basis_metadata(
        record,
        shells,
        atoms,
        representation="spherical",
        charge=charge,
        multiplicity=multiplicity,
    )
    return (
        {
            "atom_count": len(atoms),
            "ao_count_spherical": ao_count,
            "shell_count": len(shells),
            "basis_primitive_count": primitive_count,
            "ao_primitive_count_peak": ao_primitive_peak,
            "component_primitive_sum": component_primitive_sum,
        },
        (synthetic, metadata),
    )


def _method_resources(
    basis: Any,
    *,
    atom_count: int,
    functional: int,
    spin: str,
    limits: dict[str, Any],
    plan: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    if plan is None:
        plan = _qualified_aot_plan(functional, spin)
    source_names = stationary_runtime_sources(plan)
    tile_points = limits["tile_points"]
    primitive_tile = limits["primitive_tile"]
    integral_terms = limits["integral_terms"]
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=1 if functional == 0 else 2,
        tile_points=tile_points,
        # The frozen census calls the default no-map diagnostic endpoint.
        # Its force consumer borrows features and owns no indexed output staging.
        active_ao_capacity=None,
        features_only=True,
        # Estimate even a losing case so the report cannot hide it by raising
        # from the live budget gate before recording the required bytes.
        budget_bytes=(1 << 63) - 1,
    )
    shape = {
        "atoms": atom_count,
        "aos": basis.nao,
        "primitives": basis.nprimitive,
        "points": tile_points,
        "tasks": primitive_tile,
        "spins": plan.spin_blocks,
        "sources": len(source_names),
    }
    minimum = stationary_cuda_allocation_bytes(
        **shape, geometry_lanes=min(32, tile_points)
    )
    native_host_reserve = stationary_native_pair_reserve(
        atoms=atom_count, aos=basis.nao, primitives=basis.nprimitive
    )
    native_reserve = min(
        max(0, limits["additional_device_bytes"] - grid_plan.peak_bytes - minimum),
        native_host_reserve,
    )
    resources = plan_stationary_cuda_resources(
        **shape,
        target=cuda_target_info("sm_120"),
        # Preserve a losing minimum-byte requirement instead of hiding it by
        # throwing before the capacity report records the failure.
        budget_bytes=max(
            minimum,
            limits["additional_device_bytes"] - grid_plan.peak_bytes - native_reserve,
        ),
        phased_becke=_resolve_phased_becke_policy(atom_count, None),
        becke_primitive=bool(_resolve_becke_primitive_policy()),
    )
    source_bytes = resources.allocation_bytes
    device_bound = grid_plan.peak_bytes + source_bytes
    host_bound = grid_plan.host_bytes + 8 * (
        82 * primitive_tile
        + 4 * plan.spin_blocks * basis.nao * basis.nao
        + 120 * atom_count
        + 12 * (len(source_names) - len(STATIONARY_RUNTIME_SOURCE_NAMES)) * atom_count
        + 26 * integral_terms
        + len(QUALIFIED_SPD_COMPONENTS) ** 4
        + 3 * len(QUALIFIED_SPD_COMPONENTS) ** 2
        + 3 * tile_points
        + 2 * basis.nprimitive
        + 11 * basis.nao
        + 80
    )
    return (
        {
            "resource_scope": (
                "additional stationary/grid and Direct paired-provider numeric owners; "
                "excludes retained SCF state, snapshot exports, Python/compiler objects and driver/modules"
            ),
            "grid_tile_peak_bytes": grid_plan.peak_bytes,
            "grid_active_ao_capacity": grid_plan.active_ao_capacity,
            "grid_features_only": grid_plan.features_only,
            "stationary_source_bytes": source_bytes,
            "stationary_geometry_lanes": resources.geometry_lanes,
            "stationary_geometry_scratch_bytes": resources.geometry_scratch_bytes,
            "stationary_center_geometry_bytes": resources.center_geometry_bytes,
            "stationary_phased_becke_bytes": resources.phased_becke_bytes,
            "stationary_grid_device_peak_bound": device_bound,
            # The provider has not executed. Charge its reserved allowance,
            # rather than reporting the stationary/grid owners as the full peak.
            "additional_device_peak_bound": device_bound + native_reserve,
            "native_integral_device_budget": max(
                0, limits["additional_device_bytes"] - device_bound
            ),
            "minimum_additional_device_bytes": grid_plan.peak_bytes + minimum,
            "stationary_native_pair_reserve_bytes": native_reserve,
            "additional_device_budget": limits["additional_device_bytes"],
            "stationary_native_integral_host_reserve_bytes": native_host_reserve,
            "additional_host_numeric_bound": host_bound + native_host_reserve,
            "additional_host_budget": limits["additional_host_bytes"],
        },
        plan,
    )


def _failure(
    gate: str,
    message: str,
    *,
    required: Any,
    cap: Any,
    exceeded: list[str] | None = None,
) -> dict[str, Any]:
    resource_function = (
        "stationary_cuda_requires_native_integrals"
        if gate == "native_owner_capacity"
        else "plan_stationary_cuda_grid_work"
        if gate
        in (
            "grid_work_capacity",
            "grid_point_work_budget",
            "grid_pair_work_budget",
            "pending_grid_pair_budget",
        )
        else None
    )
    result = {
        "gate": gate,
        "owner": STATIONARY_OWNER
        if resource_function is None
        else {
            "file": "python/generativeqc_compiler/method/stationary_resources.py",
            "function": resource_function,
        },
        "message": message,
        "required": required,
        "cap": cap,
    }
    if exceeded is not None:
        result["exceeded"] = exceeded
    return result


def _case_failures(
    shape: dict[str, Any],
    requirements: dict[str, Any],
    memory: dict[str, Any],
    limits: dict[str, Any],
    *,
    work_mode: str = "public",
) -> list[dict[str, Any]]:
    failures = []
    capacity = limits["native_owner_capacity"]
    for name, actual in (
        ("atom_count", shape["atom_count"]),
        ("ao_count", shape["ao_count_spherical"]),
        ("basis_primitive_count", shape["basis_primitive_count"]),
    ):
        if not 1 <= actual <= capacity[name]:
            failures.append(
                _failure(
                    "native_owner_capacity",
                    "stationary CUDA "
                    + {
                        "atom_count": "atoms",
                        "ao_count": "aos",
                        "basis_primitive_count": "primitives",
                    }[name]
                    + " exceeds resource caps",
                    required={name: actual},
                    cap={name: capacity[name]},
                    exceeded=[name],
                )
            )
    for key in ("grid_points", "grid_pair_visits"):
        minimum = 1 if key == "grid_points" else 0
        if not minimum <= requirements[key] <= limits["grid_work_capacity"][key]:
            failures.append(
                _failure(
                    "grid_work_capacity",
                    "stationary grid work exceeds resource/metric range",
                    required={key: requirements[key]},
                    cap={key: limits["grid_work_capacity"][key]},
                )
            )
    for key, gate, message in (
        ("grid_points", "grid_point_work_budget", "grid point work budget exceeded"),
        ("grid_pair_visits", "grid_pair_work_budget", "grid work budget exceeded"),
    ):
        cap = limits[f"{work_mode}_work_limits"][key]
        if cap is not None and requirements[key] > cap:
            failures.append(
                _failure(gate, message, required=requirements[key], cap=cap)
            )
    tile_visits = (
        2
        * min(requirements["grid_points"], limits["tile_points"])
        * shape["atom_count"]
        * (shape["atom_count"] - 1)
        // 2
    )
    if tile_visits > limits["pending_grid_pair_visits"]:
        failures.append(
            _failure(
                "pending_grid_pair_budget",
                "stationary grid tile exceeds pending pair-visit budget",
                required=tile_visits,
                cap=limits["pending_grid_pair_visits"],
            )
        )
    if requirements["primitive_records"] > limits["primitive_logical_metric_limit"]:
        failures.append(
            _failure(
                "primitive_logical_metric_range",
                "primitive work count exceeds uint64 metric range",
                required=requirements["primitive_records"],
                cap=limits["primitive_logical_metric_limit"],
            )
        )
    if (
        memory.get(
            "minimum_additional_device_bytes", memory["additional_device_peak_bound"]
        )
        >= limits["additional_device_bytes"]
        or memory["additional_device_peak_bound"] > limits["additional_device_bytes"]
    ):
        failures.append(
            _failure(
                "additional_device_budget",
                "stationary additional-device budget exceeded",
                required=memory["additional_device_peak_bound"],
                cap=limits["additional_device_bytes"],
            )
        )
    if memory["additional_host_numeric_bound"] > limits["additional_host_bytes"]:
        failures.append(
            _failure(
                "additional_host_budget",
                "stationary additional-host byte budget exceeded",
                required=memory["additional_host_numeric_bound"],
                cap=limits["additional_host_bytes"],
            )
        )
    fallback_capacity = limits["ao_task_fallback_capacity"]
    requires_native = (
        shape["atom_count"] > fallback_capacity["atom_count"]
        or shape["ao_count_spherical"] > fallback_capacity["ao_count"]
        or shape["basis_primitive_count"] > fallback_capacity["basis_primitive_count"]
    )
    if (
        not requires_native
        and requirements["primitive_descriptor_peak_records"]
        > limits["primitive_records"]
    ):
        failures.append(
            _failure(
                "primitive_descriptor_page_budget",
                "stationary CUDA descriptor exceeds primitive page work budget",
                required=requirements["primitive_descriptor_peak_records"],
                cap=limits["primitive_records"],
            )
        )
    return failures


def _admission_record(failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outcome": "blocked" if failures else "passes_static_stationary_caps",
        "first_blocker": failures[0] if failures else None,
        "failures": failures,
        "scope": (
            "static source/resource preflight only; no SCF, CUDA execution, "
            "native derivative-provider availability, AOT binary, numerical, or performance qualification"
        ),
    }


def _source_package_inventory(repository: Path) -> dict[str, Any]:
    cmake = (repository / "cmake/GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    profile_policy = (
        repository / "cmake/GenerativeQCStationaryProfiles.cmake"
    ).read_text(encoding="utf-8")
    names = QUALIFIED_STATIONARY_AOT_PROFILE_NAMES
    required = (
        "generativeqc_stationary_spd_primitives",
        'OUTPUT_NAME "generativeqc_stationary_${_generativeqc_stationary_name}_spd"',
        "generativeqc_select_stationary_aot_profiles(_generativeqc_stationary_names)",
    )
    missing = [token for token in required if token not in cmake]
    if missing:
        raise RuntimeError(
            "stationary s/p/d package declaration is incomplete: " + ", ".join(missing)
        )
    wiring_digest = _source_span_sha256(
        cmake,
        begin="    # The small-domain fallback primitives",
        end=(
            "  if(GENERATIVEQC_PYTHON_WHEEL)\n    generativeqc_attach_cuda_implib(${target})"
        ),
        label="stationary packaged-AOT CMake",
    )
    # Include the deterministic subset/default policy, not just device-link
    # wiring. The static inventory is not evidence that binaries are installed.
    contract_digest = hashlib.sha256(
        (wiring_digest + "\n" + profile_policy).encode()
    ).hexdigest()
    if contract_digest != STATIONARY_AOT_CMAKE_CONTRACT_SHA256:
        raise RuntimeError("stationary packaged-AOT CMake contract changed")
    return {
        "cmake_contract_sha256": contract_digest,
        "profiles": list(names),
        "component_domain": "spd",
    }


def _source_public_route(repository: Path) -> dict[str, Any]:
    """Fail closed if the source predicates supporting the reported route move."""

    calculator = (repository / "python/generativeqc/calculator.py").read_text(
        encoding="utf-8"
    )
    batch = (repository / "python/generativeqc/batch.py").read_text(encoding="utf-8")
    calculator_tree = ast.parse(calculator)
    calculator_classes = [
        node
        for node in calculator_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    ]
    if len(calculator_classes) != 1:
        raise RuntimeError("public Calculator owner is missing or ambiguous")
    constructors = [
        node
        for node in calculator_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("public Calculator constructor is missing or ambiguous")
    semilocal_assignments = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "semilocal_force"
            for target in node.targets
        )
    ]
    hybrid_assignments = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cuda_hybrid_force"
            for target in node.targets
        )
    ]
    force_promotions = [
        node
        for node in constructors[0].body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Name) and item.id == "semilocal_force"
            for item in ast.walk(node.test)
        )
    ]
    if (
        len(semilocal_assignments) != 1
        or len(hybrid_assignments) != 1
        or len(force_promotions) != 1
    ):
        raise RuntimeError("public CUDA force capability owner is ambiguous")
    semilocal_digest = _source_node_sha256(calculator, semilocal_assignments[0])
    hybrid_digest = _source_node_sha256(calculator, hybrid_assignments[0])
    promotion_digest = _source_node_sha256(calculator, force_promotions[0])
    if semilocal_digest != PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256:
        raise RuntimeError("public semilocal force predicate changed")
    if promotion_digest != PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256:
        raise RuntimeError("public force capability promotion changed")

    batch_tree = ast.parse(batch)
    batch_classes = [
        node
        for node in batch_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedBatch"
    ]
    if len(batch_classes) != 1:
        raise RuntimeError("public PreparedBatch owner is missing or ambiguous")
    force_methods = [
        node
        for node in batch_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_public_dft_cuda_force"
    ]
    if len(force_methods) != 1:
        raise RuntimeError("public CUDA force route is missing or ambiguous")
    kwargs_nodes = [
        node.value
        for node in ast.walk(force_methods[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "kwargs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    if len(kwargs_nodes) != 1:
        raise RuntimeError("public stationary kwargs are missing or ambiguous")
    kwargs = {
        ast.literal_eval(key): value
        for key, value in zip(kwargs_nodes[0].keys, kwargs_nodes[0].values, strict=True)
        if isinstance(key, ast.Constant)
    }
    work_limits = {}
    for key in ("grid_points", "grid_pair_visits"):
        node = kwargs.get(f"max_{key}")
        if not isinstance(node, ast.Constant) or node.value is not None:
            raise RuntimeError("public complete-grid work override changed")
        work_limits[key] = None
    batch_digest = _source_node_sha256(batch, force_methods[0])
    _public_resident_ao_policy_contract(force_methods[0])
    return {
        "semilocal_force_predicate_sha256": semilocal_digest,
        "global_hybrid_force_predicate_sha256": hybrid_digest,
        "force_capability_promotion_sha256": promotion_digest,
        "cuda_force_method_sha256": batch_digest,
        "whole_grid_work_limits": work_limits,
    }


def _check_public_grid_abi(lowerer: ast.FunctionDef) -> None:
    """Execute the source-bound lowerer and check its observable grid ABI.

    The private precision extension may change the function body without
    changing the grid. Retain its source hash as provenance, but require field
    behavior rather than a historical spelling. No native library is loaded.
    """
    import ctypes

    from generativeqc import _native
    from generativeqc import ks as ks_module

    namespace = dict(vars(ks_module))
    exec(  # noqa: S102 - execute the inspected repository ABI lowerer
        compile(
            ast.Module(body=[lowerer], type_ignores=[]), "<grid-abi-contract>", "exec"
        ),
        namespace,
    )
    for grid in (
        GridSpec(48, 16, 32),
        GridSpec(
            7,
            5,
            9,
            partition_iterations=2,
            coincident_tolerance=3e-11,
            element_radii=((1, 0.8), (8, 1.5)),
        ),
    ):
        for budget in (None, (1e-5, 1e-4, 3e-5)):
            options = resolve_ks_options(
                "pbe0-rks",
                KsOptions(grid=grid, tile_points=37, _precision_budget=budget),
            )
            descriptor = namespace[lowerer.name](options)
            expected = {
                "grid_version": grid.version,
                "radial_points": grid.radial_points,
                "angular_polar": grid.angular_polar,
                "angular_azimuth": grid.angular_azimuth,
                "partition_iterations": grid.partition_iterations,
                "coincident_tolerance": grid.coincident_tolerance,
                "tile_points": 37,
            }
            descriptor_type = (
                _native.KsOptionsDescriptor
                if budget is None
                else _native.ApplicationPrecisionKsOptionsDescriptor
            )
            if (
                any(
                    getattr(descriptor, name) != value
                    for name, value in expected.items()
                )
                or descriptor.struct_size != ctypes.sizeof(descriptor_type)
                or descriptor.abi_version != _native.ABI_VERSION
                or descriptor.element_radius_count != (119 if grid.element_radii else 0)
                or any(
                    descriptor.element_radii[z] != radius
                    for z, radius in grid.element_radii
                )
            ):
                raise RuntimeError(
                    "grid point-count contract changed: public grid ABI fields"
                )


def _grid_count_contract(repository: Path) -> dict[str, str]:
    """Record grid implementation provenance and validate the retained v2 ABI.

    Point construction, emitted buffer layout and snapshot decoding are covered
    by executable CPU/CUDA contracts, not historical implementation hashes.
    This source-only inventory is not numerical or real-device qualification.
    """

    python_grid = (repository / "python/generativeqc_compiler/dft/grid.py").read_text(
        encoding="utf-8"
    )
    python_tree = ast.parse(python_grid)
    grid_classes = [
        node
        for node in python_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MolecularGrid"
    ]
    if len(grid_classes) != 1:
        raise RuntimeError("source-only MolecularGrid owner is missing or ambiguous")
    post_init = [
        node
        for node in grid_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    ]
    if len(post_init) != 1:
        raise RuntimeError(
            "source-only MolecularGrid constructor is missing or ambiguous"
        )
    python_digest = _source_node_sha256(python_grid, post_init[0])

    public_ks = (repository / "python/generativeqc/ks.py").read_text(encoding="utf-8")
    public_ks_tree = ast.parse(public_ks)
    public_lowerers = [
        node
        for node in public_ks_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "native_ks_options"
    ]
    if len(public_lowerers) != 1:
        raise RuntimeError("public GridSpec ABI lowerer is missing or ambiguous")
    public_abi_digest = _source_node_sha256(public_ks, public_lowerers[0])
    _check_public_grid_abi(public_lowerers[0])

    quadrature_source = (
        repository / "python/generativeqc_compiler/xc/quadrature_cuda.py"
    ).read_text(encoding="utf-8")
    quadrature_tree = ast.parse(quadrature_source)
    layouts = [
        node
        for node in quadrature_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_LAYOUT"
            for target in node.targets
        )
    ]
    if len(layouts) != 1:
        raise RuntimeError("generated quadrature layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(quadrature_source, layouts[0])

    cuda_source = (repository / "src/dft/cuda_quadrature.cu").read_text(
        encoding="utf-8"
    )
    cuda_digest = _source_span_sha256(
        cuda_source,
        begin="MolecularGrid MolecularGrid::from_cuda(",
        end="}  // namespace generativeqc::dft",
        label="native CUDA grid",
    )
    route_source = (repository / "src/methods/dft_method.cpp").read_text(
        encoding="utf-8"
    )
    # Bind the complete routing function, not unrelated helpers inserted before
    # the following class. Actual grid-route mutations still fail the digest.
    route_digest = _cpp_block_sha256(
        route_source, "dft::MolecularGrid ks_molecular_grid("
    )
    # Record current reader provenance. Executable real-header ABI controls
    # protect legacy/v4 field transport and malformed extension rejection; a
    # static preflight is never a native execution qualification.
    preparation_source = (
        repository / "src/methods/dft_method_preparation.cpp"
    ).read_text(encoding="utf-8")
    native_abi_digest = _source_span_sha256(
        preparation_source,
        begin="dft::GridSpec ks_grid_options(",
        end="void validate_ks_spin_state(",
        label="native GridSpec ABI lowerer",
    )
    header_source = (repository / "src/dft/grid.hpp").read_text(encoding="utf-8")
    point_count_digest = _source_span_sha256(
        header_source,
        begin="  std::size_t point_count()",
        end="  const std::vector<double>& points()",
        label="native grid point-count publication",
    )
    contracts = {
        "source_only_molecular_grid_sha256": python_digest,
        "public_grid_abi_sha256": public_abi_digest,
        "native_grid_abi_sha256": native_abi_digest,
        "generated_quadrature_layout_sha256": layout_digest,
        "native_cuda_grid_sha256": cuda_digest,
        "native_cuda_grid_route_sha256": route_digest,
        "native_grid_point_count_sha256": point_count_digest,
    }
    expected = {
        "native_cuda_grid_route_sha256": NATIVE_GRID_ROUTE_CONTRACT_SHA256,
        "native_grid_point_count_sha256": NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256,
    }
    moved = [
        name
        for name, digest in contracts.items()
        if name in expected and digest != expected[name]
    ]
    if moved:
        raise RuntimeError("grid point-count contract changed: " + ", ".join(moved))
    return {
        **contracts,
        "point_count_definition": (
            "atom_count * radial_points * angular_polar * angular_azimuth"
        ),
    }


def _prepared_aot_route_contract(repository: Path) -> str:
    """Bind no-runtime-compilation claims to the prepared production owner."""

    source = (repository / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PreparedStationaryCudaExecution"
    ]
    methods = (
        []
        if len(classes) != 1
        else [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "ensure"
        ]
    )
    if len(methods) != 1:
        raise RuntimeError("prepared stationary AOT owner is missing or ambiguous")
    # The digest records provenance only. Execute the current source with inert
    # native/compiler boundaries so equivalent owner refactors remain valid.
    from tools.dft_mp_v1.prepared_contracts import check_prepared_selection

    digest = _source_node_sha256(source, methods[0])
    module = ModuleType("generativeqc._capacity_stationary_contract")
    module.__package__ = "generativeqc"
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        _DftMpSourceOnlyLoader(
            module.__name__, str(repository / "python/generativeqc/_stationary_cuda.py")
        ).exec_module(module)
        check_prepared_selection(module)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"prepared stationary AOT selection contract changed: {error}"
        ) from error
    finally:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    _resident_ao_policy_contract(tree)
    return digest


def _public_selector_contract(
    selector: str,
    *,
    expected_functional: int,
    expected_spin: str,
    stationary_plan: Any,
    grid_spec: GridSpec | None = None,
) -> dict[str, Any]:
    """Prove that a public selector resolves to the audited packaged plan."""

    method_ir, functional = resolve_ks_method(selector)
    expected_coefficients = ks_coefficients(method_ir)
    hybrid = expected_coefficients[2] != 0.0
    registry_selector = native_dft_carrier(selector) if hybrid else selector
    try:
        metadata = generated_methods.METHOD_METADATA[registry_selector]
        expected_abi = int(metadata["abi_id"])
    except KeyError as error:
        raise RuntimeError(f"unrecognized frozen public selector {selector}") from error
    if hybrid:
        if grid_spec is None:
            raise RuntimeError(
                "global-hybrid selector contract requires the frozen grid"
            )
        options = resolve_ks_options(selector, KsOptions(grid=grid_spec))
    else:
        options = resolve_ks_options(selector)
    native_functional = int(native_xc_functional_code(selector))
    public_stationary = StationaryGradientPlan(
        method_ir,
        StationaryMeanField(SCF_POINT_MODEL),
    )
    execution_plan = options.execution_plan
    exchange = tuple(execution_plan.exchange)
    failures = []
    if metadata["family"] != "density_functional" or metadata["provider"] != "dft":
        failures.append("family/provider")
    if not hybrid and SEMILOCAL_ABI_IDS.get(selector) != expected_abi:
        failures.append("native ABI ID")
    if expected_abi not in generated_methods.NATIVE_DFT_METHOD_IDS:
        failures.append("native DFT eligibility")
    if not metadata["supports_batch"] or "energy" not in metadata["properties"]:
        failures.append("batch/energy registry eligibility")
    if method_ir.spin != expected_spin or functional.spin != expected_spin:
        failures.append("spin")
    if native_functional != expected_functional:
        failures.append("native functional-family lowering")
    if options.coefficients != expected_coefficients:
        failures.append(
            "scientific coefficients" if hybrid else "semilocal coefficients"
        )
    if (
        execution_plan.method.identity != method_ir.identity
        or execution_plan.nonlocal_correlation is not None
        or execution_plan.post_scf
    ):
        failures.append("KS execution plan")
    if hybrid:
        if (
            len(exchange) != 1
            or exchange[0].operator != "full-range"
            or float(exchange[0].fock_coefficient) != expected_coefficients[2]
            or exchange[0].omega != 0
        ):
            failures.append("exact-exchange execution plan")
    elif exchange:
        failures.append("KS execution plan")
    if public_stationary.identity != stationary_plan.identity:
        failures.append("stationary plan identity")
    if failures:
        raise RuntimeError(
            f"public selector {selector} disagrees with packaged DFT plan: "
            + ", ".join(failures)
        )
    return {
        "selector": selector,
        "registry_selector": registry_selector,
        "native_abi_id": expected_abi,
        "native_dft_eligible": True,
        "supports_batch": True,
        "native_functional_code": native_functional,
        "spin": expected_spin,
        "coefficients": list(options.coefficients),
        "exchange": [term.semantic_payload() for term in exchange],
        "method_ir_identity": method_ir.identity,
        "ks_execution_plan_identity": execution_plan.identity,
        "stationary_plan_identity": public_stationary.identity,
    }


def _artifact_verification(
    directory: Path | None,
    *,
    functional: int,
    spin: str,
    plan: Any,
) -> dict[str, Any]:
    if directory is None:
        return {"status": "not_checked", "detail": "no AOT directory supplied"}
    try:
        artifact = load_stationary_aot_artifact(
            directory,
            functional=functional,
            spin=spin,
            plan=plan,
            architecture="sm_120",
            component_domain=QUALIFIED_SPD_COMPONENTS,
        )
    except OSError as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except (NotImplementedError, TypeError, ValueError) as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except KeyError as error:
        field = error.args[0] if error.args else "unknown"
        return {
            "status": "missing_or_invalid",
            "detail": f"missing AOT manifest field: {field}",
        }
    except AttributeError as error:
        return {
            "status": "missing_or_invalid",
            "detail": f"invalid AOT manifest schema: {error}",
        }
    return {
        "status": "verified",
        "detail": "packaged s/p/d AOT contract and binary identity verified",
        "library": str(Path(artifact.library).resolve()),
        "binary_sha256": artifact.metadata["binary_sha256"],
        "artifact_key": artifact.metadata["key"],
        "code_kinds": list(artifact.metadata["identity"]["target"]["code_kinds"]),
        "driver_ptx_jit_required": artifact.metadata["driver_ptx_jit_required"],
    }


def _validate_aot_evidence_directory(
    aot_directory: Path | None,
) -> Path | None:
    if aot_directory is None:
        return None
    aot_path = Path(aot_directory).absolute()

    def linked(candidate: Path) -> bool:
        junction = getattr(candidate, "is_junction", None)
        return candidate.is_symlink() or (junction is not None and junction())

    if any(linked(candidate) for candidate in (aot_path, *aot_path.parents)) or any(
        linked(candidate) for candidate in aot_path.rglob("*")
    ):
        raise ValueError("AOT evidence must not contain symlinks")
    return aot_path


def _build_report(
    repository: Path,
    *,
    source_sha: str,
    aot_directory: Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository).resolve()
    _assert_local_imports()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be a full lowercase Git commit SHA")
    _validate_aot_evidence_directory(aot_directory)
    _basis_pack.cache_clear()
    _named_basis_record.cache_clear()
    root = repository / "tools/dft_mp_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frozen_hash = manifest.get("contract_sha256")
    unhashed = dict(manifest)
    unhashed.pop("contract_sha256", None)
    if frozen_hash != _canonical_sha256(unhashed):
        raise ValueError("DFT-MP-v1 frozen contract hash mismatch")

    basis_pack = repository / "python/generativeqc/data/basis_pack.json"
    basis_pack_sha = _lf_sha256(basis_pack.read_bytes())
    basis_pack_matches = basis_pack_sha == manifest["basis"]["basis_pack_sha256"]
    if not basis_pack_matches:
        raise ValueError(
            "current basis pack differs from the frozen DFT-MP-v1 contract"
        )

    limits = _source_limits(repository)
    basis_layout = _basis_layout_contract(repository)
    spd_expansion = _spd_expansion_contract(repository)
    source_package = _source_package_inventory(repository)
    public_route = _source_public_route(repository)
    limits["public_work_limits"] = dict(public_route["whole_grid_work_limits"])
    public_route["registry_manifest_sha256"] = _lf_sha256(
        (repository / "manifests/public_methods.json").read_bytes()
    )
    public_route["prepared_aot_selection_sha256"] = _prepared_aot_route_contract(
        repository
    )
    grid_contract = _grid_count_contract(repository)
    grid_spec = _grid_spec(manifest["model"]["grid_spec"])

    required_rows = [
        row
        for row in manifest["rows"]
        if row["required"]
        and row["method"] in FP64_FORCE_METHODS
        and row["level"] == "fp64_energy_forces"
    ]
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in required_rows:
        rows_by_case.setdefault(row["case"], []).append(row)

    cases = []
    case_work: dict[str, dict[str, Any]] = {}
    ao_counts_match = True
    for case_name, frozen in sorted(manifest["cases"].items()):
        input_path = root / frozen["input"]
        raw = input_path.read_bytes()
        if _lf_sha256(raw) != frozen["input_sha256"]:
            raise ValueError(f"frozen input hash mismatch for {case_name}")
        value = json.loads(raw)
        atoms = tuple(Atom.from_value(atom) for atom in value["atoms"])
        changed_path = root / frozen["changed_input"]
        changed_raw = changed_path.read_bytes()
        if _lf_sha256(changed_raw) != frozen["changed_input_sha256"]:
            raise ValueError(f"frozen changed input hash mismatch for {case_name}")
        changed_value = json.loads(changed_raw)
        changed_atoms = tuple(Atom.from_value(atom) for atom in changed_value["atoms"])
        shape, (basis, basis_metadata) = _basis_shape(
            atoms, value["charge"], value["multiplicity"]
        )
        ao_match = shape["ao_count_spherical"] == frozen["ao_count_spherical"]
        ao_counts_match = ao_counts_match and ao_match
        if not ao_match:
            raise ValueError(
                f"actual spherical AO count differs from manifest for {case_name}"
            )
        grid = MolecularGrid(
            atoms,
            grid_spec,
            charge=value["charge"],
            multiplicity=value["multiplicity"],
        )
        if grid.identity != frozen["grid_identity"]:
            raise ValueError(
                f"current grid identity differs from manifest for {case_name}"
            )
        changed_grid = MolecularGrid(
            changed_atoms,
            grid_spec,
            charge=changed_value["charge"],
            multiplicity=changed_value["multiplicity"],
        )
        if changed_grid.identity != frozen["changed_grid_identity"]:
            raise ValueError(
                f"current changed grid identity differs from manifest for {case_name}"
            )
        native_grid_points = (
            len(atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        changed_native_grid_points = (
            len(changed_atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        if grid.npoint != native_grid_points or (
            changed_grid.npoint != changed_native_grid_points
        ):
            raise RuntimeError("source-only grid count disagrees with native contract")

        atom_pairs = shape["atom_count"] * (shape["atom_count"] - 1) // 2
        primitive_sum = shape["component_primitive_sum"]
        requirements = {
            "primitive_records": primitive_sum**4
            + (shape["atom_count"] + 2) * primitive_sum**2
            + atom_pairs,
            "primitive_descriptor_peak_records": shape["ao_primitive_count_peak"] ** 4,
            "grid_points": native_grid_points,
            "grid_pair_visits": (1 + 2 * native_grid_points) * atom_pairs,
        }
        grid_work = plan_stationary_cuda_grid_work(
            atoms=shape["atom_count"],
            grid_points=native_grid_points,
            tile_points=limits["tile_points"],
            max_grid_points=limits["public_work_limits"]["grid_points"],
            max_grid_pair_visits=limits["public_work_limits"]["grid_pair_visits"],
            max_pending_tiles=limits["pending_grid_tiles"],
            max_pending_pair_visits=limits["pending_grid_pair_visits"],
        )
        if grid_work.grid_pair_visits != requirements["grid_pair_visits"]:
            raise RuntimeError("compiler grid work disagrees with native census")
        native_integral_admission = {
            "required": stationary_cuda_requires_native_integrals(
                atoms=shape["atom_count"],
                aos=shape["ao_count_spherical"],
                primitives=shape["basis_primitive_count"],
            ),
            "provider_and_budget_qualification": "NOT_RUN",
            "enlarged_domain_failure_behavior": (
                "fail closed; enlarged domains cannot use AO-task fallback"
            ),
            "small_domain_fallback": "bounded AO-task route remains available",
            "stationary_primitive_records_if_native_complete": atom_pairs,
            "ao_task_descriptors_if_native_complete": 0,
        }
        method_memory = {}
        method_plans = {}
        method_requirements = {}
        for row in rows_by_case.get(case_name, ()):
            spin = "unpolarized" if row["spin"] == "rks" else "polarized"
            selector = f"{row['method']}-{row['spin']}"
            key = f"{row['method']}/{row['spin']}"
            if key not in method_memory:
                method_ir, _ = resolve_ks_method(selector)
                plan = StationaryGradientPlan(
                    method_ir,
                    StationaryMeanField(SCF_POINT_MODEL),
                )
                method_requirements[key] = {
                    **requirements,
                    "primitive_records": (
                        (1 + int(bool(method_ir.full_range_exact_exchange)))
                        * primitive_sum**4
                        + (shape["atom_count"] + 2) * primitive_sum**2
                        + atom_pairs
                    ),
                }
                method_memory[key], method_plans[key] = _method_resources(
                    basis,
                    atom_count=shape["atom_count"],
                    functional=int(native_xc_functional_code(selector)),
                    spin=spin,
                    limits=limits,
                    plan=plan,
                )
        admission_by_method_spin = {
            key: _admission_record(
                _case_failures(shape, method_requirements[key], memory, limits)
            )
            for key, memory in method_memory.items()
        }
        maximum_memory = {
            "minimum_additional_device_bytes": max(
                item["minimum_additional_device_bytes"]
                for item in method_memory.values()
            ),
            "additional_device_peak_bound": max(
                item["additional_device_peak_bound"] for item in method_memory.values()
            ),
            "additional_host_numeric_bound": max(
                item["additional_host_numeric_bound"] for item in method_memory.values()
            ),
        }
        maximum_requirements = {
            **requirements,
            "primitive_records": max(
                item["primitive_records"] for item in method_requirements.values()
            ),
        }
        failures = _case_failures(shape, maximum_requirements, maximum_memory, limits)
        record = {
            "id": case_name,
            "classification": frozen["classification"],
            "charge": value["charge"],
            "multiplicity": value["multiplicity"],
            "shape": {**shape, "grid_points": native_grid_points},
            "identities": {
                "input_sha256": frozen["input_sha256"],
                "changed_input_sha256": frozen["changed_input_sha256"],
                "basis": basis_metadata["mathematical_identity"],
                "basis_pack_sha256": basis_pack_sha,
                "ao_representation": "real_spherical/libcint-PySCF-order",
                "grid": grid.identity,
                "changed_grid": changed_grid.identity,
            },
            "requirements": {
                **maximum_requirements,
                "primitive_records_scope": "maximum logical AO-task reference across requested methods; not executed native work",
                "grid_work_plan": asdict(grid_work),
                "native_integral_admission": native_integral_admission,
                "memory_by_method_spin": method_memory,
            },
            "diagnostic_admission": _admission_record(
                _case_failures(
                    shape,
                    maximum_requirements,
                    maximum_memory,
                    limits,
                    work_mode="diagnostic",
                )
            ),
            "requested_rows": sorted(
                row["id"] for row in rows_by_case.get(case_name, ())
            ),
            "admission": _admission_record(failures),
            "admission_by_method_spin": admission_by_method_spin,
        }
        cases.append(record)
        case_work[case_name] = {
            "basis": basis,
            "plans": method_plans,
            "requirements": method_requirements,
            "memory": method_memory,
            "admission_by_method_spin": admission_by_method_spin,
            "record": record,
        }

    artifact_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    selector_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for frozen_row in sorted(required_rows, key=lambda item: item["id"]):
        method = frozen_row["method"]
        spin = "unpolarized" if frozen_row["spin"] == "rks" else "polarized"
        selector = f"{method}-{'rks' if spin == 'unpolarized' else 'uks'}"
        functional = int(native_xc_functional_code(selector))
        method_key = f"{method}/{frozen_row['spin']}"
        plan = case_work[frozen_row["case"]]["plans"][method_key]
        packaged = stationary_aot_profile_for_plan(functional, spin, plan) is not None
        aot_key = (functional, spin, plan.identity)
        if aot_key not in artifact_cache:
            artifact_cache[aot_key] = _artifact_verification(
                aot_directory,
                functional=functional,
                spin=spin,
                plan=plan,
            )
        if selector not in selector_cache:
            selector_cache[selector] = _public_selector_contract(
                selector,
                expected_functional=functional,
                expected_spin=spin,
                stationary_plan=plan,
                grid_spec=grid_spec,
            )
        native_properties = list(
            generated_methods.METHOD_METADATA[
                selector_cache[selector]["registry_selector"]
            ]["properties"]
        )
        rows.append(
            {
                **frozen_row,
                "stationary_plan": {
                    "identity": plan.identity,
                    "source_names": list(stationary_runtime_sources(plan)),
                    "component_domain": list(QUALIFIED_SPD_COMPONENTS),
                },
                "resource_requirements": {
                    "atom_count": case_work[frozen_row["case"]]["record"]["shape"][
                        "atom_count"
                    ],
                    "ao_count_spherical": case_work[frozen_row["case"]]["record"][
                        "shape"
                    ]["ao_count_spherical"],
                    **{
                        key: case_work[frozen_row["case"]]["requirements"][method_key][
                            key
                        ]
                        for key in (
                            "primitive_records",
                            "primitive_descriptor_peak_records",
                            "grid_points",
                            "grid_pair_visits",
                        )
                    },
                    "primitive_records_scope": "logical AO-task reference; not executed native work",
                    "grid_work_plan": case_work[frozen_row["case"]]["record"][
                        "requirements"
                    ]["grid_work_plan"],
                    "native_integral_admission": case_work[frozen_row["case"]][
                        "record"
                    ]["requirements"]["native_integral_admission"],
                    "primitive_page_work_budget": limits["primitive_records"],
                    **case_work[frozen_row["case"]]["memory"][method_key],
                },
                "public_capability": {
                    "forces": True,
                    "native_registry_properties": native_properties,
                    "promotion": "python direct-CUDA stationary-force predicate",
                    "owner": "python/generativeqc/calculator.py::Calculator.__init__",
                    "source_audited": True,
                    "selector_contract": selector_cache[selector],
                },
                "public_route": {
                    "scientific_runtime_compilation_required": not packaged,
                    "selection": (
                        "runtime-compiled stationary CUDA"
                        if not packaged
                        else "all-electron packaged stationary CUDA"
                    ),
                    "owner": "python/generativeqc/batch.py::_public_dft_cuda_force",
                    "missing_aot_behavior": (
                        "not selected by this public route"
                        if not packaged
                        else "fail closed; no NVCC fallback"
                    ),
                    "source_audited": True,
                },
                "packaged_aot": {
                    "name": _stationary_aot_name(
                        functional,
                        spin,
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                        plan=plan,
                    ),
                    "architecture": "sm_120",
                    "source_package_declared": True,
                    "selected_by_public_route": packaged,
                    "scope": "package availability evidence only; not execution qualification",
                    "source_owner": "cmake/GenerativeQCCuda.cmake",
                    "contract_identity": stationary_aot_profile_contract_identity(
                        f"{method}_{'rks' if spin == 'unpolarized' else 'uks'}",
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                    ),
                    "binary_verification": artifact_cache[aot_key],
                },
                "admission": case_work[frozen_row["case"]]["admission_by_method_spin"][
                    method_key
                ],
            }
        )

    owner_files = (
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/_ks_snapshot.py",
        "python/generativeqc/_snapshot_grid_cache.py",
        "python/generativeqc/resources_ks.py",
        "python/generativeqc/calculator.py",
        "python/generativeqc/batch.py",
        "python/generativeqc/ks.py",
        "python/generativeqc_compiler/method/stationary_cuda.py",
        "python/generativeqc_compiler/method/stationary_resources.py",
        "python/generativeqc_compiler/dft/ao.py",
        "python/generativeqc_compiler/dft/grid.py",
        "python/generativeqc_compiler/xc/quadrature_cuda.py",
        "src/molecule/basis.cpp",
        "src/molecule/basis.hpp",
        "src/dft/ao_grid.cpp",
        "src/dft/bridge.cpp",
        "src/dft/grid.hpp",
        "src/dft/cuda_quadrature.cu",
        "src/dft/stationary_gradient_cuda.cuh",
        "src/methods/dft_method.cpp",
        "src/methods/dft_method_preparation.cpp",
        "cmake/GenerativeQCCuda.cmake",
    )
    blocked = sum(row["admission"]["outcome"] == "blocked" for row in rows)
    return {
        "schema": SCHEMA,
        "source": {
            "sha": source_sha,
            "qualifier_sha256": _IMPORTED_TOOL_SOURCE_SHA256,
            "imported_module_sha256": {
                name: digest
                for name, (_, digest) in sorted(_IMPORTED_LOCAL_MODULE_SOURCES.items())
            },
            "runtime_owner_sha256": {
                path: _lf_sha256((repository / path).read_bytes())
                for path in owner_files
            },
        },
        "contract": {
            "id": manifest["contract_id"],
            "version": manifest["version"],
            "sha256": frozen_hash,
        },
        "basis": {
            "name": manifest["basis"]["name"],
            "representation": manifest["basis"]["representation"],
            "basis_pack_sha256": basis_pack_sha,
            "basis_pack_sha256_match": basis_pack_matches,
            "manifest_ao_counts_match": ao_counts_match,
            **basis_layout,
            **spd_expansion,
            "component_primitive_sum_derivation": (
                "sum(shell primitive count * sparse public-AO Cartesian term count); "
                "s=1, p=3, spherical d=8 from src/molecule/basis.cpp"
            ),
        },
        "grid": grid_contract,
        "public_route": public_route,
        "stationary_aot_source_package": source_package,
        "admission_limits": limits,
        "aot_binary_directory": None
        if aot_directory is None
        else str(Path(aot_directory).resolve()),
        "cases": cases,
        "rows": rows,
        "summary": {
            "required_fp64_force_rows": len(rows),
            "required_semilocal_fp64_force_rows": sum(
                row["method"] in SEMILOCAL_FUNCTIONALS for row in rows
            ),
            "statically_blocked_rows": blocked,
            "rows_passing_static_stationary_caps": len(rows) - blocked,
            "scientific_qualification": "NOT_RUN",
        },
    }


def _clean_git_sha(
    repository: Path,
    *,
    ignored_path: Path | None = None,
) -> str:
    tagged = subprocess.run(
        ["git", "ls-files", "-v", "--", "."],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    hidden = [
        line[2:] for line in tagged if line and (line[0] == "S" or line[0].islower())
    ]
    if hidden:
        raise RuntimeError(
            "capacity report rejects assume-unchanged/skip-worktree paths: "
            + ", ".join(hidden)
        )
    status_command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
    ]
    if ignored_path is not None:
        resolved_ignored = Path(ignored_path).resolve()
        if resolved_ignored.is_relative_to(repository):
            relative = resolved_ignored.relative_to(repository).as_posix()
            status_command.append(f":(exclude,literal){relative}")
    status = subprocess.run(
        status_command,
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("capacity report requires a clean Git worktree")
    return _git_head(repository)


def _report_output_exemption(
    repository: Path,
    output_path: Path,
    *,
    aot_directory: Path | None = None,
) -> Path | None:
    """Allow only an untracked JSON report to be ignored inside the checkout."""

    output_path = Path(output_path).resolve()
    if aot_directory is not None:
        aot_path = _validate_aot_evidence_directory(aot_directory)
        assert aot_path is not None
        if output_path.is_relative_to(aot_path.resolve()):
            raise ValueError("capacity report output must not overlap AOT evidence")
    if output_path.exists() and output_path.stat().st_nlink != 1:
        raise ValueError("capacity report output must not be a hard link")
    if not output_path.is_relative_to(repository):
        return None
    if output_path.suffix.lower() != ".json":
        raise ValueError("in-checkout capacity report output must be a JSON file")
    relative = output_path.relative_to(repository).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise ValueError("capacity report output must not replace a tracked file")
    return output_path


def build_report(
    repository: Path,
    *,
    aot_directory: Path | None = None,
) -> dict[str, Any]:
    """Build a report whose source identity is the clean tool-checkout HEAD."""

    repository = Path(repository).resolve()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    _assert_local_imports(require_fresh=True)
    return _build_report(
        repository,
        source_sha=_clean_git_sha(repository),
        aot_directory=aot_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=SOURCE_REPOSITORY,
        help="checkout containing this tool; other repositories are rejected",
    )
    parser.add_argument(
        "--aot-directory",
        type=Path,
        help="optionally verify the actual sm_120 packaged s/p/d binaries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output_path = None if args.output is None else args.output.resolve()
    if output_path is None:
        payload = build_report(repository, aot_directory=args.aot_directory)
    else:
        if repository != SOURCE_REPOSITORY:
            raise ValueError(
                "capacity report must run against the checkout containing this tool"
            )
        _assert_local_imports(require_fresh=True)
        exemption = _report_output_exemption(
            repository,
            output_path,
            aot_directory=args.aot_directory,
        )
        payload = _build_report(
            repository,
            source_sha=_clean_git_sha(repository, ignored_path=exemption),
            aot_directory=args.aot_directory,
        )
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
