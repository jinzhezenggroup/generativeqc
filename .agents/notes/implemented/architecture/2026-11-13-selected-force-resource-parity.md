# Decision: admit strict resources by selected force precision

Status: implemented; endpoint qualification recorded below
Date: 2026-11-13

## Problem

The [execution-parity audit](../../proposed/2026-11-05-standard-fast-execution-parity.md)
identified budget-presence exclusions on optional numeric fleet reservations
and large nonlocal force tiles. A nonzero allowance is not evidence that a
particular force consumer selects FP32. Conversely, FP64 SCF does not imply
FP64 derivatives: strict-density Fast can select mixed geometry independently.

## Decision

Make the existing complete-model gradient qualification consume the actual
stationarity controls as well as budget, shape and linked typed entry point.
Prepared execution, descriptor capacity, optional fleet reservation and force
tile selection use that same decision. Migrate all internal callers directly.

Retain the fleet reservation's existing FP64 SCF/provider/model domain. Within
that domain, budget-bearing owners with FP64 derivatives use the same bound
as zero-budget owners. An actually qualified mixed gradient still declines
this optional strict-only reservation before constructing a promise. Its
existing bounded mixed-capacity path remains authoritative. Do not pretend
that this change counts mixed scratch in the strict fleet proof.

Large eligible FP64 nonlocal owners use the same automatic force tile whether
or not their descriptor carries an application budget. Explicit requests,
device/shape/model checks, public resource promises, memory headroom and the
256-point allocation-failure fallback remain unchanged. No SCF threshold,
Hamiltonian, mixed arithmetic domain or observable audit is changed.

## Remaining differences

This is not a precision-only parity claim. The facade still resolves an FP64
model before constructing a differently configured Fast candidate. Removing
that control-plane duplication requires separating immutable model resolution
from execution admission; mutating precision after capability validation is
not a safe shortcut. Fixed-anchor code is shared, but strict automatic
profitability admission remains off pending its independent complete-endpoint
qualification. FP32 proposals, source screening and mandatory independent
audits remain explicit numerical-policy differences, not general execution
optimizations to copy across profiles.

## Evidence

The host reservation test executes the production function and real ledger
with independent deterministic capacity leaves. It checks Direct/DF,
small/large shapes, paired zero/nonzero-budget bounds, exact-fit/one-byte-short
limits, mixed-gradient rejection, unsupported arithmetic and release of all
unused promises. Companion tests check invalid stationarity controls and the
complete-model qualification. Leaf stand-ins are not numerical GPU evidence.

The opt-in public CUDA regression compares Standard/Fast FP64 DF hybrid
cold, frozen and moved energy/force endpoints against independent PySCF
moving-grid observables with unchanged scientific controls. Device results
and build/cache provenance must remain distinguished from the host proof;
no timing or general speedup is inferred from this structural change.

The focused host suite passes 402 tests, the changed production translation
unit compiles with the actual Release/CUDA build flags, and Sphinx HTML passes
with warnings as errors. The host policy queries use the independently
identified baseline library; the resource probes compile the changed source.

The first scheduled baseline harness check (n1, job 17305) fails closed because
the native-only library deployment lacks the packaged `pbe0_rks_spd` force
artifact. A linked native profile does not replace that separate public DF
consumer. Existing archived artifacts have different authenticated contracts
and are not relabeled or borrowed. Build the matching artifact with
`GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON` and the normal manifest target;
keep its manifest and shared library beside the selected core library.

With that separately compiled Release/sm120 artifact, job 17306 passes all
five selected public DF PBE0 oracle cases, including the new cold/frozen/moved
FP64 parity case. Core library SHA256 is
`9a50430e953fc7322edde32eaf63ec414a22a7b072b17eb2c9e0ba99325d56e8`;
artifact SHA256 is
`1f916a9727542c24730bb357ffb0a3c283f3288bd386923aa905a405fd470866`.
This checks the new harness and unchanged numerical consumer against the
baseline, not the subsequently rebuilt core. Failed and successful attempts
remain distinct under ignored `.artifacts/fast-parity-20261113/`.

### Rebuilt-source qualification

The normal Release/sm120 core completes its full build and a final incremental
identity rebuild with verified sccache 0.16.0. Native stationary profiles are
enabled; fast-compile and test hooks are disabled. The resulting core SHA256 is
`8aac3eedc52d08934755e26d759ddc1e9f71781dd66cf112ea2e34e195c59e8d`.
Its source identity, independently recomputed from 2,127 inputs in both the
checkout and deployment, is
`5a3cd69f652290766e36d230a14b9a9907241f34feeb29c7f5a5f5ace2855447`.
Compiler commands invoke the explicit sccache launcher; before/after cache
statistics and complete build logs are retained, without clearing the cache.
The focused host suite rerun against this rebuilt library passes all 402 tests.

Finite n1 Slurm job 17307 verifies that deployed identity and the actual
RTX 5090 device, Release build and disabled fast-compile flags. The new
matched-control FP64 cold/frozen/moved DF parity regression passes. The four
existing PBE0 mixed-gradient cases fail at their DF-Python qualification-name
assertion: this native-enabled core returns `pbe0-rks-budget-fp32-v1` with the
native schema-v3 work receipt, while those tests require the Python DF
`pbe0-df-rks-budget-fp32-v1` schema-v4 receipt. This is **one passed, four failed**,
not a green five-case run. It must not be confused with baseline job 17306.
The native and Python receipts describe different mixed consumers; they are
not relabeled as one another, and no scientific or work-accounting assertion
is removed to hide the mismatch.

A separate finite job 17308 reruns those four native mixed endpoints with
independent PySCF oracles and all their original numerical gates, without
requiring the Python-consumer receipt. All four pass: energy error below
`1e-7` Eh, strict/mixed maximum force error below `1e-6`/`1e-5` Eh/Bohr,
mixed-versus-strict force RMS below `3e-6` Eh/Bohr, and total force components
below `2e-8` Eh/Bohr. Observed maximum mixed energy/force errors are
`3.129e-8` Eh and `8.999e-8` Eh/Bohr. This diagnoses the receipt-contract failure;
it does not turn the failed pytest cases green or prove their schema-v4 work
assertions for the native consumer.

Logs, library/artifact hashes and JUnit reports are retained in ignored
`.artifacts/fast-parity-20261113/`, including `rebuilt-host-tests.log`,
`rebuilt-gpu-test.log` and `rebuilt-native-mixed-diagnostic.log`. The large
nonlocal force-tile branch and fleet reservation have host admission/capacity
coverage, not a new measured GPU peak-memory, profitability or large-shape
numerical qualification. There is no full execution-parity or speedup claim.

## Revisit when

Mixed fleet storage has a complete selected-operator proof, immutable model
resolution has its own ownership boundary, or matched strict-anchor endpoints
qualify a shared automatic profitability rule. None is implied by a strict
returned state or by a zero mixed-work count on a frozen replay.

## Subsequent engineering closure

The [common construction and anchor-selection decision](2026-11-13-standard-fast-common-construction-and-anchor-selection.md)
supersedes the remaining constructor and anchor-default differences recorded
here. Its refreshed-master source and qualification are separate from the
earlier identities and failed/successful GPU attempts above.
