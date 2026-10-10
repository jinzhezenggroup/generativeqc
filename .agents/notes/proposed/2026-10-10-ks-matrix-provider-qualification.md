# Decision: qualify the existing KS matrix provider without production changes

Status: proposed
Date: 2026-10-10

## Problem

Merged #2058 prepared a shared matrix-library owner but its host doubles do not
establish CUDA arithmetic, masking, capture, or complete-SCF profitability.
The historical custom-kernel-only diagnosis in #1873 is superseded. This slice
qualifies current source; it does not implement provider-neutral ownership.

## Protocol

Compile the complete, unchanged `matrix_library.cpp`, `scf_matrix_kernels.cu`
and `runtime_support.cpp` with a separately linked device driver. Generate
non-symmetric column-major inputs in Python, with distinct system/spin blocks,
odd tails and finite sentinel outputs. NumPy long-double matrix multiplication
is the independent oracle, outside measured device work. Platforms without a
wider long-double mantissa must report that limitation rather than claim a
wide-precision oracle.

Cross 16/17 AO admission, odd 19/97 AO, singleton/multiple systems, ordinary and
one/two-spin products, physical operand broadcasting on either side, transpose,
scale, all-active/mixed/all-inactive masks, failed-item NaN poison and two graph
replays. Reset outputs before each replay. Native and library routes must each
agree with the oracle; agreement between routes alone is insufficient. Inactive
outputs must retain their sentinel, and poison must not reach another system.
The explicit library primitive probe at small shapes uses a separately prepared
97-AO owner; it is labelled forced diagnostic, not production admission. Separate
owner probes exercise actual 16/17-AO admission.

Record default owner selection, retained bytes and reset/reprepare lifetime.
Inject cuBLAS allocation failure and CUDA memory-query failure only at link-time
in the isolated driver, forwarding all other calls to the real CUDA library.
Never exhaust shared GPU memory to manufacture a resource failure. Capture
preparation rejection and each unsupported capture are retained negative rows.
Every expected case has one result; missing rows, unavailable devices, crashes,
nonfinite active values and mask mismatches prevent qualification PASS.

Complete endpoint pilots must compare identical source, basis/grid, precision,
geometry trajectory, stopping rules, and force request. RKS/UKS 96+ AO cold,
warm and moved E+F need independent PySCF energy/force/density/physical residual
and final-state gates. Preserve all SCF histories. Endpoint and matrix-phase
timing, actual semantic calls, prepared lifetime and memory are separate views.
Missing force or Graph support is INCOMPLETE, never energy-only E+F acceptance.
Clean endpoint timing and intrusive matrix event attribution are separate runs.

## Boundaries and current risks

The public calculation API has no forced KS provider or matrix-phase counter.
A controlled diagnostic build may use linker wrapping of the existing shared
primitive, with its interception verified and recorded. It must not edit runtime,
compiler, public API or other owners' files. If interception cannot be proven,
the paired endpoint gate stays blocked. No speedup can be inferred from primitive
timing, equal iteration counts or different work domains.

Source audit at `7f342546d887796e6a92a005ba033029ed73ce2a` shows that cuBLAS
branches do not consume `active`, whereas native branches do. The driver must
retain any measured failure. This audit does not prove that published KS results
are affected: KS uses singleton owners and other guards require investigation.
The runtime fix belongs to a separately coordinated owner, outside this slice.

## Identity and delivery gates

Receipts bind Git head, tracked blob/mode inventory, dirty inputs, tool sources,
compiler/cache versions and binaries, dependency files, built objects/executable,
loaded CUDA/cuBLAS libraries, driver/device/visibility, resource and workload
identity, fixed inputs and every raw result by SHA-256. No compile/RUNNING/smoke
or tool PR merge closes #1873. Actual qualification and independent final-head
review remain distinct from implementation, CI, and protected merge.

Endpoint pilots record the requested and actually selected libraries separately,
including their hashes, native source identity, profile diagnostics and Linux
process mappings. A clean checkout, matching canonical source identity and a
mapped selected library are prerequisites for numerical acceptance. A local
profile may select another binary; a requested path hash alone cannot establish
which binary executed. Retain a public endpoint row before reading its snapshot
or oracle so a postprocessing failure cannot erase the endpoint observation.

The executed `cases.txt` bytes must equal the canonical protocol regenerated
from the validated manifest, including provider, Graph, order and scale fields.
Bind this digest into the manifest, arithmetic receipt and build identity, and
recheck manifest/raw/protocol hashes when attaching identity. Manifest labels
and input hashes alone could attribute native/no-Graph output to a changed
library/Graph command. Mutation regressions retain this independent review
finding as a fail-closed gate.

## Rejected alternatives

Do not duplicate production arithmetic in a host probe, use a production formula
as its own oracle, silently ignore masked cases, weaken gates to obtain PASS,
change selection defaults, or borrow another issue's running instance.

## Device evidence, 2026-10-10

The complete standalone driver at clean source
`f6f9279f6d06909563abeef0b7b0ae767bd7fbc1` ran on a real H100 80GB HBM3
(sm_90), CUDA toolkit/runtime 12.9, driver API 12.8 and GCC 11.4, with verified
sccache 0.16.0. NumPy reported a 63-bit long-double mantissa. All 2,880 cases
returned observations and both executions were assessed: all 1,440 native cases
and 360 all-active library cases passed; all 1,080 mixed/inactive/failed-mask
library cases failed because inactive outputs lost their sentinel. This held
with and without Graph replay. The receipt correctly remains NOT_QUALIFIED.
The five owner admission/allocation/memory-query/capture/reset probes passed.

The successful driver process and its actually loaded library maps were retained.
Build identity SHA-256:
`a94c666685bb47b7fd9323bdb6fb0614f8327415487e15da9d4d55a85a93151f`;
driver SHA-256:
`b9e93b9dedf930719bb1f48193741e80b502cfd68ecdcf67fd89a752ceea35bc`;
raw device JSONL SHA-256:
`7c754d95fe43ff7dcf0f7590e6412b08ee5005ef6941c40c608fea0492ad5a11`.
These hashes bind this historical run; they do not identify later tool builds.
The submission counts in that run were calculated from requested shapes,
not observed vendor-call counters. New receipts label them as nominal/requested
and explicitly report that vendor submission counts were not observed.

The independent PySCF 2.14.0 / Libxc 7.0.0 cold RKS probe confirmed 96 AO,
energy -152.75863503292015 hartree and physical residual RMS 4.73e-12.
This single probe does not establish force acceptance, UKS acceptance, matched
provider endpoints, complete-SCF work counts or profitability. H100's current
shell-profile/force support and the missing paired-provider diagnostic seam
leave complete endpoint qualification INCOMPLETE. The shared primitive failure
does not alone prove an incorrect public KS endpoint; its runtime remediation
belongs to a separately coordinated production owner.

## Revisit when

Independent device and endpoint evidence is available, the shared adapter changes,
or a provider-neutral semantic owner exposes a stronger diagnostic seam.
