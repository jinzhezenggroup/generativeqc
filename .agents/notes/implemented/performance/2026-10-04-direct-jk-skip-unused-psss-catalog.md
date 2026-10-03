# Decision: omit unused resident-PSSS catalogs from Direct J/K preparation

Status: implemented
Date: 2026-10-04

## Problem

The native Direct J/K provider calls the shared Direct-HF topology packer with
full shell/AO metadata and an explicit spherical-to-Cartesian transform. That
also constructed Direct-HF's resident-PSSS ket indices and task catalog. The
Direct J/K provider never uploads or consumes these two vectors: its generated
and bounded shell owners construct their own schedules. The vectors were only
included in its host preparation diagnostic before being discarded.

For s/p shell counts S/P and the compiler's T-thread resident schedule, the
unused catalog contains S*P*ceil(S*(S+1)/(2*T)) descriptors per system. This
quartic preparation work is separate from the actual screened integral work.
The issue was found while auditing a CUDA LDA preliminary-density provider for
WB97M-V cold starts: an unused preparation catalog should not become a required
second-owner resource allowance.

## Decision and invariants

Add an explicit `ResidentPsssPolicy` argument to the shared packer. Its default
builds the existing Direct-HF metadata. Only the Direct J/K caller selects
`Skip`, avoiding the PSSS classification scan, temporary bra list, ket list and
task construction. Shell-pair indices, primitive offsets, public AO expansion,
warm input validation and the direct transform remain unchanged.

Do not use `matrix_direct=true` as a substitute: that existing switch also
omits the spherical-to-Cartesian transform needed by this caller. Do not remove
the default catalogs globally, because the ordinary Direct-HF consumer uses
them. This change does not modify a device kernel, screening rule, integral
formula, AO cutoff, SCF control or any numerical gate.

The existing host preparation diagnostic continues to sum actual vector
capacities, so skipped catalogs contribute zero without changing the meaning
of that field. The general KS resource planner remains conservatively bounded;
this patch does not claim a new public preliminary-provider or budget contract.

## Validation scope

Host regressions compare the complete packed topology and warm payload against
the default path for a multi-item restricted batch and a larger unrestricted
packing case. Both contain spherical d functions, so losing the direct transform
cannot pass. They check the retained legacy task/ket counts and zero allocation
of both omitted vectors. Existing HF resource-layout cases protect the default
consumer. Independent complete CUDA WB97M-V endpoints and native DFT regression
are required for qualification; packing work alone is not an endpoint speedup.

Build/cache, source, device, numerical and timing receipts are retained in
ignored `.artifacts/cold-pack-20261004/`. No endpoint timing claim is made before
the controlled runs finish.

The standalone host regression compiles the candidate topology and test sources
with `-DNDEBUG`, linking the existing qualified library only for unchanged
normalization/resource-layout helpers. All cases pass. The larger constructed
325-AO case retains 532,480 legacy task descriptors and 8,256 ket indices,
occupying 12,648,448 bytes of vector capacity; the skipped catalogs allocate
zero bytes. The two-item small case retains four legacy tasks and six ket
indices. These are actual constructed packing counts, not molecular timings.
Replacing only the test's `Skip` request with `Build` fails the zero-catalog
assertion under the same Release flags, providing a negative control.

The test now keeps assertions enabled in Release and uses the current
`cuda_execution` namespace for the pre-existing transform checks. Its first full
build attempt failed because the new test included a private generated schedule
header absent from its include path. The test instead checks total pair coverage
independently of the schedule's chunk width. That failed attempt is retained;
full-library/device qualification remains separate from the host probe.

The subsequent full build exposed an existing standalone DFT test linkage gap:
the resident-grid test and borrowed-grid XC validation require
`cuda_quadrature.cu`, which that executable omitted. Add that source and its
code-generation dependency to the test target. This affects only the test
executable, not production ownership or numerical behavior. The failed link
receipt is retained separately from the successful build.

After merging master `38fc52352`, the full Release/sm_120 build succeeds and
the host packing regression passes against the newly built library. All 452
C++/CUDA compiler commands use ccache 4.5.1; the final incremental build records
three hits and five misses in the shared-cache counters. Library SHA-256 is
`282889cd499f70ee700ded665dc81633e7b91b3361912e764aebe00835efe360` and the
standalone DFT executable is
`25a8092b8dcc56ea2882d859710a17f50405b7ce0bb78331787683d619894e82`.
Independent GPU qualification and complete endpoint comparisons are still
pending at this checkpoint.

The first real-device full DFT run (n2 Slurm 2183) exposed a pre-existing test
expectation: it subtracted all five doubles per resident grid point from an XC
arena that owned only four (xyz and partitioned weight). The molecular grid's
fifth value is the atomic weight, never owned by XC. Correct the exact resource
expectation without changing the allocation. The same test review also found
that the mixed-density acceptance loop included B3LYP despite the production
contract rejecting every functional ID above two. Exercise the existing
rejection/state-preservation test for both B3LYP and WB97M-V. These two test
corrections already existed in the larger integration branch; no production
precision gate is widened here. Keep the failed device receipt and rerun the
complete standalone test before claiming qualification.

## Revisit when

If Direct J/K gains a consumer of the legacy resident-PSSS catalog, that consumer
must explicitly restore preparation and include its actual memory/work cost.
The separately prepared generated/bounded schedules are not such a consumer.
