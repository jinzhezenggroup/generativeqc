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

With these test-only corrections, n2 Slurm 2185 passes the complete native DFT
regression and all seven independent WB97M-V energy/analytic-force, displaced
geometry and stale-state/failure-isolation cases (194.66 seconds). The library
hash above is unchanged; the corrected standalone executable is
`1d5d9e1c535bc69b4e4899e110ef8ff1a45df1210b3411e4390b6fcc26592c0c`.
The 1351-input source identity is
`aeb864b833898c11bb4bef40923bf3dae54b539aa27cb74e257751719a6dd748`, built from
master `38fc52352` plus this patch; the frozen test/source commit is `54a1d40e1`.
An intermediate deployment attempt (2184) passed native DFT but could not run
the Python force endpoints because the private compiler wrapper directory
lacked its `ptxas` link. That environment failure is retained separately and
does not count as numerical qualification. No device visibility was overridden.

The n5 RTX 5090 complete comparator first passes all ten 3-atom master/candidate
cold, priming and warm reference pairs at unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates.
Both paths take 15 cold iterations and one iteration per replay; force work
counts agree. These first-use cold numbers cannot establish a packing speedup:
force preparation takes 17.767120 s in the first baseline process but 2.328321 s
in the later candidate, whereas their SCF preparation is 0.433469/0.435216 s.
This is an existing persistent-cache asymmetry outside the changed packing
code. Retain the measurements, then run fresh processes in reversed order
after both caches have been exercised. Neither ccache nor JIT caches are
cleared. The 24-atom and reversed-order cache controls are pending at this
checkpoint; no complete-cold acceleration is claimed.

After merging the subsequent master `79418329e`, source `3a6ee7aea` is rebuilt
and separately qualified in n2 Slurm 2186. Full native DFT and all seven
independent endpoint cases pass again (194.84 seconds); the new AO-component
host suite reports 20 passes, with 34 device cases explicitly skipped by that
host-only command. The 1352-input identity is
`4eab0c4fa2bd533a216349604959dcbb334cfa57de014e2b914369c93b7605b3` and library
SHA-256 is `dc3fca86b9c44013268e6883f521b515a4a2afcf53ec8ea675d7825a0d2a62c1`.
The standalone native test hash is unchanged. All 452 compiler commands still
use ccache; the incremental build compiles one identity-bearing C++ object,
recorded as one miss, and reuses the remaining existing objects. These receipts
live separately in `.artifacts/master794-20261004/`. The ongoing n5 measurements
keep their frozen master-38 source/binaries; they are not relabeled to this
subsequent composition. The code is ready for review as removal of unconsumed
preparation work, with no claimed complete-endpoint timing improvement.

## Completed endpoint and persistent-cache controls

Finite n5 Slurm 1417 subsequently completes all twenty 3/24-atom
baseline/candidate cold/priming/warm E/F pairs. Slurm 1418 then repeats them in
fresh processes with reversed candidate-to-baseline order, after both persistent
caches have been exercised; neither ccache nor JIT caches is cleared. The
independent verifier uses explicit raising checks and passes all twenty pairs
in each campaign under `PYTHONOPTIMIZE=1`. Both allocations exit successfully.

The ordinary master path is used here, without the separate integration's
active-AO/indexed/seed experiments. Baseline source is master 38fc52352,
identity `11d7ee80dedea8b2a8a4f0010688d412d25017e87a74d59e3f02cdc749d7b3c3`,
library `26f73730003d5f146b8154d1572670f266ba71076e4eb2af57e19ed23781f87f`.
Candidate source is frozen 54a1d40e1/aeb864b8 with library 282889cd as recorded
above; these timings are not relabeled to the subsequent master794 build.

| Campaign | Atoms | Baseline complete cold (s) | Candidate complete cold (s) | Baseline warm median (s) | Candidate warm median (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial order | 3 | 27.193990 | 11.758068 | 1.190615 | 1.190061 |
| Initial order | 24 | 268.449402 | 268.497837 | 39.381974 | 39.748734 |
| Exercised caches, reversed order | 3 | 10.333119 | 10.355456 | 1.192410 | 1.192648 |
| Exercised caches, reversed order | 24 | 268.895760 | 267.954915 | 39.674858 | 39.378665 |

The 3-atom first-use difference disappears with comparable persistent-cache
state. At 24 atoms the two cold controls differ by only about 0.94 s out of
269 s in the second campaign, with no benefit in the first; these single cold
runs do not establish a repeatable endpoint speedup. Native cold takes 15 and
18 iterations at 3 and 24 atoms in both variants, versus reference 11 and 14;
every priming/warm call takes one iteration. Force work counts match between
variants, all reference XC components stay on GPU, and unchanged E/F gates
apply to every sample. The synthetic packing/allocation saving is real, but
no complete-endpoint acceleration is claimed.

The raw results, source receipts and independent summaries are retained in
`.artifacts/cold-pack-20261004/benchmark-{results,receipts}/` and
`cache-control-{results,receipts}/`, with summaries
`benchmark-verified-3-24.json` and `cache-control-verified-3-24.json`. The
cache-control launch script SHA-256 is
`da14f7aec18543af928662fcaf5867d26dc374568c7b8645fc5562004c80344a`.

## Revisit when

If Direct J/K gains a consumer of the legacy resident-PSSS catalog, that consumer
must explicitly restore preparation and include its actual memory/work cost.
The separately prepared generated/bounded schedules are not such a consumer.
